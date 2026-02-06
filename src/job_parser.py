from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import unquote, urlparse

from openai import OpenAI

from . import env  # noqa: F401  # ensure .env loaded
from .models import Job, JobRequirements, JobSource

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PARSE_SYSTEM_PROMPT = """You are an expert job posting parser. Extract structured data even from messy web scrapes.

RESPOND ONLY WITH VALID JSON:
{
  "company": "Company Name",
  "title": "Full Job Title",
  "location": "City, State or Remote",
  "remote_policy": "remote | hybrid | onsite",
  "level": "intern | entry | mid | senior | lead | principal | staff",
  "years_experience_min": 0,
  "must_have_skills": ["Python", "SQL", "Machine Learning"],
  "nice_to_have_skills": ["Spark", "Kubernetes"],
  "tech_stack": ["Python", "TensorFlow", "PyTorch", "AWS"],
  "responsibilities": ["Build ML models", "Deploy to production"],
  "visa_sponsorship": "yes | no | unknown"
}

CRITICAL EXTRACTION RULES:
1. JOB TITLE: Find a concrete role title (e.g., Data Scientist, ML Engineer, Software Engineer). Never output Unknown.
2. COMPANY: Infer from company/about sections or domain hints.
3. SKILLS: Include all explicit technical skills, tools, and frameworks.
4. LEVEL: Infer from years requirement and role wording.
5. Prefer precise extraction over generic wording.
"""

TITLE_HINT_WORDS = {
    "engineer",
    "scientist",
    "developer",
    "analyst",
    "manager",
    "architect",
    "researcher",
    "specialist",
    "consultant",
    "intern",
    "director",
    "principal",
}

TITLE_BLACKLIST = {
    "unknown",
    "job",
    "jobs",
    "career",
    "careers",
    "home",
    "apply",
    "apply now",
    "opportunity",
    "new jobs",
}

LOCATION_NOISE_TERMS = {
    "equal opportunity",
    "applicants",
    "without regard",
    "race",
    "religion",
    "gender",
    "sexual orientation",
    "national origin",
    "disability",
    "protected veteran",
    "job description",
    "requirements",
    "qualifications",
    "responsibilities",
    "summary",
}


def extract_company_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        if "greenhouse.io" in hostname:
            match = re.search(r"greenhouse\.io/([^/]+)", url)
            if match:
                return match.group(1).replace("-", " ").title()

        if "lever.co" in hostname:
            match = re.search(r"lever\.co/([^/]+)", url)
            if match:
                return match.group(1).replace("-", " ").title()

        if "ashbyhq.com" in hostname:
            match = re.search(r"ashbyhq\.com/([^/]+)", url)
            if match:
                return match.group(1).replace("-", " ").title()

        if "myworkdayjobs.com" in hostname:
            match = re.search(r"([^.]+)\.wd\d+\.myworkdayjobs", hostname)
            if match:
                return match.group(1).replace("-", " ").title()

        if "linkedin.com" in hostname or "indeed.com" in hostname:
            return None

        parts = hostname.replace("www.", "").split(".")
        if len(parts) >= 2:
            if parts[0] in ["careers", "jobs", "job", "work", "apply", "boards"]:
                return parts[1].replace("-", " ").title()
            return parts[0].replace("-", " ").title()
    except Exception:
        pass
    return None


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _clean_title_candidate(text: str) -> Optional[str]:
    candidate = _normalize_spaces(text)
    if not candidate:
        return None
    candidate = candidate.replace("|", " ").replace("  ", " ").strip(" -:\t")
    candidate = re.sub(r"\b(remote|hybrid|onsite|full[- ]?time|part[- ]?time)\b", "", candidate, flags=re.I)
    candidate = _normalize_spaces(candidate)
    if len(candidate) < 4 or len(candidate) > 90:
        return None
    if candidate.lower() in TITLE_BLACKLIST:
        return None
    if not any(word in candidate.lower() for word in TITLE_HINT_WORDS):
        return None
    return candidate


def _looks_like_title_text(text: str) -> bool:
    low = _normalize_spaces(text).lower()
    return any(word in low for word in TITLE_HINT_WORDS)


def _clean_location_candidate(text: str) -> Optional[str]:
    candidate = _normalize_spaces(text).strip(" -|\t")
    if not candidate:
        return None
    if len(candidate) < 2 or len(candidate) > 90:
        return None
    low = candidate.lower()
    if any(token in low for token in LOCATION_NOISE_TERMS):
        return None
    if re.search(r"\b(we seek|our company|this role)\b", low):
        return None
    if _looks_like_title_text(candidate) and "," not in candidate and "remote" not in low:
        return None
    return candidate


def _extract_title_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    segments = [s for s in path.split("/") if s and s.lower() not in {"jobs", "job", "careers", "career", "apply"}]
    for segment in reversed(segments):
        cleaned = re.sub(r"[-_]+", " ", segment)
        cleaned = re.sub(r"\b\d+\b", "", cleaned)
        candidate = _clean_title_candidate(cleaned)
        if candidate:
            return candidate
    return None


def _extract_title_patterns(text: str) -> Optional[str]:
    patterns = [
        r"(?im)^\s*(?:job\s*title|title|position|role)\s*[:\-]\s*(.{4,90})$",
        r"(?im)^\s*about the role\s*[:\-]\s*(.{4,90})$",
        r"(?im)^\s*we(?:'re| are)\s+hiring\s+(?:for\s+)?(?:a|an)?\s*(.{4,90})$",
        r"(?im)^\s*opening\s*[:\-]\s*(.{4,90})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = _clean_title_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def _extract_title_from_lines(text: str) -> Optional[str]:
    best = None
    best_score = -999
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines[:120]:
        candidate = _clean_title_candidate(line)
        if not candidate:
            continue
        score = 0
        low = candidate.lower()
        if any(word in low for word in TITLE_HINT_WORDS):
            score += 4
        if 2 <= len(candidate.split()) <= 8:
            score += 2
        if "senior" in low or "lead" in low or "principal" in low:
            score += 1
        if "page title" in line.lower() or "og title" in line.lower():
            score += 3
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _extract_title_heuristic(text: str, url: Optional[str] = None) -> Optional[str]:
    candidates = [
        _extract_title_patterns(text),
        _extract_title_from_lines(text),
        _extract_title_from_url(url),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _infer_generic_title(text: str) -> str:
    low = (text or "").lower()
    candidates = [
        ("data scientist", "Data Scientist"),
        ("machine learning engineer", "Machine Learning Engineer"),
        ("ml engineer", "Machine Learning Engineer"),
        ("ai engineer", "AI Engineer"),
        ("research scientist", "Research Scientist"),
        ("data engineer", "Data Engineer"),
        ("software engineer", "Software Engineer"),
        ("backend engineer", "Backend Engineer"),
        ("frontend engineer", "Frontend Engineer"),
        ("full stack engineer", "Full Stack Engineer"),
        ("analyst", "Data Analyst"),
    ]
    for key, label in candidates:
        if key in low:
            return label
    return "Software Engineer"


def _extract_company_from_text(text: str) -> Optional[str]:
    patterns = [
        r"(?im)^\s*company\s*[:\-]\s*(.{2,70})$",
        r"(?im)^\s*about\s+([A-Z][A-Za-z0-9&,\-. ]{2,60})$",
        r"(?im)join\s+([A-Z][A-Za-z0-9&,\-. ]{2,60})\s+(?:team|as)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        company = _normalize_spaces(match.group(1))
        if 2 <= len(company) <= 60 and company.lower() not in TITLE_BLACKLIST:
            return company
    return None


def _parse_posted_date_candidate(raw: str) -> Optional[str]:
    value = (raw or "").strip()
    if not value:
        return None

    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        return dt.isoformat()
    except Exception:
        pass

    for fmt in [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    ]:
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            continue
    return None


def _extract_posted_date_heuristic(text: str) -> Optional[str]:
    raw = text or ""
    patterns = [
        r"(?im)^\s*jobposting date posted:\s*([^\n]+)$",
        r"(?im)\bdate posted\s*[:\-]\s*([A-Za-z0-9,\-/ ]{6,40})\b",
        r"(?im)\bposted on\s*[:\-]?\s*([A-Za-z0-9,\-/ ]{6,40})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw)
        if not m:
            continue
        parsed = _parse_posted_date_candidate(m.group(1))
        if parsed:
            return parsed

    rel = re.search(r"(?im)\bposted\s+(\d{1,3})\s+days?\s+ago\b", raw)
    if rel:
        try:
            days = int(rel.group(1))
            dt = datetime.now(timezone.utc) - timedelta(days=days)
            return dt.isoformat()
        except Exception:
            pass
    return None


def _is_weak_title(title: Optional[str]) -> bool:
    if not title:
        return True
    cleaned = _normalize_spaces(title).lower()
    if not cleaned or cleaned in TITLE_BLACKLIST:
        return True
    if cleaned.startswith("unknown"):
        return True
    if not any(word in cleaned for word in TITLE_HINT_WORDS):
        return True
    return False


def parse_job_with_openai(text: str, url: Optional[str] = None, job_id: Optional[str] = None) -> Job:
    client = OpenAI()
    url_company = extract_company_from_url(url) if url else None
    heuristic_title = _extract_title_heuristic(text, url=url)
    heuristic_company = _extract_company_from_text(text) or url_company

    user_prompt = f"Parse this job posting:\n\n{text[:14000]}"
    if heuristic_company:
        user_prompt = f"Company hint from source: {heuristic_company}\n\n{user_prompt}"
    if heuristic_title:
        user_prompt = f"Title hint from source: {heuristic_title}\n\n{user_prompt}"

    parsed_data: dict = {}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed_data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        parsed_data = {}

    requirements = JobRequirements(
        must_have=parsed_data.get("must_have_skills") or [],
        nice_to_have=parsed_data.get("nice_to_have_skills") or [],
        years_experience_gate=_format_years(
            parsed_data.get("years_experience_min"),
            parsed_data.get("years_experience_max"),
        ),
        visa=parsed_data.get("visa_sponsorship"),
        onsite_requirement=parsed_data.get("remote_policy"),
    )

    title = parsed_data.get("title") or heuristic_title or "Unknown"
    company = parsed_data.get("company") or heuristic_company or "Unknown"
    location = parsed_data.get("location") or _extract_location_heuristic(text)
    location = _clean_location_candidate(location) if location else None

    # Cross-field repair: model sometimes puts title-like text into location.
    if _is_weak_title(title) and location and _looks_like_title_text(location):
        recovered_title = _clean_title_candidate(location)
        if recovered_title:
            title = recovered_title
            location = None

    if _is_weak_title(title):
        title = heuristic_title or _extract_title_from_url(url) or _infer_generic_title(text)
    if not company or company.lower() == "unknown":
        company = heuristic_company or "Unknown"

    job = Job(
        id=job_id or "parsed_job",
        company=company,
        title=title,
        location=location,
        remote_policy=parsed_data.get("remote_policy") or _extract_remote_heuristic(text),
        level=parsed_data.get("level"),
        source=JobSource(url=url, ats=_detect_ats(url), source_type=_detect_source_type(url)),
        requirements=requirements,
        stack=parsed_data.get("tech_stack") or [],
        responsibilities=parsed_data.get("responsibilities") or [],
        red_flags=parsed_data.get("red_flags") or [],
        evaluation_rubric=[],
        raw_text=text,
        posted_at=_extract_posted_date_heuristic(text),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    if not job.requirements.years_experience_gate:
        job.requirements.years_experience_gate = _extract_years_heuristic(text)
    if job_id:
        job.id = job_id
    return job


def _format_years(min_years: Optional[int], max_years: Optional[int]) -> Optional[str]:
    if min_years is not None and max_years is not None:
        return f"{min_years}-{max_years} years"
    if min_years is not None:
        return f"{min_years}+ years"
    return None


def _detect_ats(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if "greenhouse.io" in url or "grnh.se" in url:
        return "greenhouse"
    if "lever.co" in url:
        return "lever"
    if "ashbyhq.com" in url:
        return "ashby"
    if "myworkdayjobs.com" in url or "workdayjobs.com" in url or "apply.workday.com" in url:
        return "workday"
    if "smartrecruiters.com" in url or "smrtr.io" in url:
        return "smartrecruiters"
    if "jobvite.com" in url:
        return "jobvite"
    if "workable.com" in url:
        return "workable"
    if "bamboohr.com" in url:
        return "bamboohr"
    if "icims.com" in url:
        return "icims"
    if "linkedin.com" in url:
        return "linkedin"
    if "indeed.com" in url:
        return "indeed"
    return None


def _detect_source_type(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    aggregators = (
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
        "ziprecruiter.com",
        "wellfound.com",
        "angel.co",
        "builtin.com",
        "dice.com",
        "hired.com",
    )
    if any(host == dom or host.endswith(f".{dom}") for dom in aggregators):
        return "aggregator"
    return "direct"


def _extract_location_heuristic(text: str) -> Optional[str]:
    labeled_patterns = [
        r"(?im)^\s*(?:location|office location|based in)\s*[:\-]\s*([^\n]{2,90})$",
        r"(?im)^\s*(?:location|office location|based in)\s+([^\n]{2,90})$",
    ]
    for pattern in labeled_patterns:
        match = re.search(pattern, text or "")
        if not match:
            continue
        candidate = _clean_location_candidate(match.group(1))
        if candidate:
            return candidate

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in lines[:140]:
        candidate = _clean_location_candidate(line)
        if not candidate:
            continue
        if re.match(r"^[A-Z][A-Za-z .'\-]+,\s*[A-Z]{2}(?:,\s*(?:USA|US|United States))?$", candidate):
            return candidate
        if re.match(r"^[A-Z][A-Za-z .'\-]+,\s*[A-Z][A-Za-z .'\-]+,\s*[A-Z][A-Za-z .'\-]+$", candidate):
            return candidate
        if re.match(r"^(Remote|Hybrid|On[- ]?site)$", candidate, re.I):
            return candidate.title().replace("On-Site", "Onsite")

    if "remote" in (text or "").lower()[:1200]:
        return "Remote"
    return None


def _extract_remote_heuristic(text: str) -> Optional[str]:
    lower = text.lower()[:1200]
    if "fully remote" in lower or "100% remote" in lower:
        return "remote"
    if "hybrid" in lower:
        return "hybrid"
    if "on-site" in lower or "onsite" in lower or "in-office" in lower:
        return "onsite"
    if "remote" in lower:
        return "remote"
    return None


def _extract_years_heuristic(text: str) -> Optional[str]:
    patterns = [
        r"(\d+)\s*\+?\s*years",
        r"(\d+)\s*-\s*(\d+)\s*years",
        r"at least\s+(\d+)\s+years",
        r"minimum\s+of\s+(\d+)\s+years",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if len(match.groups()) == 2 and match.group(2):
            return f"{match.group(1)}-{match.group(2)} years"
        return f"{match.group(1)}+ years"
    return None
