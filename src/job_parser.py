from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from openai import OpenAI

from . import env  # noqa: F401  # ensure .env loaded
from .models import Job, JobRequirements, JobSource

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Structured extraction prompt - more aggressive about extracting data
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
1. JOB TITLE: Look for patterns like "About the role: X", "Position: X", "We're hiring a X", or the most prominent job-like phrase. Common titles: Data Scientist, ML Engineer, AI Engineer, Software Engineer, Research Scientist. NEVER use "Unknown" - make your best guess.

2. COMPANY: Look for "at [Company]", "About [Company]", or domain names. Extract from phrases like "Join [Company]'s team".

3. SKILLS: Extract ALL technical terms mentioned. Look for:
   - Programming: Python, Java, JavaScript, C++, R, Scala, Go, Rust
   - ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras, Hugging Face, LangChain, OpenAI
   - Data: SQL, Pandas, NumPy, Spark, Hadoop, Airflow, dbt
   - Cloud: AWS, GCP, Azure, Docker, Kubernetes, MLflow
   - Domains: NLP, Computer Vision, LLM, Deep Learning, Reinforcement Learning

4. LEVEL: Infer from years of experience or explicit mentions:
   - 0-2 years = entry
   - 3-5 years = mid  
   - 5-8 years = senior
   - 8+ years = lead/principal

5. For tech_stack, include ALL technologies mentioned, even in passing.

IMPORTANT: Be aggressive - extract something for every field. Never leave title empty."""


def extract_company_from_url(url: str) -> Optional[str]:
    """Extract company name from job posting URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        
        # Common ATS patterns
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
        
        # Generic: extract from domain
        parts = hostname.replace("www.", "").split(".")
        if len(parts) >= 2:
            if parts[0] in ["careers", "jobs", "job", "work", "apply", "boards"]:
                return parts[1].replace("-", " ").title()
            else:
                return parts[0].replace("-", " ").title()
    except Exception:
        pass
    return None


def parse_job_with_openai(text: str, url: Optional[str] = None, job_id: Optional[str] = None) -> Job:
    """
    Use OpenAI to parse raw JD text into the Job schema.
    Uses structured JSON output for reliable extraction.
    """
    client = OpenAI()
    
    # Try to extract company from URL as hint
    url_company = extract_company_from_url(url) if url else None
    
    user_prompt = f"Parse this job posting:\n\n{text[:12000]}"
    if url_company:
        user_prompt = f"Company hint from URL: {url_company}\n\n{user_prompt}"
    
    job = None
    parsed_data = {}
    
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        import json
        parsed_data = json.loads(content)
        
        # Build JobRequirements
        requirements = JobRequirements(
            must_have=parsed_data.get("must_have_skills") or [],
            nice_to_have=parsed_data.get("nice_to_have_skills") or [],
            years_experience_gate=_format_years(
                parsed_data.get("years_experience_min"),
                parsed_data.get("years_experience_max")
            ),
            visa=parsed_data.get("visa_sponsorship"),
            onsite_requirement=parsed_data.get("remote_policy"),
        )
        
        # Build Job
        job = Job(
            id=job_id or "parsed_job",
            company=parsed_data.get("company") or url_company or "Unknown",
            title=parsed_data.get("title") or "Unknown",
            location=parsed_data.get("location"),
            remote_policy=parsed_data.get("remote_policy"),
            level=parsed_data.get("level"),
            source=JobSource(url=url, ats=_detect_ats(url)),
            requirements=requirements,
            stack=parsed_data.get("tech_stack") or [],
            responsibilities=parsed_data.get("responsibilities") or [],
            red_flags=parsed_data.get("red_flags") or [],
            evaluation_rubric=[],
            raw_text=text,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        
    except Exception as e:
        # Fallback to heuristic parsing
        job = Job(
            id=job_id or (url or "job_fallback"),
            company=url_company or "Unknown",
            title=_extract_title_heuristic(text) or "Unknown",
            location=_extract_location_heuristic(text),
            remote_policy=_extract_remote_heuristic(text),
            level=None,
            source=JobSource(url=url, ats=_detect_ats(url)),
            requirements=JobRequirements(),
            stack=[],
            responsibilities=[],
            red_flags=[],
            evaluation_rubric=[],
            raw_text=text,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
    
    # Always preserve the original URL
    if url:
        if not job.source:
            job.source = JobSource(url=url, ats=_detect_ats(url))
        else:
            job.source.url = url
    
    # Fill years requirement if missing
    if not job.requirements.years_experience_gate:
        job.requirements.years_experience_gate = _extract_years_heuristic(text)

    # Final company name fallback
    if not job.company or job.company.lower() == "unknown":
        if url_company:
            job.company = url_company
    
    if job_id:
        job.id = job_id
    
    return job


def _format_years(min_years: Optional[int], max_years: Optional[int]) -> Optional[str]:
    """Format years experience as a string."""
    if min_years is not None and max_years is not None:
        return f"{min_years}-{max_years} years"
    elif min_years is not None:
        return f"{min_years}+ years"
    return None


def _detect_ats(url: Optional[str]) -> Optional[str]:
    """Detect ATS from URL."""
    if not url:
        return None
    if "greenhouse.io" in url:
        return "greenhouse"
    if "lever.co" in url:
        return "lever"
    if "ashbyhq.com" in url:
        return "ashby"
    if "myworkdayjobs.com" in url:
        return "workday"
    if "linkedin.com" in url:
        return "linkedin"
    if "indeed.com" in url:
        return "indeed"
    return None


def _extract_title_heuristic(text: str) -> Optional[str]:
    """Try to extract job title from first few lines."""
    lines = text.strip().split("\n")[:5]
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            if any(kw in line.lower() for kw in ["engineer", "developer", "scientist", "manager", "analyst", "designer"]):
                return line
    return None


def _extract_location_heuristic(text: str) -> Optional[str]:
    """Try to extract location from text."""
    patterns = [
        r"(?:location|based in|office in)[:\s]+([^,\n]+(?:,\s*[A-Z]{2})?)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:CA|NY|TX|WA|MA|IL|CO|GA|NC|VA|FL|OR|AZ|PA|OH))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if "remote" in text.lower()[:500]:
        return "Remote"
    return None


def _extract_remote_heuristic(text: str) -> Optional[str]:
    """Detect remote policy from text."""
    lower = text.lower()[:1000]
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
    """Extract years of experience requirement from text."""
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
