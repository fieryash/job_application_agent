from __future__ import annotations

import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from . import env  # noqa: F401  # ensure .env loaded
from .db import get_job_by_url, job_exists_by_url
from .job_parser import parse_job_with_openai
from .models import Job, JobSource

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)

SEARCH_TIMEOUT = 10  # seconds
FETCH_TIMEOUT = 8  # seconds
MAX_WORKERS = 5

DIRECT_JOB_DOMAINS = {
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "greenhouse.io",
    "grnh.se",
    "jobs.lever.co",
    "lever.co",
    "jobs.ashbyhq.com",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "workdayjobs.com",
    "myworkdaysite.com",
    "apply.workday.com",
    "icims.com",
    "smartrecruiters.com",
    "jobs.smartrecruiters.com",
    "smrtr.io",
    "bamboohr.com",
    "jobvite.com",
    "jobs.jobvite.com",
    "talent.jobvite.com",
    "workable.com",
    "apply.workable.com",
    "recruitee.com",
    "teamtailor.com",
    "personio.com",
    "personio.de",
    "jazzhr.com",
    "breezy.hr",
    "applicantpro.com",
    "hiringthing.com",
    "pinpointhq.com",
    "comeet.co",
    "rippling.com",
    "hire.dover.com",
    "dover.com",
    "successfactors.com",
    "sapsf.com",
    "jobs2web.com",
    "oraclecloud.com",
    "taleo.net",
    "brassring.com",
    "kenexa.com",
    "paylocity.com",
    "recruiting.paylocity.com",
    "ukg.com",
    "ultipro.com",
    "recruiting.ultipro.com",
    "paycomonline.net",
    "paycor.com",
    "careers.paycor.com",
    "amazon.jobs",
    "careers.microsoft.com",
    "careers.google.com",
    "jobs.apple.com",
}

AGGREGATOR_DOMAINS = {
    "indeed.com",
    "linkedin.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "builtin.com",
    "dice.com",
    "hired.com",
    "wellfound.com",
    "angel.co",
}

ALLOWED_DOMAINS = DIRECT_JOB_DOMAINS | AGGREGATOR_DOMAINS

BLOCKED_DOMAINS = {
    "geeksforgeeks.org",
    "w3schools.com",
    "amazon.com",
    "github.com",
    "medium.com",
    "towardsdatascience.com",
    "youtube.com",
    "udemy.com",
    "coursera.org",
    "edx.org",
    "freecodecamp.org",
    "stackoverflow.com",
    "stackexchange.com",
    "developer.mozilla.org",
}

TRACKING_QUERY_PREFIXES = ("utm_", "gclid", "fbclid", "trk", "ref", "source")
KEEP_QUERY_KEYS = {
    "jk",
    "vjk",
    "jobid",
    "job_id",
    "gh_jid",
    "gh_src",
    "lever-via",
    "gh_src",
    "gh_jid",
    "url",
    "target",
    "dest",
    "destination",
    "redirect",
    "redirect_url",
    "job_url",
    "apply",
}

JOB_URL_HINTS = (
    "/jobs/",
    "/careers/",
    "/career/",
    "/positions/",
    "/job/",
    "/role/",
    "/opening",
    "/opportunities",
    "/apply",
    "gh_jid=",
)

JOB_TEXT_HINTS = (
    "responsibilities",
    "qualifications",
    "requirements",
    "job description",
    "what you will do",
    "what you'll do",
    "about the role",
    "equal opportunity employer",
    "we are looking for",
    "benefits",
)

PRIORITY_SITE_FILTERS = [
    "site:myworkdayjobs.com",
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.ashbyhq.com",
    "site:jobs.smartrecruiters.com",
]

_URL_RESOLUTION_CACHE: Dict[str, Optional[str]] = {}


def _normalize_host(url: str) -> str:
    host = urlparse(url).netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_in(host: str, domains: Set[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _is_exact_aggregator_job_url(url: str) -> bool:
    host = _normalize_host(url)
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    if _host_in(host, {"indeed.com"}):
        return path.startswith("/viewjob") and ("jk" in query or "vjk" in query)
    if _host_in(host, {"linkedin.com"}):
        return "/jobs/view/" in path
    if _host_in(host, {"glassdoor.com"}):
        return "/job-listing/" in path or "/job/" in path
    if _host_in(host, {"ziprecruiter.com", "wellfound.com", "angel.co"}):
        return "/jobs/" in path
    if _host_in(host, {"builtin.com", "hired.com"}):
        return "/job/" in path
    if _host_in(host, {"dice.com"}):
        return "/job-detail/" in path
    return False


def _is_blocked_url(url: str) -> bool:
    host = _normalize_host(url)
    return _host_in(host, BLOCKED_DOMAINS)


def _canonicalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        if scheme not in {"http", "https"}:
            return None
        host = parsed.netloc.lower()
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        raw_query = parse_qs(parsed.query, keep_blank_values=False)
        kept = {}
        for key, value in raw_query.items():
            lk = key.lower()
            if lk in KEEP_QUERY_KEYS:
                kept[key] = value
                continue
            if any(lk.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
                continue
        query = urlencode(kept, doseq=True)
        return urlunparse((scheme, host, path, "", query, ""))
    except Exception:
        return None


def _is_allowed_url(url: str) -> bool:
    if not url:
        return False
    if _is_blocked_url(url):
        return False

    host = _normalize_host(url)
    path = urlparse(url).path.lower()

    if _host_in(host, AGGREGATOR_DOMAINS):
        return _is_exact_aggregator_job_url(url)

    if host.startswith(("jobs.", "careers.", "apply.")):
        return path not in {"", "/"}
    if _host_in(host, DIRECT_JOB_DOMAINS):
        return True
    return any(hint in path for hint in JOB_URL_HINTS)


def _is_direct_application_url(url: str) -> bool:
    host = _normalize_host(url)
    if _host_in(host, AGGREGATOR_DOMAINS):
        return False
    return _is_allowed_url(url)


def _looks_like_job_page(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    hits = sum(1 for hint in JOB_TEXT_HINTS if hint in lowered)
    return hits >= 2 or len(text) >= 900


def _tavily_search(query: str, max_results: int, include_domains: List[str]) -> List[Dict[str, str]]:
    try:
        payload: Dict[str, object] = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if include_domains:
            payload["include_domains"] = include_domains

        resp = requests.post(TAVILY_URL, json=payload, timeout=SEARCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return [{"url": res.get("url"), "content": res.get("content")} for res in data.get("results", [])]
    except Exception:
        return []


def _extract_external_from_query(url: str) -> Optional[str]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("url", "target", "dest", "destination", "redirect", "redirect_url", "job_url", "apply"):
        for value in query.get(key, []):
            if value.startswith("//"):
                candidate = f"https:{value}"
            else:
                candidate = value
            canon = _canonicalize_url(candidate)
            if canon and not _is_blocked_url(canon):
                return canon
    return None


def _score_candidate(url: str) -> int:
    host = _normalize_host(url)
    path = urlparse(url).path.lower()
    score = 0

    if _is_blocked_url(url):
        return -100
    if _host_in(host, DIRECT_JOB_DOMAINS):
        score += 7
    elif _host_in(host, AGGREGATOR_DOMAINS):
        score -= 3
    else:
        score += 2

    if host.startswith(("jobs.", "careers.", "apply.")):
        score += 2
    if any(hint in path for hint in JOB_URL_HINTS):
        score += 2
    if re.search(r"\d{5,}", path):
        score += 1
    if path in {"", "/"}:
        score -= 3
    if _is_exact_aggregator_job_url(url):
        score += 1
    return score


def _resolve_from_html(base_url: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    candidates: List[str] = []

    canonical = soup.find("link", attrs={"rel": lambda x: x and "canonical" in x})
    if canonical and canonical.get("href"):
        candidates.append(urljoin(base_url, canonical["href"]))

    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        candidates.append(urljoin(base_url, og["content"]))

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        candidates.append(abs_url)

    best_url = None
    best_score = -999
    for candidate in candidates:
        canon = _canonicalize_url(candidate)
        if not canon:
            continue
        score = _score_candidate(canon)
        if score > best_score:
            best_score = score
            best_url = canon

    if best_url and best_score >= 1:
        return best_url
    return None


def resolve_job_url(url: str, *, max_hops: int = 2) -> Optional[str]:
    """Resolve generic/aggregator links to a canonical job posting URL."""
    if not url:
        return None
    cached = _URL_RESOLUTION_CACHE.get(url)
    if cached is not None or url in _URL_RESOLUTION_CACHE:
        return cached

    current = _canonicalize_url(url)
    if not current:
        _URL_RESOLUTION_CACHE[url] = None
        return None

    visited = set()
    try:
        for _ in range(max_hops + 1):
            if current in visited:
                break
            visited.add(current)

            if _is_blocked_url(current):
                _URL_RESOLUTION_CACHE[url] = None
                return None

            query_target = _extract_external_from_query(current)
            if query_target and query_target not in visited:
                current = query_target
                continue

            host = _normalize_host(current)
            if _host_in(host, AGGREGATOR_DOMAINS):
                try:
                    resp = requests.get(
                        current,
                        timeout=FETCH_TIMEOUT,
                        headers={"User-Agent": USER_AGENT},
                        allow_redirects=True,
                    )
                except Exception:
                    _URL_RESOLUTION_CACHE[url] = None
                    return None

                final_url = _canonicalize_url(resp.url) or current
                if final_url != current and final_url not in visited:
                    current = final_url
                    continue

                html_direct = _resolve_from_html(current, resp.text)
                if html_direct and html_direct not in visited:
                    current = html_direct
                    continue

                _URL_RESOLUTION_CACHE[url] = None
                return None

            if _is_allowed_url(current):
                _URL_RESOLUTION_CACHE[url] = current
                return current

            # Some company pages expose canonical job URLs in HTML.
            try:
                resp = requests.get(current, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                html_direct = _resolve_from_html(current, resp.text)
                if html_direct and html_direct not in visited:
                    current = html_direct
                    continue
            except Exception:
                pass
            break
    except Exception:
        _URL_RESOLUTION_CACHE[url] = None
        return None

    result = current if _is_direct_application_url(current) else None
    _URL_RESOLUTION_CACHE[url] = result
    return result


def search_jobs(queries: List[str], max_results: int = 10) -> List[Dict[str, str]]:
    """Use Tavily to find job links/snippets and resolve to canonical posting URLs."""
    if not TAVILY_API_KEY or not queries:
        return []

    raw_results: List[Dict[str, str]] = []
    direct_domains = sorted(DIRECT_JOB_DOMAINS)
    all_domains = sorted(ALLOWED_DOMAINS)

    query_variants: List[str] = []
    for query in queries:
        query_variants.append(query)
        for site_filter in PRIORITY_SITE_FILTERS:
            query_variants.append(f"{query} {site_filter}")

    per_query_results = max(12, min(35, max_results))
    for query in query_variants:
        raw_results.extend(_tavily_search(query, per_query_results, direct_domains))
        if len(raw_results) >= max_results * 3:
            break

    if len(raw_results) < max_results * 2:
        for query in queries:
            raw_results.extend(_tavily_search(query, per_query_results, all_domains))

    if len(raw_results) < max_results:
        for query in queries:
            raw_results.extend(_tavily_search(query, per_query_results, []))

    seen = set()
    deduped: List[Dict[str, str]] = []
    for res in raw_results:
        original_url = res.get("url")
        if not original_url:
            continue

        resolved_url = resolve_job_url(original_url)
        if not resolved_url or resolved_url in seen:
            continue
        if not _is_direct_application_url(resolved_url):
            continue

        seen.add(resolved_url)
        deduped.append(
            {
                "url": resolved_url,
                "content": res.get("content") or "",
                "original_url": original_url,
            }
        )
        if len(deduped) >= max_results:
            break
    return deduped


def fetch_greenhouse(url: str) -> str:
    patterns = [
        r"(?:boards|job-boards)\.greenhouse\.io/([^/]+)/jobs/([0-9]+)",
        r"greenhouse\.io/([^/]+)/jobs/([0-9]+)",
    ]
    board = None
    job_id = None
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            board, job_id = m.group(1), m.group(2)
            break
    if not board or not job_id:
        return ""

    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    try:
        resp = requests.get(api_url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        desc = data.get("content", "")
        title = data.get("title", "")
        location = (data.get("location") or {}).get("name", "")
        company = board
        return f"Company: {company}\nTitle: {title}\nLocation: {location}\n\n{desc}"
    except Exception:
        return ""


def fetch_lever(url: str) -> str:
    m = re.search(r"lever\.co/([^/]+)/([^/?#]+)", url)
    if not m:
        return ""
    company, job_id = m.group(1), m.group(2)
    api_url = f"https://api.lever.co/v0/postings/{company}/{job_id}"
    try:
        resp = requests.get(api_url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        desc = data.get("descriptionPlain", "") or data.get("description", "")
        title = data.get("text", "")
        location = data.get("categories", {}).get("location", "")
        return f"Company: {company}\nTitle: {title}\nLocation: {location}\n\n{desc}"
    except Exception:
        return ""


def fetch_ashby(url: str) -> str:
    m = re.search(r"ashbyhq\.com/([^/]+)/job/([^/?#]+)", url)
    if not m:
        return ""
    company, job_id = m.group(1), m.group(2)
    api_url = f"https://jobs.ashbyhq.com/api/posting/{company}/{job_id}"
    try:
        resp = requests.get(api_url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        desc = (data.get("jobPosting") or {}).get("description", "")
        title = (data.get("jobPosting") or {}).get("title", "")
        location = (data.get("jobPosting") or {}).get("location", "")
        company_name = (data.get("organization") or {}).get("name", company)
        return f"Company: {company_name}\nTitle: {title}\nLocation: {location}\n\n{desc}"
    except Exception:
        return ""


def fetch_generic(url: str) -> str:
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        title_parts: List[str] = []
        if soup.title and soup.title.string:
            title_parts.append(f"Page Title: {soup.title.string.strip()}")
        for meta_key, label in [
            ("og:title", "OG Title"),
            ("twitter:title", "Twitter Title"),
            ("description", "Meta Description"),
            ("og:description", "OG Description"),
        ]:
            node = soup.find("meta", attrs={"property": meta_key}) or soup.find("meta", attrs={"name": meta_key})
            if node and node.get("content"):
                title_parts.append(f"{label}: {node['content'].strip()}")

        # Capture JSON-LD JobPosting fields before removing script tags.
        for ld in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = (ld.string or "").strip()
            if not raw:
                continue
            try:
                import json

                payload = json.loads(raw)
            except Exception:
                continue

            records = payload if isinstance(payload, list) else [payload]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                rec_type = str(rec.get("@type", "")).lower()
                if "jobposting" not in rec_type:
                    continue
                if rec.get("title"):
                    title_parts.append(f"JobPosting Title: {rec.get('title')}")
                if rec.get("hiringOrganization"):
                    org = rec.get("hiringOrganization")
                    if isinstance(org, dict) and org.get("name"):
                        title_parts.append(f"JobPosting Company: {org.get('name')}")
                if rec.get("description"):
                    title_parts.append(f"JobPosting Description: {rec.get('description')}")
                if rec.get("datePosted"):
                    title_parts.append(f"JobPosting Date Posted: {rec.get('datePosted')}")
                if rec.get("validThrough"):
                    title_parts.append(f"JobPosting Valid Through: {rec.get('validThrough')}")

        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        composed = title_parts + lines[:260]
        return "\n".join(composed)
    except Exception:
        return ""


def fetch_job_text(url: str) -> str:
    host = _normalize_host(url)
    if _host_in(host, {"greenhouse.io", "boards.greenhouse.io", "job-boards.greenhouse.io", "grnh.se"}):
        return fetch_greenhouse(url) or fetch_generic(url)
    if _host_in(host, {"lever.co", "jobs.lever.co"}):
        return fetch_lever(url) or fetch_generic(url)
    if _host_in(host, {"ashbyhq.com", "jobs.ashbyhq.com"}):
        return fetch_ashby(url) or fetch_generic(url)
    return fetch_generic(url)


def _finalize_source(job: Job, resolved_url: str, original_url: Optional[str]) -> None:
    host = _normalize_host(resolved_url)
    source_type = "aggregator" if _host_in(host, AGGREGATOR_DOMAINS) else "direct"
    if job.source:
        job.source.url = resolved_url
        job.source.source_type = source_type
        if original_url and original_url != resolved_url:
            job.source.resolved_from = original_url
    else:
        job.source = JobSource(
            url=resolved_url,
            ats=None,
            source_type=source_type,
            resolved_from=original_url if original_url and original_url != resolved_url else None,
        )


def _process_single_job(link: Dict[str, str]) -> Optional[Job]:
    original_url = link.get("original_url") or link.get("url")
    raw_url = link.get("url")
    if not raw_url:
        return None

    resolved_url = resolve_job_url(raw_url)
    if not resolved_url or not _is_allowed_url(resolved_url):
        return None

    if job_exists_by_url(resolved_url):
        existing = get_job_by_url(resolved_url)
        if existing:
            return existing
        return None

    content = link.get("content") or ""
    if content and not _looks_like_job_page(content):
        content = ""
    if not content:
        content = fetch_job_text(resolved_url)
    if not _looks_like_job_page(content):
        return None

    job_id = f"tavily_{uuid.uuid4().hex[:8]}"
    job = parse_job_with_openai(content, url=resolved_url, job_id=job_id)
    _finalize_source(job, resolved_url, original_url)

    if _is_invalid_job(job):
        return None
    return job


def ingest_jobs_from_queries(queries: List[str], limit: int = 10) -> List[Job]:
    """Search via Tavily and parse results into Job objects with parallel processing."""
    links = search_jobs(queries, max_results=limit)
    jobs: List[Job] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_single_job, link): link for link in links}
        for future in as_completed(futures):
            try:
                job = future.result()
                if job:
                    jobs.append(job)
            except Exception:
                continue

    # Dedupe by canonical source URL and keep stable order by insertion.
    unique_jobs: List[Job] = []
    seen_urls = set()
    for job in jobs:
        url = job.source.url if job.source else None
        key = url or job.id
        if key in seen_urls:
            continue
        seen_urls.add(key)
        unique_jobs.append(job)
    return unique_jobs


def ingest_jobs_from_urls(urls: List[str]) -> List[Job]:
    """Ingest jobs from explicit URLs with deduplication and canonicalization."""
    jobs: List[Job] = []
    for input_url in urls:
        resolved_url = resolve_job_url(input_url)
        if not resolved_url or not _is_allowed_url(resolved_url):
            continue

        if job_exists_by_url(resolved_url):
            existing = get_job_by_url(resolved_url)
            if existing:
                jobs.append(existing)
            continue

        content = fetch_job_text(resolved_url)
        if not content or not _looks_like_job_page(content):
            continue

        job_id = f"url_{uuid.uuid4().hex[:8]}"
        job = parse_job_with_openai(content, url=resolved_url, job_id=job_id)
        _finalize_source(job, resolved_url, input_url)
        if _is_invalid_job(job):
            continue
        jobs.append(job)
    return jobs


def _is_invalid_job(job: Job) -> bool:
    """Heuristic to drop unusable/empty jobs."""
    if (job.company or "").lower() == "unknown" and (job.title or "").lower() == "unknown":
        return True
    text_len = len(job.raw_text or "")
    if not job.requirements.must_have and not job.stack and text_len < 280:
        return True
    return False
