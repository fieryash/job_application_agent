from __future__ import annotations

import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from . import env  # noqa: F401  # ensure .env loaded
from .db import job_exists_by_url, get_job_by_url
from .job_parser import parse_job_with_openai
from .models import Job

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
)

# Reduced timeouts for faster performance
SEARCH_TIMEOUT = 10  # seconds
FETCH_TIMEOUT = 8    # seconds

ALLOWED_DOMAINS = {
    # --- Greenhouse ---
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "greenhouse.io",
    "grnh.se",

    # --- Lever ---
    "jobs.lever.co",
    "lever.co",

    # --- Ashby ---
    "jobs.ashbyhq.com",
    "ashbyhq.com",

    # --- Workday (common variants) ---
    "myworkdayjobs.com",
    "workdayjobs.com",
    "myworkdaysite.com",   # e.g., company.wd1.myworkdaysite.com
    "workday.com",
    "apply.workday.com",

    # --- iCIMS ---
    "icims.com",

    # --- SmartRecruiters ---
    "smartrecruiters.com",
    "jobs.smartrecruiters.com",
    "smrtr.io",            # SmartRecruiters short links

    # --- BambooHR ---
    "bamboohr.com",

    # --- Jobvite ---
    "jobvite.com",
    "jobs.jobvite.com",
    "talent.jobvite.com",

    # --- Workable ---
    "workable.com",
    "apply.workable.com",

    # --- Recruitee ---
    "recruitee.com",

    # --- Teamtailor ---
    "teamtailor.com",

    # --- Personio (EU-heavy) ---
    "personio.com",
    "personio.de",

    # --- JazzHR ---
    "jazzhr.com",

    # --- Breezy HR ---
    "breezy.hr",

    # --- ApplicantPro ---
    "applicantpro.com",

    # --- HiringThing ---
    "hiringthing.com",

    # --- Pinpoint ---
    "pinpointhq.com",

    # --- Comeet ---
    "comeet.co",

    # --- Rippling (some companies host jobs here) ---
    "rippling.com",

    # --- Dover (recruiting platform; common for startups) ---
    "dover.com",
    "hire.dover.com",

    # --- SAP SuccessFactors / Jobs2Web ---
    "successfactors.com",
    "sapsf.com",
    "jobs2web.com",

    # --- Oracle / Taleo ---
    "oraclecloud.com",
    "taleo.net",

    # --- IBM BrassRing / Kenexa (older but still used) ---
    "brassring.com",
    "kenexa.com",

    # --- Paylocity recruiting ---
    "paylocity.com",
    "recruiting.paylocity.com",

    # --- UKG / UltiPro recruiting ---
    "ukg.com",
    "ultipro.com",
    "recruiting.ultipro.com",

    # --- Paycom / Paycor (some companies host postings here) ---
    "paycomonline.net",
    "paycor.com",
    "careers.paycor.com",

    # --- Big company career portals you already started ---
    "amazon.jobs",
    "careers.microsoft.com",
    "careers.google.com",
    "jobs.apple.com",

    # --- Startup boards / aggregators (optional, but useful) ---
    "wellfound.com",
    "angel.co",
    "linkedin.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "builtin.com",
    "dice.com",
    "hired.com",
}


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


def _normalize_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_allowed_url(url: str) -> bool:
    if not url:
        return False
    host = _normalize_host(url)
    if any(host == blocked or host.endswith(f".{blocked}") for blocked in BLOCKED_DOMAINS):
        return False
    if host.startswith(("jobs.", "careers.")):
        return True
    if any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_DOMAINS):
        return True
    path = urlparse(url).path.lower()
    return any(hint in path for hint in JOB_URL_HINTS)


def _looks_like_job_page(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    hits = sum(1 for hint in JOB_TEXT_HINTS if hint in lowered)
    return hits >= 2 or len(text) >= 1200


def _tavily_search(query: str, max_results: int, include_domains: List[str]) -> List[Dict[str, str]]:
    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_domains": include_domains,
                "search_depth": "basic",
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [{"url": res.get("url"), "content": res.get("content")} for res in data.get("results", [])]
    except Exception:
        return []


def search_jobs(queries: List[str], max_results: int = 10) -> List[Dict[str, str]]:
    """Use Tavily to get job links + snippets; returns list of dicts with url and content."""
    if not TAVILY_API_KEY or not queries:
        return []
    results: List[Dict[str, str]] = []
    include_domains = sorted(ALLOWED_DOMAINS)
    for query in queries:
        results.extend(_tavily_search(query, max_results, include_domains))
    # If strict search is too narrow, fall back to open search
    if len(results) < max_results:
        for query in queries:
            results.extend(_tavily_search(query, max_results, []))

    # Deduplicate by URL
    seen = set()
    deduped = []
    for res in results:
        url = res.get("url")
        if url and url not in seen and _is_allowed_url(url):
            seen.add(url)
            deduped.append(res)
    return deduped


def fetch_greenhouse(url: str) -> str:
    """Fetch structured job content from a Greenhouse posting."""
    m = re.search(r"greenhouse.io/([^/]+)/jobs/([0-9]+)", url)
    if not m:
        return ""
    board, job_id = m.group(1), m.group(2)
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
    m = re.search(r"lever.co/([^/]+)/([^/?#]+)", url)
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
    m = re.search(r"ashbyhq.com/([^/]+)/job/([^/?#]+)", url)
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
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text("\n")
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:200])  # Limit to first 200 lines
    except Exception:
        return ""


def fetch_job_text(url: str) -> str:
    if "greenhouse.io" in url:
        return fetch_greenhouse(url)
    if "lever.co" in url:
        return fetch_lever(url)
    if "ashbyhq.com" in url:
        return fetch_ashby(url)
    return fetch_generic(url)


def _process_single_job(link: Dict[str, str]) -> Optional[Job]:
    """Process a single job link. Returns Job or None."""
    url = link.get("url")
    if not url:
        return None
    if not _is_allowed_url(url):
        return None

    # Skip if job already exists in DB (deduplication)
    if job_exists_by_url(url):
        existing = get_job_by_url(url)
        if existing:
            return existing
        return None

    content = link.get("content") or ""
    if content and not _looks_like_job_page(content):
        content = ""
    if not content:
        content = fetch_job_text(url)
    if not _looks_like_job_page(content):
        return None

    job_id = f"tavily_{uuid.uuid4().hex[:8]}"
    job = parse_job_with_openai(content, url=url, job_id=job_id)
    
    if _is_invalid_job(job):
        return None
    
    return job


def ingest_jobs_from_queries(queries: List[str], limit: int = 10) -> List[Job]:
    """Search via Tavily and parse results into Job objects with parallel processing."""
    links = search_jobs(queries, max_results=limit)
    jobs: List[Job] = []
    
    # Use ThreadPoolExecutor for parallel job fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_process_single_job, link): link for link in links}
        for future in as_completed(futures):
            try:
                job = future.result()
                if job:
                    jobs.append(job)
            except Exception:
                continue
    
    return jobs


def ingest_jobs_from_urls(urls: List[str]) -> List[Job]:
    """Ingest jobs from explicit URLs with deduplication."""
    jobs: List[Job] = []
    for url in urls:
        # Skip if already exists
        if job_exists_by_url(url):
            existing = get_job_by_url(url)
            if existing:
                jobs.append(existing)
            continue
        
        content = fetch_job_text(url)
        if not content:
            continue
        job_id = f"url_{uuid.uuid4().hex[:8]}"
        job = parse_job_with_openai(content, url=url, job_id=job_id)
        if _is_invalid_job(job):
            continue
        jobs.append(job)
    return jobs


def _is_invalid_job(job: Job) -> bool:
    """Heuristic to drop unusable/empty jobs."""
    # Both company and title unknown = invalid
    if (job.company or "").lower() == "unknown" and (job.title or "").lower() == "unknown":
        return True
    # No requirements, no stack, and very short text = probably not a job posting
    text_len = len(job.raw_text or "")
    if not job.requirements.must_have and not job.stack and text_len < 300:
        return True
    return False
