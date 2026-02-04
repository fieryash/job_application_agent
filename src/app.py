from __future__ import annotations

import os
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import env  # noqa: F401  # ensures .env is loaded
from .db import (
    get_job,
    get_tailored_resume,
    list_jobs,
    save_tailored_resume,
    upsert_jobs,
    set_job_status,
    remove_job_status,
    get_job_status,
    list_jobs_by_status,
    get_all_job_statuses,
)
from .job_ingest import ingest_jobs_from_queries, ingest_jobs_from_urls
from .matcher import score_job, score_jobs
from .models import FitResult, Job, Profile
from .resume_ingest import build_profile_from_resume, generate_tag
from .storage import (
    DEFAULT_PROFILE_PATH,
    ensure_data_dirs,
    list_profiles,
    load_profile,
    load_profile_by_tag,
    save_profile,
    save_profile_by_tag,
)
from .tailor import tailor_resume

app = FastAPI(title="Job Application Agent", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_WINDOW = 60  # seconds
_REQUEST_BUCKETS: dict[str, Deque[float]] = defaultdict(deque)


@app.middleware("http")
async def auth_and_rate_limit(request, call_next):
    path = request.url.path
    # Allow static UI and health without auth/rate-limit
    if path.startswith("/web") or path.startswith("/health") or path.startswith("/favicon"):
        return await call_next(request)
    # Allow OpenAPI/Swagger docs without key but keep rate limit
    if path.startswith("/docs") or path.startswith("/openapi"):
        pass
    else:
        # API key check
        if API_KEY:
            key = request.headers.get("x-api-key")
            if key != API_KEY:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            bucket_key = key
        else:
            client = getattr(request, "client", None)
            bucket_key = (client.host if client else "anon") or "anon"

        # Rate limit
        now = time.monotonic()
        dq = _REQUEST_BUCKETS[bucket_key]
        while dq and dq[0] < now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MIN:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        dq.append(now)

    response = await call_next(request)
    return response


class IngestRequest(BaseModel):
    queries: List[str] = []
    urls: List[str] = []
    limit: int = 10
    profile_tag: Optional[str] = None


class UpdateTargetsRequest(BaseModel):
    targets: List[str]


class JobStatusRequest(BaseModel):
    status: str  # 'saved' | 'applied' | 'hidden'
    notes: Optional[str] = None


def default_queries(profile: Optional[Profile]) -> List[str]:
    if not profile:
        return ["machine learning engineer remote US", "data scientist remote US"]
    targets = profile.identity.targets or []
    base = "remote US"
    skill_tokens = [s.name for s in profile.skills[:3]]
    queries = []
    for t in targets[:3]:
        queries.append(f"{t.replace('_', ' ')} {base}")
    if skill_tokens:
        queries.append(f"{targets[0] if targets else 'machine learning engineer'} {' '.join(skill_tokens)} {base}")
    return queries or ["machine learning engineer remote US", "data scientist remote US"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/profiles")
def get_profiles() -> List[dict]:
    ensure_data_dirs()
    profiles = []
    for meta in list_profiles():
        path = Path(meta["path"])
        try:
            profile = load_profile(path)
            profiles.append(
                {
                    "tag": profile.tag,
                    "name": profile.identity.name,
                    "targets": profile.identity.targets,
                    "path": str(path),
                }
            )
        except Exception:
            profiles.append({"tag": meta["tag"], "name": "invalid", "targets": [], "path": str(path)})
    return profiles


@app.get("/profiles/{tag}")
def get_profile(tag: str) -> Profile:
    profile = load_profile_by_tag(tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {tag} not found")
    return profile


@app.post("/profiles/from-resume")
async def create_profile_from_resume(file: UploadFile = File(...), base_tag: str | None = None) -> Profile:
    ensure_data_dirs()
    base_profile = load_profile_by_tag(base_tag) if base_tag else None
    suffix = Path(file.filename or "resume.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    new_tag = generate_tag()
    profile = build_profile_from_resume(tmp_path, base_profile=base_profile, explicit_tag=new_tag)
    save_profile_by_tag(profile)
    return profile


@app.post("/match")
def match_job(profile_tag: str, job: Job) -> FitResult:
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    return score_job(profile, job)


@app.post("/match/batch")
def match_jobs(profile_tag: str, jobs: List[Job]) -> List[FitResult]:
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    return score_jobs(profile, jobs)


@app.post("/profiles/save-default")
def save_default_profile(profile: Profile) -> dict:
    """Replace the default profile.json with the provided profile."""
    path = save_profile(profile, DEFAULT_PROFILE_PATH)
    return {"saved_to": str(path)}


@app.post("/profiles/{tag}/targets")
def update_profile_targets(tag: str, payload: UpdateTargetsRequest) -> Profile:
    profile = load_profile_by_tag(tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {tag} not found")
    profile.identity.targets = payload.targets
    save_profile_by_tag(profile)
    try:
        default_profile = load_profile(DEFAULT_PROFILE_PATH)
        if tag == default_profile.tag:
            save_profile(profile, DEFAULT_PROFILE_PATH)
    except Exception:
        pass
    return profile


@app.post("/jobs/ingest")
def ingest_jobs(req: IngestRequest) -> List[Job]:
    profile = load_profile_by_tag(req.profile_tag) if req.profile_tag else None
    combined_queries = req.queries or default_queries(profile)
    jobs: List[Job] = []
    if req.urls:
        jobs.extend(ingest_jobs_from_urls(req.urls))
    if combined_queries:
        jobs.extend(ingest_jobs_from_queries(combined_queries, limit=req.limit))
    # Filter out unknowns
    usable = [j for j in jobs if (j.company and j.company.lower() != "unknown") or (j.title and j.title.lower() != "unknown")]
    if usable:
        jobs = usable
    upsert_jobs(jobs)
    return jobs


@app.get("/jobs")
def get_jobs(limit: int = 20, profile_tag: Optional[str] = None, exclude_applied: bool = False) -> List[Job]:
    """Get jobs with optional filtering by status."""
    if exclude_applied and profile_tag:
        return list_jobs(limit=limit, exclude_status=["applied", "hidden"], profile_tag=profile_tag)
    return list_jobs(limit=limit)


@app.get("/jobs/saved")
def get_saved_jobs(profile_tag: str, limit: int = 50) -> List[Job]:
    """Get saved jobs for a profile."""
    return list_jobs_by_status(profile_tag, "saved", limit)


@app.get("/jobs/applied")
def get_applied_jobs(profile_tag: str, limit: int = 50) -> List[Job]:
    """Get applied jobs for a profile."""
    return list_jobs_by_status(profile_tag, "applied", limit)


@app.get("/jobs/statuses")
def get_job_statuses(profile_tag: str) -> dict:
    """Get all job statuses for a profile as {job_id: status}."""
    return get_all_job_statuses(profile_tag)


@app.get("/jobs/{job_id}")
def get_job_by_id(job_id: str) -> Job:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@app.post("/jobs/{job_id}/status")
def update_job_status(job_id: str, profile_tag: str, payload: JobStatusRequest) -> dict:
    """Set status for a job (saved, applied, hidden)."""
    if payload.status not in ["saved", "applied", "hidden"]:
        raise HTTPException(status_code=400, detail="status must be 'saved', 'applied', or 'hidden'")
    set_job_status(job_id, profile_tag, payload.status, payload.notes)
    return {"job_id": job_id, "status": payload.status}


@app.delete("/jobs/{job_id}/status")
def clear_job_status(job_id: str, profile_tag: str) -> dict:
    """Remove status from a job (unsave, un-apply)."""
    remove_job_status(job_id, profile_tag)
    return {"job_id": job_id, "status": None}


@app.get("/jobs/{job_id}/status")
def get_single_job_status(job_id: str, profile_tag: str) -> dict:
    """Get status for a single job."""
    status = get_job_status(job_id, profile_tag)
    return {"job_id": job_id, "status": status}


@app.post("/tailor")
def tailor(profile_tag: str, job: Job):
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = tailor_resume(profile, job)
    save_tailored_resume(
        job_id=job.id,
        profile_tag=profile_tag,
        raw_json=tailored.model_dump_json(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return tailored


@app.get("/tailor/{profile_tag}/{job_id}")
def get_tailored(profile_tag: str, job_id: str):
    raw = get_tailored_resume(job_id, profile_tag)
    if raw:
        return raw
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"tailored resume not found for job {job_id}")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = tailor_resume(profile, job)
    save_tailored_resume(
        job_id=job.id,
        profile_tag=profile_tag,
        raw_json=tailored.model_dump_json(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return tailored
