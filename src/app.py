from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import env  # noqa: F401  # ensures .env is loaded
from .auto_apply import auto_apply_job
from .db import (
    get_all_job_statuses,
    get_job,
    get_job_status,
    get_tailored_resume,
    list_jobs,
    list_jobs_by_status,
    remove_job_status,
    save_tailored_resume,
    set_job_status,
    upsert_jobs,
)
from .job_ingest import ingest_jobs_from_queries, ingest_jobs_from_urls
from .matcher import score_job, score_jobs
from .models import AutoApplyResult, FitResult, Job, Profile, TailoredEditorDraft, TailoredResume
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
from .tailor import (
    build_default_editor_draft,
    normalize_editor_draft,
    render_editor_preview_html,
    render_tailored_resume_text,
    tailor_resume,
)

app = FastAPI(title="Job Application Agent", version="0.3.0")

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_LOCAL_PER_MIN = int(os.getenv("RATE_LIMIT_LOCAL_PER_MIN", "1000"))
RATE_LIMIT_WINDOW = 60  # seconds
_REQUEST_BUCKETS: dict[str, Deque[float]] = defaultdict(deque)
_AUTO_APPLY_RUNS: dict[str, dict] = {}
_AUTO_APPLY_CONTROLS: dict[str, dict] = {}
_AUTO_APPLY_LOCK = threading.Lock()


class IngestRequest(BaseModel):
    queries: List[str] = []
    urls: List[str] = []
    limit: int = 10
    profile_tag: Optional[str] = None


class UpdateTargetsRequest(BaseModel):
    targets: List[str]


class JobStatusRequest(BaseModel):
    status: str  # saved | applied | hidden
    notes: Optional[str] = None


class AutoApplyRequest(BaseModel):
    auto_submit: bool = False
    headless: bool = True


class AutoApplyActionRequest(BaseModel):
    action: str = "close_browser"  # close_browser | close_and_mark_applied


class TailorEditorUpdateRequest(BaseModel):
    summary: str = ""
    skills: List[str] = []
    bullets: List[str] = []
    persist: bool = True


def _set_apply_run(run_key: str, **updates) -> None:
    with _AUTO_APPLY_LOCK:
        current = _AUTO_APPLY_RUNS.get(run_key, {})
        merged = {**current, **updates, "updated_at": datetime.now(timezone.utc).isoformat()}
        _AUTO_APPLY_RUNS[run_key] = merged


def _append_apply_step(run_id: str, step: str) -> None:
    with _AUTO_APPLY_LOCK:
        current = _AUTO_APPLY_RUNS.get(run_id)
        if not current:
            return
        steps = list(current.get("steps") or [])
        steps.append(step)
        current["steps"] = steps
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        _AUTO_APPLY_RUNS[run_id] = current


def _set_apply_control(run_id: str, **updates) -> None:
    with _AUTO_APPLY_LOCK:
        current = _AUTO_APPLY_CONTROLS.get(run_id, {})
        _AUTO_APPLY_CONTROLS[run_id] = {**current, **updates}


def _get_apply_control(run_id: str) -> Optional[dict]:
    with _AUTO_APPLY_LOCK:
        return _AUTO_APPLY_CONTROLS.get(run_id)


def _pop_apply_control(run_id: str) -> None:
    with _AUTO_APPLY_LOCK:
        _AUTO_APPLY_CONTROLS.pop(run_id, None)


def _prepare_auto_apply_context(profile_tag: str, job_id: str) -> tuple[Profile, Job, Path]:
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")

    tailored = _ensure_tailored_resume(profile_tag, job_id, profile, job)
    resume_txt = Path(tailored.exports.get("txt_path", "")) if tailored.exports else None
    resume_path: Optional[Path] = None
    if resume_txt and resume_txt.exists():
        resume_path = resume_txt
    elif profile.source_resume:
        source_path = Path(profile.source_resume)
        if source_path.exists():
            resume_path = source_path

    if not resume_path:
        raise HTTPException(status_code=500, detail="No resume file available for auto-apply.")
    return profile, job, resume_path


@app.middleware("http")
async def auth_and_rate_limit(request, call_next):
    path = request.url.path
    if path.startswith("/web") or path.startswith("/health") or path.startswith("/favicon"):
        return await call_next(request)

    if path.startswith("/docs") or path.startswith("/openapi"):
        pass
    else:
        client = getattr(request, "client", None)
        client_host = (client.host if client else "anon") or "anon"
        is_local = client_host in {"127.0.0.1", "::1", "localhost"}
        rate_limit_per_min = RATE_LIMIT_LOCAL_PER_MIN if is_local else RATE_LIMIT_PER_MIN

        if API_KEY:
            key = request.headers.get("x-api-key")
            if key != API_KEY:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            bucket_key = key
        else:
            bucket_key = client_host

        now = time.monotonic()
        dq = _REQUEST_BUCKETS[bucket_key]
        while dq and dq[0] < now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= rate_limit_per_min:
            retry_after = 1
            if dq:
                retry_after = max(1, int(math.ceil((dq[0] + RATE_LIMIT_WINDOW) - now)))
            return JSONResponse(
                {"detail": "rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        dq.append(now)

    response = await call_next(request)
    return response


def default_queries(profile: Optional[Profile]) -> List[str]:
    if not profile:
        return ["machine learning engineer remote US", "data scientist remote US"]
    targets = profile.identity.targets or []
    base = "remote US"
    skill_tokens = [s.name for s in profile.skills[:3]]
    queries = []
    for target in targets[:3]:
        queries.append(f"{target.replace('_', ' ')} {base}")
    if skill_tokens:
        queries.append(f"{targets[0] if targets else 'machine learning engineer'} {' '.join(skill_tokens)} {base}")
    return queries or ["machine learning engineer remote US", "data scientist remote US"]


def _tailored_export_dir(profile_tag: str, job_id: str) -> Path:
    path = EXPORTS_DIR / profile_tag / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _wrap_preview_html(profile: Profile, job: Job, preview_fragment: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tailored Resume Preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; padding: 24px; color: #0f172a; }}
    .container {{ max-width: 900px; margin: 0 auto; background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; }}
    .meta {{ margin-bottom: 18px; }}
    .meta h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
    .meta p {{ margin: 0; color: #475569; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="meta">
      <h1>{profile.identity.name}</h1>
      <p>Tailored for {job.title} at {job.company}</p>
    </div>
    {preview_fragment or "<p>No preview available.</p>"}
  </div>
</body>
</html>"""


def _load_tailored_model(job_id: str, profile_tag: str) -> Optional[TailoredResume]:
    raw = get_tailored_resume(job_id, profile_tag)
    if not raw:
        return None
    try:
        return TailoredResume.model_validate_json(raw)
    except Exception:
        try:
            obj = json.loads(raw)
            return TailoredResume.model_validate(obj)
        except Exception:
            return None


def _persist_tailored_with_exports(
    *,
    profile: Profile,
    job: Job,
    profile_tag: str,
    tailored: TailoredResume,
) -> TailoredResume:
    export_dir = _tailored_export_dir(profile_tag, job.id)
    txt_path = export_dir / "tailored_resume.txt"
    html_path = export_dir / "tailored_resume_preview.html"
    json_path = export_dir / "tailored_resume.json"

    if tailored.editor_draft:
        tailored.editor_draft = normalize_editor_draft(tailored.editor_draft)
        tailored.preview_html = render_editor_preview_html(job, tailored.editor_draft)

    text_blob = render_tailored_resume_text(profile, job, tailored)
    txt_path.write_text(text_blob, encoding="utf-8")
    html_path.write_text(
        _wrap_preview_html(profile, job, tailored.preview_html or ""),
        encoding="utf-8",
    )

    tailored.exports = {
        "download_txt_url": f"/tailor/{profile_tag}/{job.id}/download?format=txt",
        "download_html_url": f"/tailor/{profile_tag}/{job.id}/download?format=html",
        "preview_url": f"/tailor/{profile_tag}/{job.id}/preview",
        "txt_path": str(txt_path),
        "html_path": str(html_path),
        "json_path": str(json_path),
    }
    json_path.write_text(tailored.model_dump_json(indent=2), encoding="utf-8")

    save_tailored_resume(
        job_id=job.id,
        profile_tag=profile_tag,
        raw_json=tailored.model_dump_json(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return tailored


def _ensure_tailored_resume(profile_tag: str, job_id: str, profile: Profile, job: Job) -> TailoredResume:
    tailored = _load_tailored_model(job_id, profile_tag)
    if not tailored:
        tailored = tailor_resume(profile, job)
        return _persist_tailored_with_exports(profile=profile, job=job, profile_tag=profile_tag, tailored=tailored)

    if not tailored.editor_draft:
        tailored.editor_draft = build_default_editor_draft(profile, job, tailored)
        return _persist_tailored_with_exports(profile=profile, job=job, profile_tag=profile_tag, tailored=tailored)

    export_txt = tailored.exports.get("txt_path") if tailored.exports else None
    export_html = tailored.exports.get("html_path") if tailored.exports else None
    if (
        not tailored.exports
        or not export_txt
        or not export_html
        or not Path(export_txt).exists()
        or not Path(export_html).exists()
    ):
        return _persist_tailored_with_exports(profile=profile, job=job, profile_tag=profile_tag, tailored=tailored)

    return tailored


def _profile_with_editor_draft(profile: Profile, draft: TailoredEditorDraft) -> Profile:
    clone = profile.model_copy(deep=True)
    normalized = normalize_editor_draft(draft)
    clone.raw_resume_text = "\n".join(
        [
            normalized.summary,
            " ".join(normalized.skills),
            "\n".join(normalized.bullets),
        ]
    ).strip()
    return clone


def _tailor_editor_response(
    *,
    profile: Profile,
    job: Job,
    tailored: TailoredResume,
) -> dict:
    draft = normalize_editor_draft(
        tailored.editor_draft or build_default_editor_draft(profile, job, tailored)
    )
    fit = score_job(_profile_with_editor_draft(profile, draft), job)
    missing = []
    for kw in fit.keyword_coverage.must_have_missing + fit.keyword_coverage.stack_missing:
        if kw not in missing:
            missing.append(kw)

    return {
        "job_id": job.id,
        "profile_tag": profile.tag,
        "draft": draft.model_dump(),
        "preview_html": render_editor_preview_html(job, draft),
        "fit_score": fit.fit_score,
        "confidence": fit.confidence,
        "score_components": [comp.model_dump() for comp in fit.score_components],
        "missing_keywords": missing[:20],
        "matched_keywords": (fit.keyword_coverage.must_have_matched + fit.keyword_coverage.stack_matched)[:20],
        "why": fit.why,
        "gaps": fit.gaps,
        "exports": tailored.exports,
        "validation": tailored.validation,
    }


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

    # Drop placeholders with no role signal.
    usable = [
        j
        for j in jobs
        if (j.company and j.company.lower() != "unknown") or (j.title and j.title.lower() != "unknown")
    ]
    if usable:
        jobs = usable
    upsert_jobs(jobs)
    return jobs


@app.get("/jobs")
def get_jobs(limit: int = 20, offset: int = 0, profile_tag: Optional[str] = None, exclude_applied: bool = False) -> List[Job]:
    if exclude_applied and profile_tag:
        return list_jobs(limit=limit, offset=offset, exclude_status=["applied", "hidden"], profile_tag=profile_tag)
    return list_jobs(limit=limit, offset=offset)


@app.get("/jobs/saved")
def get_saved_jobs(profile_tag: str, limit: int = 50, offset: int = 0) -> List[Job]:
    return list_jobs_by_status(profile_tag, "saved", limit, offset)


@app.get("/jobs/applied")
def get_applied_jobs(profile_tag: str, limit: int = 50, offset: int = 0) -> List[Job]:
    return list_jobs_by_status(profile_tag, "applied", limit, offset)


@app.get("/jobs/statuses")
def get_job_statuses(profile_tag: str) -> dict:
    return get_all_job_statuses(profile_tag)


@app.get("/jobs/{job_id}")
def get_job_by_id(job_id: str) -> Job:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


@app.post("/jobs/{job_id}/status")
def update_job_status(job_id: str, profile_tag: str, payload: JobStatusRequest) -> dict:
    if payload.status not in ["saved", "applied", "hidden"]:
        raise HTTPException(status_code=400, detail="status must be 'saved', 'applied', or 'hidden'")
    set_job_status(job_id, profile_tag, payload.status, payload.notes)
    return {"job_id": job_id, "status": payload.status}


@app.delete("/jobs/{job_id}/status")
def clear_job_status(job_id: str, profile_tag: str) -> dict:
    remove_job_status(job_id, profile_tag)
    return {"job_id": job_id, "status": None}


@app.get("/jobs/{job_id}/status")
def get_single_job_status(job_id: str, profile_tag: str) -> dict:
    status = get_job_status(job_id, profile_tag)
    return {"job_id": job_id, "status": status}


@app.post("/jobs/{job_id}/auto-apply")
def auto_apply(job_id: str, profile_tag: str, payload: AutoApplyRequest) -> AutoApplyResult:
    profile, job, resume_path = _prepare_auto_apply_context(profile_tag, job_id)

    result = auto_apply_job(
        profile=profile,
        job=job,
        resume_path=resume_path,
        auto_submit=False,  # Safety policy: never auto-submit
        headless=True,  # Non-interactive endpoint stays headless
    )

    if result.submitted:
        set_job_status(job_id, profile_tag, "applied", notes=result.message)
    elif result.status == "ready_to_submit":
        set_job_status(job_id, profile_tag, "saved", notes="Auto-fill completed; pending manual submit.")
    return result


@app.post("/jobs/{job_id}/auto-apply/start")
def auto_apply_start(job_id: str, profile_tag: str, payload: AutoApplyRequest) -> dict:
    profile, job, resume_path = _prepare_auto_apply_context(profile_tag, job_id)

    run_id = f"apply_{uuid.uuid4().hex[:10]}"
    started_at = datetime.now(timezone.utc).isoformat()
    close_event = threading.Event()
    _set_apply_control(run_id, close_event=close_event)
    _set_apply_run(
        run_id,
        run_id=run_id,
        job_id=job_id,
        profile_tag=profile_tag,
        status="queued",
        started_at=started_at,
        updated_at=started_at,
        steps=[],
        result=None,
        error=None,
        user_action_required=False,
        user_prompt=None,
        missing_fields=[],
        user_confirmed_submit=False,
    )

    def worker() -> None:
        try:
            _set_apply_run(run_id, status="running")
            result = auto_apply_job(
                profile=profile,
                job=job,
                resume_path=resume_path,
                auto_submit=False,  # Safety policy: never auto-submit
                headless=payload.headless,
                progress_cb=lambda step: _append_apply_step(run_id, step),
                pause_cb=lambda info: _set_apply_run(
                    run_id,
                    status="waiting_for_user",
                    user_action_required=True,
                    user_prompt=info.get("prompt"),
                    missing_fields=info.get("missing_fields") or [],
                    current_url=info.get("url"),
                    waiting_reason=info.get("reason"),
                ),
                wait_for_close_cb=close_event.wait if not payload.headless else None,
            )

            with _AUTO_APPLY_LOCK:
                manual_submitted = bool((_AUTO_APPLY_RUNS.get(run_id) or {}).get("user_confirmed_submit"))
            if manual_submitted:
                result.submitted = True
                result.status = "submitted"
                result.message = "Application submitted manually (user confirmed)."
                result.steps.append("User confirmed manual submission")

            if result.submitted:
                set_job_status(job_id, profile_tag, "applied", notes=result.message)
            elif result.status == "ready_to_submit":
                set_job_status(job_id, profile_tag, "saved", notes="Auto-fill completed; pending manual submit.")

            _set_apply_run(
                run_id,
                status="completed",
                user_action_required=False,
                user_prompt=None,
                result=result.model_dump(),
            )
        except Exception as exc:
            _set_apply_run(run_id, status="failed", error=str(exc))
        finally:
            _pop_apply_control(run_id)

    t = threading.Thread(target=worker, daemon=True, name=f"auto-apply-{run_id}")
    t.start()
    return {"run_id": run_id, "status": "queued", "job_id": job_id, "profile_tag": profile_tag}


@app.get("/jobs/auto-apply/{run_id}")
def auto_apply_status(run_id: str) -> dict:
    with _AUTO_APPLY_LOCK:
        run = _AUTO_APPLY_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"auto-apply run {run_id} not found")
    return run


@app.post("/jobs/auto-apply/{run_id}/action")
def auto_apply_action(run_id: str, payload: AutoApplyActionRequest) -> dict:
    with _AUTO_APPLY_LOCK:
        run = _AUTO_APPLY_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"auto-apply run {run_id} not found")

    action = (payload.action or "").strip().lower()
    if action not in {"close_browser", "close_and_mark_applied"}:
        raise HTTPException(status_code=400, detail="action must be close_browser or close_and_mark_applied")

    updates = {"status": "closing_browser", "user_action_required": False, "user_prompt": "Closing browser..."}
    if action == "close_and_mark_applied":
        updates["user_confirmed_submit"] = True
    _set_apply_run(run_id, **updates)

    control = _get_apply_control(run_id) or {}
    close_event = control.get("close_event")
    if close_event:
        close_event.set()

    return {"run_id": run_id, "action": action, "status": "accepted"}


@app.post("/tailor")
def tailor(profile_tag: str, job: Job) -> TailoredResume:
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = tailor_resume(profile, job)
    tailored = _persist_tailored_with_exports(profile=profile, job=job, profile_tag=profile_tag, tailored=tailored)
    return tailored


@app.get("/tailor/{profile_tag}/{job_id}")
def get_tailored(profile_tag: str, job_id: str) -> TailoredResume:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    return _ensure_tailored_resume(profile_tag, job_id, profile, job)


@app.get("/tailor/{profile_tag}/{job_id}/editor")
def get_tailor_editor(profile_tag: str, job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = _ensure_tailored_resume(profile_tag, job_id, profile, job)
    return _tailor_editor_response(profile=profile, job=job, tailored=tailored)


@app.post("/tailor/{profile_tag}/{job_id}/editor")
def update_tailor_editor(profile_tag: str, job_id: str, payload: TailorEditorUpdateRequest) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")

    tailored = _ensure_tailored_resume(profile_tag, job_id, profile, job)
    draft = normalize_editor_draft(
        TailoredEditorDraft(summary=payload.summary, skills=payload.skills, bullets=payload.bullets)
    )

    if payload.persist:
        tailored.editor_draft = draft
        tailored.preview_html = render_editor_preview_html(job, draft)
        tailored = _persist_tailored_with_exports(profile=profile, job=job, profile_tag=profile_tag, tailored=tailored)
    else:
        tailored.editor_draft = draft

    response = _tailor_editor_response(profile=profile, job=job, tailored=tailored)
    response["saved"] = bool(payload.persist)
    return response


@app.get("/tailor/{profile_tag}/{job_id}/preview")
def get_tailored_preview(profile_tag: str, job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = _ensure_tailored_resume(profile_tag, job_id, profile, job)
    return {
        "job_id": job_id,
        "profile_tag": profile_tag,
        "preview_html": tailored.preview_html,
        "exports": tailored.exports,
    }


@app.get("/tailor/{profile_tag}/{job_id}/download")
def download_tailored(profile_tag: str, job_id: str, format: str = "txt"):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    profile = load_profile_by_tag(profile_tag)
    if not profile:
        raise HTTPException(status_code=404, detail=f"profile with tag {profile_tag} not found")
    tailored = _ensure_tailored_resume(profile_tag, job_id, profile, job)

    fmt = (format or "txt").lower()
    if fmt not in {"txt", "html"}:
        raise HTTPException(status_code=400, detail="format must be txt or html")

    path_key = "txt_path" if fmt == "txt" else "html_path"
    media_type = "text/plain" if fmt == "txt" else "text/html"
    path_raw = tailored.exports.get(path_key) if tailored.exports else None
    if not path_raw:
        raise HTTPException(status_code=404, detail="tailored export not found")
    path = Path(path_raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail="tailored export file missing")

    filename = f"{profile_tag}_{job_id}_tailored_resume.{fmt}"
    return FileResponse(path, media_type=media_type, filename=filename)
