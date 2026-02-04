from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Job

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "jobs.db"


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        # Jobs table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                company TEXT,
                title TEXT,
                location TEXT,
                remote_policy TEXT,
                level TEXT,
                source_url TEXT UNIQUE,
                ats TEXT,
                raw_json TEXT NOT NULL,
                ingested_at TEXT
            )
            """
        )
        # Create index on source_url for fast lookup
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_source_url ON jobs(source_url)"
        )
        
        # Tailored resumes table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tailored_resumes (
                job_id TEXT NOT NULL,
                profile_tag TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (job_id, profile_tag)
            )
            """
        )
        
        # Job user status table (saved, applied, hidden)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_user_status (
                job_id TEXT NOT NULL,
                profile_tag TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                applied_at TEXT,
                notes TEXT,
                PRIMARY KEY (job_id, profile_tag)
            )
            """
        )
        
        conn.commit()


def job_exists_by_url(url: str) -> bool:
    """Check if a job with this source URL already exists."""
    if not url:
        return False
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE source_url = ?", (url,)
        ).fetchone()
    return row is not None


def get_job_by_url(url: str) -> Optional[Job]:
    """Get a job by its source URL."""
    if not url:
        return None
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT raw_json FROM jobs WHERE source_url = ?", (url,)
        ).fetchone()
    if not row:
        return None
    try:
        return Job.model_validate_json(row["raw_json"])
    except Exception:
        return None


def upsert_job(job: Job) -> None:
    init_db()
    raw = job.model_dump_json()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, company, title, location, remote_policy, level, source_url, ats, raw_json, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                company=excluded.company,
                title=excluded.title,
                location=excluded.location,
                remote_policy=excluded.remote_policy,
                level=excluded.level,
                source_url=excluded.source_url,
                ats=excluded.ats,
                raw_json=excluded.raw_json,
                ingested_at=excluded.ingested_at
            """,
            (
                job.id,
                job.company,
                job.title,
                job.location,
                job.remote_policy,
                job.level,
                job.source.url if job.source else None,
                job.source.ats if job.source else None,
                raw,
                job.ingested_at,
            ),
        )
        conn.commit()


def upsert_jobs(jobs: Iterable[Job]) -> None:
    for job in jobs:
        upsert_job(job)


def list_jobs(limit: int = 50, exclude_status: Optional[List[str]] = None, profile_tag: Optional[str] = None) -> List[Job]:
    """
    List jobs with optional status filtering.
    exclude_status: List of statuses to exclude (e.g., ['applied', 'hidden'])
    """
    init_db()
    with _conn() as conn:
        if exclude_status and profile_tag:
            placeholders = ",".join("?" * len(exclude_status))
            rows = conn.execute(
                f"""
                SELECT j.raw_json FROM jobs j
                LEFT JOIN job_user_status jus ON j.id = jus.job_id AND jus.profile_tag = ?
                WHERE jus.status IS NULL OR jus.status NOT IN ({placeholders})
                ORDER BY j.ingested_at DESC LIMIT ?
                """,
                (profile_tag, *exclude_status, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT raw_json FROM jobs ORDER BY ingested_at DESC LIMIT ?", (limit,)
            ).fetchall()
    
    results: List[Job] = []
    for row in rows:
        try:
            results.append(Job.model_validate_json(row["raw_json"]))
        except Exception:
            continue
    return results


def get_job(job_id: str) -> Optional[Job]:
    init_db()
    with _conn() as conn:
        row = conn.execute("SELECT raw_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    try:
        return Job.model_validate_json(row["raw_json"])
    except Exception:
        return None


def save_tailored_resume(job_id: str, profile_tag: str, raw_json: str, created_at: str) -> None:
    init_db()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO tailored_resumes (job_id, profile_tag, raw_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id, profile_tag) DO UPDATE SET
                raw_json=excluded.raw_json,
                created_at=excluded.created_at
            """,
            (job_id, profile_tag, raw_json, created_at),
        )
        conn.commit()


def get_tailored_resume(job_id: str, profile_tag: str) -> Optional[str]:
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT raw_json FROM tailored_resumes WHERE job_id = ? AND profile_tag = ?",
            (job_id, profile_tag),
        ).fetchone()
    if not row:
        return None
    return row["raw_json"]


# Job status functions (saved, applied, hidden)

def set_job_status(job_id: str, profile_tag: str, status: str, notes: Optional[str] = None) -> None:
    """Set job status for a user. Status: 'saved' | 'applied' | 'hidden'"""
    from datetime import datetime, timezone
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    applied_at = now if status == "applied" else None
    
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO job_user_status (job_id, profile_tag, status, updated_at, applied_at, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, profile_tag) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                applied_at=COALESCE(excluded.applied_at, job_user_status.applied_at),
                notes=COALESCE(excluded.notes, job_user_status.notes)
            """,
            (job_id, profile_tag, status, now, applied_at, notes),
        )
        conn.commit()


def remove_job_status(job_id: str, profile_tag: str) -> None:
    """Remove job status (unsave, unapply)."""
    init_db()
    with _conn() as conn:
        conn.execute(
            "DELETE FROM job_user_status WHERE job_id = ? AND profile_tag = ?",
            (job_id, profile_tag),
        )
        conn.commit()


def get_job_status(job_id: str, profile_tag: str) -> Optional[str]:
    """Get job status for a user."""
    init_db()
    with _conn() as conn:
        row = conn.execute(
            "SELECT status FROM job_user_status WHERE job_id = ? AND profile_tag = ?",
            (job_id, profile_tag),
        ).fetchone()
    return row["status"] if row else None


def list_jobs_by_status(profile_tag: str, status: str, limit: int = 50) -> List[Job]:
    """List jobs with a specific status for a user."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT j.raw_json FROM jobs j
            JOIN job_user_status jus ON j.id = jus.job_id
            WHERE jus.profile_tag = ? AND jus.status = ?
            ORDER BY jus.updated_at DESC LIMIT ?
            """,
            (profile_tag, status, limit)
        ).fetchall()
    
    results: List[Job] = []
    for row in rows:
        try:
            results.append(Job.model_validate_json(row["raw_json"]))
        except Exception:
            continue
    return results


def get_all_job_statuses(profile_tag: str) -> dict:
    """Get all job statuses for a user as {job_id: status}."""
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT job_id, status FROM job_user_status WHERE profile_tag = ?",
            (profile_tag,)
        ).fetchall()
    return {row["job_id"]: row["status"] for row in rows}
