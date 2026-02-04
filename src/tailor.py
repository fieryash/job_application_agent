from __future__ import annotations

import difflib
import os
import re
from typing import Dict, List, Tuple

from openai import OpenAI

from . import env  # noqa: F401  # ensure .env loaded
from .models import Bullet, Job, Profile, TailoredBullet, TailoredResume

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def select_bullets(profile: Profile, job: Job, max_bullets: int = 6) -> List[Bullet]:
    """Pick bullets that overlap job requirements/stack/keywords."""
    job_tokens = set(tok.lower() for tok in job.requirements.must_have + job.stack + job.requirements.nice_to_have)
    scored: List[Tuple[float, Bullet]] = []
    for b in profile.bullet_bank:
        hit = len(job_tokens & {k.lower() for k in b.skills + b.keywords})
        scored.append((hit, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [b for score, b in scored if score > 0][:max_bullets]
    if not selected:
        selected = profile.bullet_bank[:max_bullets]
    return selected


def rewrite_bullet(bullet: Bullet, job: Job, allowed_keywords: List[str]) -> TailoredBullet:
    """
    Rewrite a bullet to mirror JD language without adding unsupported claims.
    Guardrails: forbid new numbers and tools not in allowed_keywords.
    """
    client = OpenAI()
    prompt = (
        "You rewrite resume bullets to mirror the job description language WITHOUT changing facts. "
        "Do not add new metrics, dates, or tools not already present. Keep tense consistent. "
        f"Allowed keywords/tools: {allowed_keywords}. "
        f"Job title: {job.title}, company: {job.company}. "
        "Return only the rewritten bullet."
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=f"Original bullet: {bullet.text}\nJob requirements: {job.requirements.must_have + job.requirements.nice_to_have}\nJob stack: {job.stack}",
            system=prompt,
            temperature=0.2,
        )
        rewritten = resp.output_text.strip()
    except Exception:
        rewritten = bullet.text
    diff = "\n".join(difflib.ndiff([bullet.text], [rewritten]))
    return TailoredBullet(source_id=bullet.id, rewritten_text=rewritten, diff_from_source=diff)


def validate_tailoring(selected: List[Bullet], rewritten: List[TailoredBullet]) -> Dict[str, object]:
    original_numbers = re.findall(r"\d[\d.,%]*", " ".join(b.text for b in selected))
    new_numbers = re.findall(r"\d[\d.,%]*", " ".join(tb.rewritten_text for tb in rewritten))
    unsupported_numbers = [n for n in new_numbers if n not in original_numbers]
    return {
        "factuality": "fail" if unsupported_numbers else "pass",
        "new_numbers_added": bool(unsupported_numbers),
        "unsupported_numbers": unsupported_numbers,
        "length_ok": all(len(tb.rewritten_text) <= 220 for tb in rewritten),
    }


def tailor_resume(profile: Profile, job: Job) -> TailoredResume:
    selected = select_bullets(profile, job)
    allowed_keywords = {kw.lower() for b in selected for kw in (b.skills + b.keywords)}
    rewritten: List[TailoredBullet] = []
    for bullet in selected:
        rewritten.append(rewrite_bullet(bullet, job, sorted(allowed_keywords)))
    validation = validate_tailoring(selected, rewritten)
    ats_keyword_report = {
        "before": sorted({kw.lower() for b in selected for kw in (b.skills + b.keywords)}),
        "after": sorted({kw.lower() for tb in rewritten for kw in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]+", tb.rewritten_text)}),
    }
    ats_keyword_report["coverage_delta"] = len(ats_keyword_report["after"]) - len(ats_keyword_report["before"])
    return TailoredResume(
        job_id=job.id,
        selected_bullets=[b.id for b in selected],
        rewritten_bullets=rewritten,
        ats_keyword_report=ats_keyword_report,
        validation=validation,
        exports={},
        human_approval={"status": "pending"},
    )
