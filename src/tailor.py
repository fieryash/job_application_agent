from __future__ import annotations

import difflib
import html
import os
import re
from typing import Dict, List, Sequence, Tuple

from openai import OpenAI

from . import env  # noqa: F401  # ensure .env loaded
from .models import (
    Bullet,
    Job,
    Profile,
    TailoredBullet,
    TailoredEditorDraft,
    TailoredResume,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ACTION_VERBS = {
    "built",
    "designed",
    "led",
    "implemented",
    "developed",
    "improved",
    "optimized",
    "scaled",
    "reduced",
    "increased",
    "owned",
    "deployed",
    "automated",
    "created",
    "launched",
}


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _looks_like_skill_list(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if len(cleaned) < 35:
        return True
    if cleaned.count(",") >= 4 and ":" in cleaned and not any(v in cleaned.lower() for v in ACTION_VERBS):
        return True
    if cleaned.lower().startswith(("programming:", "tools:", "skills:", "ml & ai:", "tech stack:")):
        return True
    return False


def _bullet_quality_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return -10.0
    if _looks_like_skill_list(t):
        return -2.0
    score = 0.0
    words = t.split()
    score += min(1.5, len(words) / 22)
    if re.search(r"\d", t):
        score += 0.7
    if any(v in t.lower() for v in ACTION_VERBS):
        score += 1.0
    if len(t) >= 60:
        score += 0.5
    if t.endswith("."):
        score += 0.2
    return score


def _keyword_set(job: Job) -> List[str]:
    terms = [*(job.requirements.must_have or []), *(job.requirements.nice_to_have or []), *(job.stack or [])]
    seen = set()
    out: List[str] = []
    for term in terms:
        term_clean = (term or "").strip()
        key = _canon(term_clean)
        if not term_clean or not key or key in seen:
            continue
        seen.add(key)
        out.append(term_clean)
    return out


def select_bullets(profile: Profile, job: Job, max_bullets: int = 6) -> List[Bullet]:
    """Pick high-quality bullets that overlap with job requirements/stack/keywords."""
    job_tokens = {tok.lower() for tok in _keyword_set(job)}
    ranked: List[Tuple[float, Bullet]] = []
    for bullet in profile.bullet_bank:
        hit = len(job_tokens & {k.lower() for k in (bullet.skills + bullet.keywords)})
        quality = _bullet_quality_score(bullet.text)
        ranked.append((quality + 1.4 * hit, bullet))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [bullet for score, bullet in ranked if score > 0][:max_bullets]
    if not selected:
        selected = [b for _, b in ranked[:max_bullets]]
    return selected


def _inject_missing_keywords(text: str, keywords: Sequence[str]) -> str:
    updated = text.strip()
    low = _canon(updated)
    for kw in keywords:
        kw_clean = (kw or "").strip()
        if not kw_clean:
            continue
        if _canon(kw_clean) in low:
            continue
        if len(updated) + len(kw_clean) + 3 <= 260:
            updated = f"{updated}; {kw_clean}"
            low = _canon(updated)
    return updated


def _looks_truncated(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    tail = cleaned[-1]
    if tail in {",", ";", ":", "-", "/"}:
        return True
    if re.search(r"\b(and|or|with|to|for|of|in|on|by)\s*$", cleaned, flags=re.I):
        return True
    return False


def rewrite_bullet(
    bullet: Bullet,
    job: Job,
    allowed_keywords: List[str],
    missing_priority_keywords: List[str],
) -> TailoredBullet:
    """
    Rewrite a bullet to mirror JD language without changing facts.
    Guardrails: forbid new numbers and tools not in allowed_keywords.
    """
    client = OpenAI()
    prompt = (
        "Rewrite ONE resume bullet for ATS alignment while preserving facts.\n"
        "Rules:\n"
        "1) Keep the same achievement and factual content.\n"
        "2) Do not invent new metrics, dates, tools, or outcomes.\n"
        "3) Prefer active voice and concise wording.\n"
        "4) Integrate missing JD keywords only if they are semantically true to the original bullet.\n"
        f"Allowed keywords/tools: {allowed_keywords}.\n"
        f"Priority missing JD keywords: {missing_priority_keywords}.\n"
        f"Job title: {job.title}. Company: {job.company}.\n"
        "Return only the rewritten bullet text."
    )
    rewritten = bullet.text
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=(
                f"Original bullet: {bullet.text}\n"
                f"Job requirements: {job.requirements.must_have + job.requirements.nice_to_have}\n"
                f"Job stack: {job.stack}"
            ),
            system=prompt,
            temperature=0.1,
        )
        candidate = (resp.output_text or "").strip()
        if candidate:
            rewritten = candidate
    except Exception:
        rewritten = bullet.text

    # Guardrail: reject suspiciously shortened/truncated rewrites.
    source_clean = (bullet.text or "").strip()
    rewritten_clean = (rewritten or "").strip()
    if source_clean:
        shrink_ratio = len(rewritten_clean) / max(1, len(source_clean))
        if shrink_ratio < 0.72 or _looks_truncated(rewritten_clean):
            rewritten = source_clean
            rewritten_clean = source_clean

    # Deterministic fallback: if model output unchanged, gently inject missing keywords.
    if _canon(rewritten) == _canon(bullet.text) and missing_priority_keywords:
        rewritten = _inject_missing_keywords(rewritten, missing_priority_keywords[:2])

    diff = "\n".join(difflib.ndiff([bullet.text], [rewritten]))
    preview_html = _inline_diff_html(bullet.text, rewritten)
    return TailoredBullet(
        source_id=bullet.id,
        source_text=bullet.text,
        rewritten_text=rewritten,
        diff_from_source=diff,
        preview_html=preview_html,
    )


def validate_tailoring(selected: List[Bullet], rewritten: List[TailoredBullet]) -> Dict[str, object]:
    original_numbers = re.findall(r"\d[\d.,%]*", " ".join(b.text for b in selected))
    new_numbers = re.findall(r"\d[\d.,%]*", " ".join(tb.rewritten_text for tb in rewritten))
    unsupported_numbers = [n for n in new_numbers if n not in original_numbers]
    changed = sum(1 for src, out in zip(selected, rewritten) if _canon(src.text) != _canon(out.rewritten_text))
    return {
        "factuality": "fail" if unsupported_numbers else "pass",
        "new_numbers_added": bool(unsupported_numbers),
        "unsupported_numbers": unsupported_numbers,
        "length_ok": all(len(tb.rewritten_text) <= 260 for tb in rewritten),
        "changed_bullets": changed,
        "total_bullets": len(rewritten),
    }


def _inline_diff_html(source: str, rewritten: str) -> str:
    src_tokens = source.split()
    out_tokens = rewritten.split()
    matcher = difflib.SequenceMatcher(a=src_tokens, b=out_tokens)
    chunks: List[str] = []

    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            chunks.extend(html.escape(tok) for tok in out_tokens[b0:b1])
            continue
        if op == "insert":
            chunks.extend(
                f"<mark class='bg-emerald-100 text-emerald-900 px-0.5 rounded'>{html.escape(tok)}</mark>"
                for tok in out_tokens[b0:b1]
            )
            continue
        if op == "replace":
            chunks.extend(
                f"<mark class='bg-amber-100 text-amber-900 px-0.5 rounded'>{html.escape(tok)}</mark>"
                for tok in out_tokens[b0:b1]
            )
            continue
        if op == "delete":
            deleted = " ".join(src_tokens[a0:a1]).strip()
            if deleted:
                chunks.append(
                    f"<span class='line-through text-slate-400 decoration-slate-400'>[{html.escape(deleted)}]</span>"
                )

    return " ".join(chunks)


def build_default_editor_draft(profile: Profile, job: Job, tailored: TailoredResume) -> TailoredEditorDraft:
    summary = (
        (profile.identity.notes or "").strip()
        or f"{profile.identity.name} aligned for {job.title} roles with relevant technical and project experience."
    )

    skill_candidates: List[str] = []
    for skill in profile.skills:
        if skill.name:
            skill_candidates.append(skill.name)
    for term in _keyword_set(job):
        if len(skill_candidates) >= 20:
            break
        if _canon(term) not in {_canon(s) for s in skill_candidates}:
            skill_candidates.append(term)

    bullets = [tb.rewritten_text for tb in tailored.rewritten_bullets if (tb.rewritten_text or "").strip()]
    if not bullets:
        bullets = [b.text for b in select_bullets(profile, job)]

    return TailoredEditorDraft(
        summary=summary,
        skills=skill_candidates[:20],
        bullets=bullets[:10],
    )


def normalize_editor_draft(draft: TailoredEditorDraft) -> TailoredEditorDraft:
    summary = re.sub(r"\s+", " ", (draft.summary or "").strip())

    skills_seen = set()
    skills: List[str] = []
    for skill in draft.skills:
        cleaned = re.sub(r"\s+", " ", (skill or "").strip())
        key = _canon(cleaned)
        if not key or key in skills_seen:
            continue
        skills_seen.add(key)
        skills.append(cleaned)

    bullets: List[str] = []
    for bullet in draft.bullets:
        cleaned = re.sub(r"\s+", " ", (bullet or "").strip())
        if not cleaned:
            continue
        bullets.append(cleaned)

    return TailoredEditorDraft(summary=summary, skills=skills[:30], bullets=bullets[:16])


def build_preview_html(job: Job, rewritten: List[TailoredBullet]) -> str:
    sections = [
        "<div class='space-y-3'>",
        f"<p class='text-sm font-semibold text-slate-700'>Tailored bullets for {html.escape(job.title or 'role')}</p>",
    ]
    for idx, bullet in enumerate(rewritten, start=1):
        sections.append("<div class='rounded-lg border border-slate-200 p-3 bg-white'>")
        sections.append(f"<p class='text-[11px] font-semibold text-slate-500 mb-1'>Bullet {idx}</p>")
        sections.append(
            "<p class='text-xs text-slate-600 mb-2'><span class='font-semibold'>Original:</span> "
            f"{html.escape(bullet.source_text or '')}</p>"
        )
        sections.append(
            "<p class='text-xs text-slate-700'><span class='font-semibold'>Preview:</span> "
            f"{bullet.preview_html or html.escape(bullet.rewritten_text)}</p>"
        )
        sections.append("</div>")
    sections.append("</div>")
    return "".join(sections)


def render_editor_preview_html(job: Job, draft: TailoredEditorDraft) -> str:
    normalized = normalize_editor_draft(draft)
    summary = html.escape(normalized.summary or "")
    skills = normalized.skills
    bullets = normalized.bullets

    skill_tags = "".join(
        f"<span class='px-2 py-1 rounded bg-slate-100 text-slate-700 text-[11px]'>{html.escape(skill)}</span>"
        for skill in skills[:30]
    )
    bullet_html = "".join(
        f"<li class='mb-2 leading-snug'>{html.escape(bullet)}</li>" for bullet in bullets
    )
    return (
        "<div class='space-y-4'>"
        f"<div><p class='text-xs uppercase tracking-wide text-slate-500 mb-1'>Target Role</p><p class='font-semibold text-slate-800'>{html.escape(job.title)} at {html.escape(job.company)}</p></div>"
        f"<div><p class='text-xs uppercase tracking-wide text-slate-500 mb-1'>Summary</p><p class='text-sm text-slate-700 leading-snug'>{summary}</p></div>"
        f"<div><p class='text-xs uppercase tracking-wide text-slate-500 mb-1'>Skills</p><div class='flex flex-wrap gap-1.5'>{skill_tags}</div></div>"
        f"<div><p class='text-xs uppercase tracking-wide text-slate-500 mb-1'>Experience Bullets</p><ul class='list-disc pl-5 text-sm text-slate-700'>{bullet_html}</ul></div>"
        "</div>"
    )


def render_tailored_resume_text(profile: Profile, job: Job, tailored: TailoredResume) -> str:
    identity = profile.identity
    contact = identity.contact
    lines: List[str] = []

    lines.append(identity.name)
    if contact:
        contact_line = " | ".join(v for v in [contact.email, contact.phone, contact.linkedin, contact.github] if v)
        if contact_line:
            lines.append(contact_line)
    lines.append("")
    lines.append(f"Target Role: {job.title} at {job.company}")
    lines.append("")

    draft = normalize_editor_draft(tailored.editor_draft) if tailored.editor_draft else None
    if draft and draft.summary:
        lines.append("SUMMARY")
        lines.append(draft.summary)
        lines.append("")

    if draft and draft.skills:
        lines.append("SKILLS")
        lines.append(", ".join(draft.skills))
        lines.append("")

    lines.append("EXPERIENCE HIGHLIGHTS")
    if draft and draft.bullets:
        for bullet in draft.bullets:
            lines.append(f"- {bullet}")
    else:
        for bullet in tailored.rewritten_bullets:
            lines.append(f"- {bullet.rewritten_text}")
    lines.append("")

    if tailored.ats_keyword_report:
        before = ", ".join(tailored.ats_keyword_report.get("before", [])[:20])
        after = ", ".join(tailored.ats_keyword_report.get("after", [])[:20])
        lines.append("ATS KEYWORDS")
        if before:
            lines.append(f"Before: {before}")
        if after:
            lines.append(f"After: {after}")
        lines.append("")

    lines.append("NOTE")
    lines.append("Review this tailored draft before submitting applications.")
    return "\n".join(lines).strip() + "\n"


def tailor_resume(profile: Profile, job: Job) -> TailoredResume:
    selected = select_bullets(profile, job)
    allowed_keywords = sorted({kw.lower() for b in selected for kw in (b.skills + b.keywords)})
    missing_priority = [kw for kw in (job.requirements.must_have + job.stack) if (kw or "").strip()][:8]

    rewritten: List[TailoredBullet] = []
    for bullet in selected:
        rewritten.append(
            rewrite_bullet(
                bullet,
                job,
                allowed_keywords=allowed_keywords,
                missing_priority_keywords=missing_priority,
            )
        )

    validation = validate_tailoring(selected, rewritten)
    before_keywords = sorted({kw.lower() for b in selected for kw in (b.skills + b.keywords)})
    after_keywords = sorted(
        {kw.lower() for tb in rewritten for kw in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]+", tb.rewritten_text)}
    )
    ats_keyword_report = {
        "before": before_keywords,
        "after": after_keywords,
        "coverage_delta": len(after_keywords) - len(before_keywords),
    }

    tailored = TailoredResume(
        job_id=job.id,
        selected_bullets=[b.id for b in selected],
        rewritten_bullets=rewritten,
        ats_keyword_report=ats_keyword_report,
        validation=validation,
        preview_html=build_preview_html(job, rewritten),
        exports={},
        human_approval={"status": "pending"},
    )
    tailored.editor_draft = build_default_editor_draft(profile, job, tailored)
    return tailored
