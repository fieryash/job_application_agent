from __future__ import annotations

import re
from datetime import date
from typing import Iterable, List, Optional, Set, Tuple

from .models import (
    ExactMatchResult,
    FitResult,
    FitSubscores,
    HardFilterResult,
    Job,
    KeywordCoverage,
    Profile,
    ScoreComponent,
)
from .tech_terms import TECH_TERMS


def _normalize(tokens: Iterable[str]) -> Set[str]:
    """Normalize skill/keyword tokens for loose comparison."""
    normalized = set()
    for t in tokens:
        if t:
            clean = t.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
            normalized.add(clean)
            normalized.add(t.lower().strip())
    return normalized


def _resume_text(profile: Profile) -> str:
    """
    Best-effort resume text for deterministic matching.
    Prefer raw_resume_text; fall back to structured bullets/skills.
    """
    if profile.raw_resume_text:
        return profile.raw_resume_text

    parts: List[str] = []
    parts.extend(s.name for s in profile.skills if s.name)
    parts.extend(b.text for b in profile.bullet_bank if b.text)
    for exp in profile.experience:
        parts.append(exp.company)
        parts.append(exp.role)
        parts.extend(exp.bullets)
    for proj in profile.projects:
        parts.append(proj.name)
        parts.extend(proj.bullets)
    return "\n".join(p for p in parts if p)


def _ws_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _compact_normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _term_in_text(term: str, *, text_ws: str, text_compact: str) -> bool:
    term_ws = _ws_normalize(term)
    if not term_ws:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(term_ws)}(?![a-z0-9])"
    if re.search(pattern, text_ws):
        return True

    term_compact = _compact_normalize(term_ws)
    if len(term_compact) >= 4 and term_compact in text_compact:
        return True

    return False


def _dedupe_terms(terms: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        canon = _ws_normalize(t)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        out.append(t.strip())
    return out


def _term_list_overlap(
    terms: Iterable[str],
    *,
    resume_ws: str,
    resume_compact: str,
) -> Tuple[List[str], List[str], List[str]]:
    clean = _dedupe_terms(terms)
    matched = [t for t in clean if _term_in_text(t, text_ws=resume_ws, text_compact=resume_compact)]
    missing = [t for t in clean if t not in matched]
    return clean, matched, missing


def _build_keyword_coverage(profile: Profile, job: Job) -> KeywordCoverage:
    resume = _resume_text(profile)
    resume_ws = _ws_normalize(resume)
    resume_compact = _compact_normalize(resume)

    must, matched_must, missing_must = _term_list_overlap(
        job.requirements.must_have,
        resume_ws=resume_ws,
        resume_compact=resume_compact,
    )
    nice, matched_nice, missing_nice = _term_list_overlap(
        job.requirements.nice_to_have,
        resume_ws=resume_ws,
        resume_compact=resume_compact,
    )
    stack_terms = _dedupe_terms(job.stack)

    # If parsed stack is missing but raw JD exists, recover a deterministic stack from TECH_TERMS.
    if not stack_terms and job.raw_text:
        jd_ws = _ws_normalize(job.raw_text)
        stack_terms = [term for term in TECH_TERMS if _ws_normalize(term) in jd_ws]
        stack_terms = _dedupe_terms(stack_terms)

    _, matched_stack, missing_stack = _term_list_overlap(
        stack_terms,
        resume_ws=resume_ws,
        resume_compact=resume_compact,
    )

    denom = len(must) + len(nice) + len(stack_terms)
    if denom > 0:
        overall = (len(matched_must) + len(matched_nice) + len(matched_stack)) / denom
    else:
        overall = 0.3

    return KeywordCoverage(
        must_have_total=len(must),
        must_have_matched=matched_must,
        must_have_missing=missing_must,
        nice_to_have_total=len(nice),
        nice_to_have_matched=matched_nice,
        nice_to_have_missing=missing_nice,
        stack_total=len(stack_terms),
        stack_matched=matched_stack,
        stack_missing=missing_stack,
        overall_coverage=round(overall, 3),
    )


def _exact_match(profile: Profile, job: Job) -> ExactMatchResult:
    """
    Deterministic exact-term match:
    - Uses job must/nice/stack terms when present.
    - Falls back to TECH_TERMS lexicon when the JD is weak.
    """
    resume = _resume_text(profile)
    resume_ws = _ws_normalize(resume)
    resume_compact = _compact_normalize(resume)

    must = _dedupe_terms(job.requirements.must_have)
    nice = _dedupe_terms(job.requirements.nice_to_have)
    stack = _dedupe_terms(job.stack)

    jd_ws = _ws_normalize(job.raw_text or "")
    if jd_ws and not (must or nice or stack):
        detected = [t for t in TECH_TERMS if _ws_normalize(t) in jd_ws]
        stack = _dedupe_terms(detected)

    matched_must = [t for t in must if _term_in_text(t, text_ws=resume_ws, text_compact=resume_compact)]
    missing_must = [t for t in must if t not in matched_must]

    matched_nice = [t for t in nice if _term_in_text(t, text_ws=resume_ws, text_compact=resume_compact)]
    missing_nice = [t for t in nice if t not in matched_nice]

    matched_stack = [t for t in stack if _term_in_text(t, text_ws=resume_ws, text_compact=resume_compact)]
    missing_stack = [t for t in stack if t not in matched_stack]

    present_weights = []
    if must:
        present_weights.append(("must", 0.7, len(matched_must), len(must)))
    if stack:
        present_weights.append(("stack", 0.2, len(matched_stack), len(stack)))
    if nice:
        present_weights.append(("nice", 0.1, len(matched_nice), len(nice)))

    if not present_weights:
        return ExactMatchResult(score=0.3, matched=[], missing=[])

    weighted = 0.0
    total_w = 0.0
    for _, w, hit, total in present_weights:
        if total <= 0:
            continue
        weighted += w * (hit / total)
        total_w += w
    score = weighted / total_w if total_w > 0 else 0.3

    matched = [*matched_must, *matched_stack, *matched_nice][:30]
    missing = [*missing_must, *missing_stack, *missing_nice][:30]
    return ExactMatchResult(score=round(score, 3), matched=matched, missing=missing)


def _skill_overlap(coverage: KeywordCoverage) -> float:
    """Skill score from must/nice requirement coverage."""
    if coverage.must_have_total == 0 and coverage.nice_to_have_total == 0:
        return 0.3

    if coverage.must_have_total > 0:
        must_score = len(coverage.must_have_matched) / coverage.must_have_total
    else:
        must_score = 0.5

    if coverage.nice_to_have_total > 0:
        nice_score = len(coverage.nice_to_have_matched) / coverage.nice_to_have_total
    else:
        nice_score = 0.4

    if coverage.must_have_total > 0 and coverage.nice_to_have_total > 0:
        return 0.75 * must_score + 0.25 * nice_score
    if coverage.must_have_total > 0:
        return must_score
    return 0.6 * nice_score


def _stack_overlap(coverage: KeywordCoverage) -> float:
    if coverage.stack_total == 0:
        return 0.3
    return len(coverage.stack_matched) / coverage.stack_total


def _domain_overlap(profile: Profile, job: Job) -> float:
    domains_from_exp = set()
    for exp in profile.experience:
        domains_from_exp.update(d.lower() for d in exp.domains)

    domains_from_projects = set()
    for proj in profile.projects:
        domains_from_projects.update(d.lower() for d in proj.domains)

    combined = domains_from_exp | domains_from_projects
    job_domains = _detect_job_domains(job)
    if not job_domains:
        return 0.4
    if not combined:
        return 0.3

    overlap = len(job_domains & combined)
    return overlap / len(job_domains)


def _detect_job_domains(job: Job) -> Set[str]:
    job_domains: Set[str] = set()
    job_text = " ".join(job.responsibilities) + " " + (job.title or "") + " " + " ".join(job.stack)
    job_lower = job_text.lower()

    domain_keywords = {
        "fintech": ["fintech", "financial", "banking", "payments", "trading"],
        "healthcare": ["healthcare", "health", "medical", "biotech", "pharma"],
        "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace"],
        "adtech": ["adtech", "advertising", "marketing tech"],
        "legaltech": ["legal", "law", "compliance"],
        "edtech": ["education", "edtech", "learning"],
        "saas": ["saas", "b2b", "enterprise"],
        "ai_ml": ["machine learning", "ai", "deep learning", "nlp", "computer vision"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in job_lower for kw in keywords):
            job_domains.add(domain)
    return job_domains


def _title_match(profile: Profile, job: Job) -> float:
    targets = [t.lower().replace("_", " ") for t in profile.identity.targets]
    job_title = (job.title or "").lower()

    if not targets:
        return 0.5
    if not job_title or job_title == "unknown":
        return 0.3

    for target in targets:
        if target in job_title or job_title in target:
            return 1.0
        target_words = set(target.split())
        title_words = set(job_title.split())
        if len(target_words & title_words) >= 2:
            return 0.8

    role_families = {
        "data scientist": ["data scientist", "ml engineer", "machine learning", "research scientist"],
        "machine learning engineer": ["ml engineer", "machine learning", "ai engineer", "data scientist"],
        "software engineer": ["software engineer", "developer", "swe", "backend", "frontend", "fullstack"],
        "ai engineer": ["ai engineer", "ml engineer", "machine learning", "deep learning"],
    }

    for target in targets:
        if target in role_families and any(r in job_title for r in role_families[target]):
            return 0.7

    return 0.4


def _seniority_alignment(profile: Profile, job: Job) -> float:
    level = (job.level or "").lower()
    seniority = (profile.identity.seniority or "").lower()

    if not level or level == "unknown":
        return 0.4
    if not seniority:
        return 0.5

    level_order = ["intern", "entry", "junior", "mid", "senior", "lead", "staff", "principal", "director"]

    def get_level_index(s: str) -> int:
        for i, lvl in enumerate(level_order):
            if lvl in s:
                return i
        return 4

    profile_level = get_level_index(seniority)
    job_level = get_level_index(level)
    diff = abs(profile_level - job_level)

    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.8
    if diff == 2:
        return 0.55
    return 0.3


def _parse_years_requirement(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    return numbers[0] if numbers else None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"present", "current", "now"}:
        return date.today()
    match = re.match(r"(\d{4})[-/](\d{1,2})", cleaned)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        return date(year, month, 1)
    match = re.match(r"(\d{4})", cleaned)
    if match:
        return date(int(match.group(1)), 1, 1)
    return None


def _estimate_profile_years(profile: Profile) -> Optional[float]:
    total_months = 0
    for exp in profile.experience:
        start = _parse_date(exp.start)
        end = _parse_date(exp.end) or date.today()
        if not start:
            continue
        months = (end.year - start.year) * 12 + (end.month - start.month)
        if months > 0:
            total_months += months
    if total_months <= 0:
        return None
    return round(total_months / 12, 1)


def _story_fit(profile: Profile, job: Job) -> float:
    """
    Compare action-words and topical tokens between profile bullets and JD narrative.
    This is intentionally shallow and deterministic.
    """
    job_text = " ".join(job.responsibilities + job.requirements.must_have + job.stack)
    profile_text = " ".join([b.text for b in profile.bullet_bank] + [s.name for s in profile.skills])

    if not job_text.strip():
        return 0.4
    if not profile_text.strip():
        return 0.3

    token_pattern = r"[a-zA-Z][a-zA-Z0-9_+.-]{2,}"
    stop = {
        "and",
        "with",
        "for",
        "the",
        "you",
        "your",
        "that",
        "this",
        "will",
        "from",
        "into",
        "our",
        "are",
        "have",
        "using",
        "build",
        "develop",
    }
    job_tokens = {t.lower() for t in re.findall(token_pattern, job_text) if t.lower() not in stop}
    profile_tokens = {t.lower() for t in re.findall(token_pattern, profile_text) if t.lower() not in stop}

    if not job_tokens:
        return 0.4
    if not profile_tokens:
        return 0.3

    overlap = len(job_tokens & profile_tokens)
    return min(1.0, overlap / max(6, int(len(job_tokens) * 0.5)))


def run_hard_filters(profile: Profile, job: Job) -> HardFilterResult:
    """Apply hard filters (location, visa, onsite requirements)."""
    reasons: List[str] = []
    passed = True

    if job.location and profile.constraints.location:
        allowed_locations = [loc.lower() for loc in profile.constraints.location]
        job_loc = job.location.lower()
        if "us_any" not in allowed_locations and "remote" not in allowed_locations:
            if not any(loc in job_loc or job_loc in loc for loc in allowed_locations):
                passed = False
                reasons.append(f"location mismatch: job={job.location}")

    if job.remote_policy:
        if "remote" in profile.constraints.remote_policy and "remote" not in job.remote_policy.lower():
            reasons.append("prefers remote; job not fully remote")

    visa_req = (job.requirements.visa or "").lower()
    if visa_req in {"no", "no_sponsorship", "citizens_only"}:
        if profile.constraints.work_authorization and "opt" in profile.constraints.work_authorization.lower():
            passed = False
            reasons.append("job disallows sponsorship, profile on OPT")

    onsite = (job.requirements.onsite_requirement or "").lower()
    if onsite in {"5_days", "full_onsite", "onsite"} and "remote" in profile.constraints.remote_policy:
        passed = False
        reasons.append("onsite required, profile prefers remote")

    return HardFilterResult(passed=passed, reasons=reasons)


def _component_reason(component_key: str, score: float, coverage: KeywordCoverage, job: Job) -> str:
    if component_key == "skills":
        if coverage.must_have_total == 0 and coverage.nice_to_have_total == 0:
            return "job did not provide enough structured requirements"
        if coverage.must_have_missing:
            return f"missing must-have keywords: {', '.join(coverage.must_have_missing[:4])}"
        return "strong requirement alignment"
    if component_key == "exact":
        if score >= 0.7:
            return "resume language overlaps with JD terms"
        return "limited exact phrase overlap with JD"
    if component_key == "stack":
        if coverage.stack_total == 0:
            return "job stack not clearly extracted"
        if coverage.stack_missing:
            return f"missing stack terms: {', '.join(coverage.stack_missing[:4])}"
        return "strong stack overlap"
    if component_key == "domain":
        return "domain alignment inferred from experience/projects"
    if component_key == "seniority":
        if score < 0.6:
            return "seniority level appears misaligned"
        return "seniority level is aligned"
    if component_key == "story":
        return "bullet narrative overlap with job responsibilities"
    if component_key == "title":
        if score >= 0.7:
            return f"title aligns with target role ({job.title})"
        return "title weakly aligned with current targets"
    return "component scored"


def _unique_append(items: List[str], text: str) -> None:
    if text and text not in items:
        items.append(text)


def score_job(profile: Profile, job: Job) -> FitResult:
    """Calculate comprehensive fit score with explainable output."""
    hard_filter = run_hard_filters(profile, job)
    coverage = _build_keyword_coverage(profile, job)
    exact = _exact_match(profile, job)

    skill_score = _skill_overlap(coverage)
    stack_score = _stack_overlap(coverage)
    domain_score = _domain_overlap(profile, job)
    seniority_score = _seniority_alignment(profile, job)
    story_score = _story_fit(profile, job)
    title_score = _title_match(profile, job)

    # Keep FitSubscores backward compatible.
    subscores = FitSubscores(
        skills=round(skill_score, 3),
        domain=round(domain_score, 3),
        seniority=round(seniority_score, 3),
        stack=round(stack_score, 3),
        story=round(story_score, 3),
        exact=exact.score,
    )

    weights = {
        "skills": 0.28,
        "exact": 0.17,
        "stack": 0.14,
        "domain": 0.12,
        "seniority": 0.10,
        "story": 0.09,
        "title": 0.10,
    }
    component_scores = {
        "skills": skill_score,
        "exact": exact.score,
        "stack": stack_score,
        "domain": domain_score,
        "seniority": seniority_score,
        "story": story_score,
        "title": title_score,
    }
    labels = {
        "skills": "Requirements",
        "exact": "Exact Keywords",
        "stack": "Tech Stack",
        "domain": "Domain",
        "seniority": "Seniority",
        "story": "Experience Story",
        "title": "Role Title",
    }

    base_score = sum(weights[k] * component_scores[k] for k in weights)
    fit_score = base_score * 100
    if not hard_filter.passed:
        fit_score *= 0.65

    components: List[ScoreComponent] = []
    for key in weights:
        score = component_scores[key]
        components.append(
            ScoreComponent(
                key=key,
                label=labels[key],
                weight=weights[key],
                score=round(score, 3),
                contribution=round(weights[key] * score * 100, 1),
                reason=_component_reason(key, score, coverage, job),
            )
        )

    # Confidence based on parse quality.
    signal = 0
    if (job.company or "").lower() not in {"", "unknown"} and (job.title or "").lower() not in {"", "unknown"}:
        signal += 1
    if job.requirements.must_have or job.stack or job.requirements.nice_to_have:
        signal += 1
    if job.responsibilities:
        signal += 1
    if job.raw_text and len(job.raw_text) > 500:
        signal += 1

    if signal >= 4:
        confidence = "high"
    elif signal >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    why: List[str] = []
    gaps: List[str] = []

    strong_components = sorted(
        [c for c in components if c.score >= 0.7],
        key=lambda c: c.contribution,
        reverse=True,
    )
    weak_components = sorted(
        [c for c in components if c.score < 0.45],
        key=lambda c: c.contribution,
        reverse=True,
    )

    if fit_score >= 75 and strong_components:
        _unique_append(why, f"high overall fit driven by {strong_components[0].label.lower()}")
    elif fit_score < 50 and weak_components:
        _unique_append(gaps, f"low overall fit driven by {weak_components[0].label.lower()}")

    if coverage.must_have_matched:
        _unique_append(why, f"matched must-have keywords: {', '.join(coverage.must_have_matched[:6])}")
    if coverage.stack_matched:
        _unique_append(why, f"matched stack keywords: {', '.join(coverage.stack_matched[:6])}")
    if exact.matched:
        _unique_append(why, f"exact JD matches: {', '.join(exact.matched[:6])}")
    if title_score >= 0.7:
        _unique_append(why, f"role title aligns with targets: {job.title}")
    for c in strong_components[:2]:
        if c.reason:
            _unique_append(why, f"{c.label}: {c.reason}")
    for reason in hard_filter.reasons:
        _unique_append(why, reason)

    if coverage.must_have_missing:
        _unique_append(gaps, f"missing must-have keywords: {', '.join(coverage.must_have_missing[:8])}")
    if coverage.stack_missing:
        _unique_append(gaps, f"missing stack keywords: {', '.join(coverage.stack_missing[:8])}")
    if exact.missing:
        _unique_append(gaps, f"missing exact JD terms: {', '.join(exact.missing[:8])}")
    if _detect_job_domains(job) and subscores.domain < 0.5:
        _unique_append(gaps, "limited domain alignment with this industry")

    profile_years = _estimate_profile_years(profile)
    req_years = _parse_years_requirement(job.requirements.years_experience_gate)
    if req_years is not None and profile_years is not None and req_years > profile_years + 3:
        _unique_append(gaps, f"seniority mismatch: requires {req_years}+ yrs, profile ~{profile_years} yrs")

    if coverage.must_have_total == 0 and coverage.stack_total == 0 and not job.responsibilities:
        _unique_append(gaps, "job requirements were unclear, reducing confidence")

    for c in weak_components[:2]:
        if c.reason:
            _unique_append(gaps, f"{c.label}: {c.reason}")

    return FitResult(
        job_id=job.id,
        profile_tag=profile.tag,
        hard_filter=hard_filter,
        fit_score=round(fit_score, 1),
        confidence=confidence,
        why=why,
        gaps=gaps,
        subscores=subscores,
        exact_match=exact,
        score_components=components,
        keyword_coverage=coverage,
    )


def score_jobs(profile: Profile, jobs: List[Job]) -> List[FitResult]:
    """Score multiple jobs."""
    return [score_job(profile, job) for job in jobs]
