from __future__ import annotations

import re
from datetime import date
from typing import Iterable, List, Optional, Set, Tuple

from .models import ExactMatchResult, FitResult, FitSubscores, HardFilterResult, Job, Profile
from .tech_terms import TECH_TERMS


def _normalize(tokens: Iterable[str]) -> Set[str]:
    """Normalize skill/keyword tokens for comparison."""
    normalized = set()
    for t in tokens:
        if t:
            # Normalize variations: PyTorch == pytorch, TensorFlow == tensorflow
            clean = t.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
            normalized.add(clean)
            # Also add original lowercase for exact matching
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
    # Keep only letters/digits to match formatting differences (e.g. scikit learn vs scikit-learn).
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _term_in_text(term: str, *, text_ws: str, text_compact: str) -> bool:
    term_ws = _ws_normalize(term)
    if not term_ws:
        return False

    # Boundary-aware match on whitespace-collapsed text.
    # Use [a-z0-9] instead of \\w to avoid underscore being treated as a word char.
    pattern = rf"(?<![a-z0-9]){re.escape(term_ws)}(?![a-z0-9])"
    if re.search(pattern, text_ws):
        return True

    # Fallback: match on compacted alnum-only text to tolerate punctuation/spacing changes.
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


def _exact_match(profile: Profile, job: Job) -> ExactMatchResult:
    """
    Deterministic "exact term match" score:
    - Uses job must/nice/stack terms when present.
    - Falls back to a curated tech-term lexicon when the JD is unstructured.
    """
    resume = _resume_text(profile)
    resume_ws = _ws_normalize(resume)
    resume_compact = _compact_normalize(resume)

    must = _dedupe_terms(job.requirements.must_have)
    nice = _dedupe_terms(job.requirements.nice_to_have)
    stack = _dedupe_terms(job.stack)

    # If the structured extraction is weak, enrich terms using deterministic lexicon hits in JD text.
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
        # No usable JD structure. Return low signal rather than a misleading 0/1.
        return ExactMatchResult(score=0.3, matched=[], missing=[])

    weighted = 0.0
    total_w = 0.0
    for _, w, hit, total in present_weights:
        if total <= 0:
            continue
        weighted += w * (hit / total)
        total_w += w
    score = weighted / total_w if total_w > 0 else 0.3

    # Keep payload small and UI-friendly.
    matched = [*matched_must, *matched_stack, *matched_nice][:20]
    missing = [*missing_must, *missing_stack, *missing_nice][:20]
    return ExactMatchResult(score=round(score, 3), matched=matched, missing=missing)


def _skill_overlap(profile: Profile, job: Job) -> Tuple[float, List[str], List[str]]:
    """Calculate skill match score with proper handling of empty requirements."""
    profile_skill_names = _normalize(s.name for s in profile.skills)
    
    # Also include skills from bullet keywords
    bullet_skills = set()
    for b in profile.bullet_bank:
        bullet_skills.update(_normalize(b.keywords))
        bullet_skills.update(_normalize(b.skills))
    
    all_profile_skills = profile_skill_names | bullet_skills
    
    must = _normalize(job.requirements.must_have)
    nice = _normalize(job.requirements.nice_to_have)
    
    matched_must = sorted(must & all_profile_skills)
    missing_must = sorted(must - all_profile_skills)
    matched_nice = sorted(nice & all_profile_skills)
    missing_nice = sorted(nice - all_profile_skills)
    
    # FIX: When no requirements found, return LOW score (not 1.0)
    # This indicates we don't have enough data to score well
    if not must and not nice:
        return 0.3, [], []  # Low score when no requirements to match
    
    # Calculate scores
    if must:
        must_score = len(matched_must) / len(must)
    else:
        must_score = 0.5  # Neutral when no must-haves
    
    if nice:
        nice_score = len(matched_nice) / len(nice)
    else:
        nice_score = 0.0  # Don't boost for missing nice-to-haves
    
    score = 0.7 * must_score + 0.3 * nice_score
    return score, matched_must + matched_nice, missing_must + missing_nice


def _stack_overlap(profile: Profile, job: Job) -> float:
    """Calculate tech stack match score."""
    profile_skill_names = _normalize(s.name for s in profile.skills)
    
    # Include bullet keywords
    for b in profile.bullet_bank:
        profile_skill_names.update(_normalize(b.keywords))
    
    stack = _normalize(job.stack)
    
    if not stack:
        return 0.3  # Low score when no stack data (not 0.5)
    
    overlap = len(stack & profile_skill_names)
    return overlap / len(stack)


def _domain_overlap(profile: Profile, job: Job) -> float:
    """Calculate domain/industry match score."""
    # Collect domains from experience and projects
    domains_from_exp = set()
    for exp in profile.experience:
        domains_from_exp.update(d.lower() for d in exp.domains)
    
    domains_from_projects = set()
    for proj in profile.projects:
        domains_from_projects.update(d.lower() for d in proj.domains)
    
    combined = domains_from_exp | domains_from_projects
    
    job_domains = _detect_job_domains(job)
    if not job_domains:
        return 0.4  # Slightly below neutral when no domain detected
    
    if not combined:
        return 0.3  # Low score if profile has no domains
    
    overlap = len(job_domains & combined)
    return overlap / len(job_domains) if job_domains else 0.4


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
    """Calculate title/role match score."""
    targets = [t.lower().replace("_", " ") for t in profile.identity.targets]
    job_title = (job.title or "").lower()
    
    if not targets:
        return 0.5  # Neutral when no targets set
    
    if not job_title or job_title == "unknown":
        return 0.3  # Low for unknown titles
    
    # Check for direct matches
    for target in targets:
        if target in job_title or job_title in target:
            return 1.0
        # Check partial matches
        target_words = set(target.split())
        title_words = set(job_title.split())
        if len(target_words & title_words) >= 2:
            return 0.8
    
    # Check for related roles
    role_families = {
        "data scientist": ["data scientist", "ml engineer", "machine learning", "research scientist"],
        "machine learning engineer": ["ml engineer", "machine learning", "ai engineer", "data scientist"],
        "software engineer": ["software engineer", "developer", "swe", "backend", "frontend", "fullstack"],
        "ai engineer": ["ai engineer", "ml engineer", "machine learning", "deep learning"],
    }
    
    for target in targets:
        if target in role_families:
            related = role_families[target]
            if any(r in job_title for r in related):
                return 0.7
    
    return 0.4


def _seniority_alignment(profile: Profile, job: Job) -> float:
    """Calculate seniority match score."""
    level = (job.level or "").lower()
    seniority = (profile.identity.seniority or "").lower()
    
    # No data cases
    if not level or level == "unknown":
        return 0.4  # Low when job level unknown
    if not seniority:
        return 0.5  # Neutral when profile seniority not set
    
    # Define level hierarchy
    level_order = ["intern", "entry", "junior", "mid", "senior", "lead", "staff", "principal", "director"]
    
    def get_level_index(s: str) -> int:
        for i, l in enumerate(level_order):
            if l in s:
                return i
        return 4  # Default to mid
    
    profile_level = get_level_index(seniority)
    job_level = get_level_index(level)
    
    diff = abs(profile_level - job_level)
    
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.8
    elif diff == 2:
        return 0.5
    else:
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
        year = int(match.group(1))
        return date(year, 1, 1)
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
    """Calculate narrative/experience fit score."""
    # Keywords from bullets
    bullet_keywords = set()
    for b in profile.bullet_bank:
        bullet_keywords.update(kw.lower() for kw in b.keywords)
    
    # Compare with job stack and responsibilities
    job_keywords = _normalize(job.stack)
    job_keywords.update(_normalize(job.responsibilities))
    
    if not job_keywords:
        return 0.4  # Low when no job data
    
    if not bullet_keywords:
        return 0.3  # Low when no profile bullets
    
    overlap = len(job_keywords & bullet_keywords)
    return min(1.0, overlap / max(5, len(job_keywords) / 2))


def run_hard_filters(profile: Profile, job: Job) -> HardFilterResult:
    """Apply hard filters (location, visa, onsite requirements)."""
    reasons: List[str] = []
    passed = True

    # Location / remote gating
    if job.location and profile.constraints.location:
        if "us_any" not in profile.constraints.location:
            if job.location.lower() not in [loc.lower() for loc in profile.constraints.location]:
                passed = False
                reasons.append(f"location mismatch: job={job.location}")
    
    if job.remote_policy:
        if "remote" in profile.constraints.remote_policy and "remote" not in job.remote_policy.lower():
            reasons.append("prefers remote; job not fully remote")

    # Visa / sponsorship gating
    visa_req = (job.requirements.visa or "").lower()
    if visa_req in {"no", "no_sponsorship", "citizens_only"}:
        if profile.constraints.work_authorization and "opt" in profile.constraints.work_authorization.lower():
            passed = False
            reasons.append("job disallows sponsorship, profile on OPT")

    # Onsite requirement
    onsite = (job.requirements.onsite_requirement or "").lower()
    if onsite in {"5_days", "full_onsite", "onsite"} and "remote" in profile.constraints.remote_policy:
        passed = False
        reasons.append("onsite required, profile prefers remote")

    return HardFilterResult(passed=passed, reasons=reasons)


def score_job(profile: Profile, job: Job) -> FitResult:
    """Calculate comprehensive fit score for a job."""
    hard_filter = run_hard_filters(profile, job)

    skill_score, matched_skills, missing_skills = _skill_overlap(profile, job)
    exact = _exact_match(profile, job)
    stack_score = _stack_overlap(profile, job)
    domain_score = _domain_overlap(profile, job)
    seniority_score = _seniority_alignment(profile, job)
    story_score = _story_fit(profile, job)
    title_score = _title_match(profile, job)

    subscores = FitSubscores(
        skills=round(skill_score, 3),
        domain=round(domain_score, 3),
        seniority=round(seniority_score, 3),
        stack=round(stack_score, 3),
        story=round(story_score, 3),
        exact=exact.score,
    )

    # Weighted scoring with title match bonus
    fit_score = (
        0.35 * skill_score
        + 0.15 * stack_score
        + 0.15 * domain_score
        + 0.10 * seniority_score
        + 0.10 * story_score
        + 0.15 * title_score
    ) * 100

    # Determine confidence based on data quality
    confidence = "high"
    if not job.raw_text:
        confidence = "medium"
    if not job.requirements.must_have and not job.stack:
        confidence = "low"
    if (job.company or "").lower() == "unknown" or (job.title or "").lower() == "unknown":
        confidence = "low"

    # Build explanations
    why = []
    if matched_skills:
        why.append(f"skills matched: {', '.join(matched_skills[:5])}")
    if exact.matched:
        why.append(f"exact JD matches: {', '.join(exact.matched[:5])}")
    if title_score >= 0.7:
        why.append(f"role matches target: {job.title}")
    if stack_score >= 0.5 and job.stack:
        matched_stack = _normalize(job.stack) & _normalize(s.name for s in profile.skills)
        if matched_stack:
            why.append(f"stack matched: {', '.join(list(matched_stack)[:5])}")
    if hard_filter.reasons:
        why.extend(hard_filter.reasons)

    gaps = []
    if missing_skills:
        gaps.append(f"missing skills: {', '.join(missing_skills[:5])}")
    if exact.missing:
        gaps.append(f"missing exact JD terms: {', '.join(exact.missing[:5])}")
    if _detect_job_domains(job) and subscores.domain < 0.5:
        gaps.append("limited domain alignment")
    profile_years = _estimate_profile_years(profile)
    req_years = _parse_years_requirement(job.requirements.years_experience_gate)
    if req_years is not None and profile_years is not None and req_years > profile_years + 3:
        gaps.append(f"seniority mismatch: requires {req_years}+ yrs, profile ~{profile_years} yrs")
    if not job.requirements.must_have and not job.stack and not job.responsibilities:
        gaps.append("job requirements unclear - low confidence score")

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
    )


def score_jobs(profile: Profile, jobs: List[Job]) -> List[FitResult]:
    """Score multiple jobs."""
    return [score_job(profile, job) for job in jobs]
