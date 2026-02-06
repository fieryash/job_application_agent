from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Contact(BaseModel):
    email: str
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class Identity(BaseModel):
    name: str
    targets: List[str] = Field(default_factory=list)
    seniority: Optional[str] = None
    industries: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    contact: Optional[Contact] = None


class Constraints(BaseModel):
    location: List[str] = Field(default_factory=list)
    remote_policy: List[str] = Field(default_factory=list)
    work_authorization: Optional[str] = None
    visa_timeline: Optional[str] = None
    salary_floor_usd: Optional[int] = None
    travel_tolerance: Optional[str] = None
    process_if_salary_missing: bool = True


class Skill(BaseModel):
    name: str
    category: Optional[str] = None
    level: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    source: Optional[str] = None


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    focus: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    gpa: Optional[float] = None
    coursework: List[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: str
    role: str
    location: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)


class Publication(BaseModel):
    id: str
    title: str
    venue: Optional[str] = None


class Bullet(BaseModel):
    id: str
    text: str
    skills: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    metrics: Dict[str, object] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    seniority_signal: List[str] = Field(default_factory=list)
    evidence_links: List[str] = Field(default_factory=list)
    tense: Optional[str] = None


class Profile(BaseModel):
    tag: str
    identity: Identity
    constraints: Constraints
    skills: List[Skill] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    bullet_bank: List[Bullet] = Field(default_factory=list)
    raw_resume_text: Optional[str] = None
    source_resume: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)


class JobRequirements(BaseModel):
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    years_experience_gate: Optional[str] = None
    visa: Optional[str] = None
    onsite_requirement: Optional[str] = None


class JobSource(BaseModel):
    url: Optional[str] = None
    ats: Optional[str] = None
    source_type: Optional[str] = None  # direct | aggregator
    resolved_from: Optional[str] = None


class Job(BaseModel):
    id: str
    company: str
    title: str
    location: Optional[str] = None
    remote_policy: Optional[str] = None
    level: Optional[str] = None
    posted_at: Optional[str] = None
    source: Optional[JobSource] = None
    requirements: JobRequirements = Field(default_factory=JobRequirements)
    stack: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    evaluation_rubric: List[Dict[str, object]] = Field(default_factory=list)
    raw_text: Optional[str] = None
    ingested_at: Optional[str] = None


class FitSubscores(BaseModel):
    skills: float = 0.0
    domain: float = 0.0
    seniority: float = 0.0
    stack: float = 0.0
    story: float = 0.0
    exact: float = 0.0


class ExactMatchResult(BaseModel):
    """Deterministic term/phrase matches (no semantic inference)."""

    score: float = 0.0
    matched: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)


class ScoreComponent(BaseModel):
    key: str
    label: str
    weight: float
    score: float
    contribution: float
    reason: Optional[str] = None


class KeywordCoverage(BaseModel):
    must_have_total: int = 0
    must_have_matched: List[str] = Field(default_factory=list)
    must_have_missing: List[str] = Field(default_factory=list)
    nice_to_have_total: int = 0
    nice_to_have_matched: List[str] = Field(default_factory=list)
    nice_to_have_missing: List[str] = Field(default_factory=list)
    stack_total: int = 0
    stack_matched: List[str] = Field(default_factory=list)
    stack_missing: List[str] = Field(default_factory=list)
    overall_coverage: float = 0.0


class HardFilterResult(BaseModel):
    passed: bool
    reasons: List[str] = Field(default_factory=list)


class FitResult(BaseModel):
    job_id: str
    profile_tag: str
    hard_filter: HardFilterResult
    fit_score: float
    confidence: str
    why: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    subscores: FitSubscores = Field(default_factory=FitSubscores)
    exact_match: ExactMatchResult = Field(default_factory=ExactMatchResult)
    score_components: List[ScoreComponent] = Field(default_factory=list)
    keyword_coverage: KeywordCoverage = Field(default_factory=KeywordCoverage)


class TailoredBullet(BaseModel):
    source_id: str
    source_text: Optional[str] = None
    rewritten_text: str
    diff_from_source: Optional[str] = None
    preview_html: Optional[str] = None


class TailoredEditorDraft(BaseModel):
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    job_id: str
    selected_bullets: List[str] = Field(default_factory=list)
    rewritten_bullets: List[TailoredBullet] = Field(default_factory=list)
    ats_keyword_report: Dict[str, object] = Field(default_factory=dict)
    validation: Dict[str, object] = Field(default_factory=dict)
    preview_html: Optional[str] = None
    editor_draft: Optional[TailoredEditorDraft] = None
    exports: Dict[str, object] = Field(default_factory=dict)
    human_approval: Dict[str, object] = Field(default_factory=dict)


class AutoApplyResult(BaseModel):
    job_id: str
    profile_tag: str
    status: str  # submitted | ready_to_submit | unsupported | failed
    submitted: bool = False
    apply_url: Optional[str] = None
    ats: Optional[str] = None
    message: str
    steps: List[str] = Field(default_factory=list)
