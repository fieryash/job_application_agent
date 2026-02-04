from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from . import env  # noqa: F401
from .models import (
    Bullet,
    Constraints,
    Contact,
    Education,
    Experience,
    Identity,
    Profile,
    Project,
    Skill,
)
from .tech_terms import TECH_TERMS

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RESUME_PARSE_PROMPT = """You are a resume parser. Extract structured data from this resume.

RESPOND ONLY WITH VALID JSON matching this exact schema:
{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1-XXX-XXX-XXXX",
  "linkedin": "https://linkedin.com/in/username",
  "github": "https://github.com/username",
  "portfolio": "https://portfolio.com",
  "seniority": "intern | entry | mid | senior | lead | staff | principal",
  "target_roles": ["data_scientist", "machine_learning_engineer", "ai_engineer", "software_engineer_ml_ai"],
  "work_authorization": "US Citizen | Green Card | H1B | OPT | Requires Sponsorship",
  "skills": [
    {"name": "Python", "category": "programming", "level": "expert"},
    {"name": "TensorFlow", "category": "ml_framework", "level": "advanced"}
  ],
  "education": [
    {
      "institution": "University Name",
      "degree": "MS",
      "focus": "Computer Science",
      "start": "2020",
      "end": "2022",
      "gpa": 3.8
    }
  ],
  "experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "location": "City, State",
      "start": "Jan 2022",
      "end": "Present",
      "domains": ["fintech", "ml"],
      "bullets": ["Bullet point 1", "Bullet point 2"]
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "url": "https://github.com/...",
      "bullets": ["Description"],
      "domains": ["nlp", "computer_vision"]
    }
  ],
  "missing_fields": ["fields that could not be determined from the resume"]
}

IMPORTANT:
1. Extract actual data from the resume - never make things up
2. For seniority, infer from years of experience and job titles
3. For target_roles, infer from current/past titles and skills
4. List missing_fields for anything you couldn't confidently extract
5. Skill categories: programming, ml_framework, cloud, database, devops, soft_skill
6. Skill levels: beginner, intermediate, advanced, expert
"""


class ParsedResumeSkill(BaseModel):
    name: str = ""
    category: Optional[str] = None
    level: Optional[str] = None


class ParsedResumeEducation(BaseModel):
    institution: str = ""
    degree: Optional[str] = None
    focus: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    gpa: Optional[float] = None


class ParsedResumeExperience(BaseModel):
    company: str = ""
    role: str = ""
    location: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)


class ParsedResumeProject(BaseModel):
    name: str = ""
    url: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    name: str = ""
    email: str = ""
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    seniority: Optional[str] = None
    target_roles: List[str] = Field(default_factory=list)
    work_authorization: Optional[str] = None
    skills: List[ParsedResumeSkill] = Field(default_factory=list)
    education: List[ParsedResumeEducation] = Field(default_factory=list)
    experience: List[ParsedResumeExperience] = Field(default_factory=list)
    projects: List[ParsedResumeProject] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)


def generate_tag(prefix: str = "profile") -> str:
    """Generate a short unique tag to identify ingested profiles."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Pull text from a PDF resume using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _parse_contacts_regex(text: str) -> Contact:
    """Fallback regex-based contact extraction."""
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_match = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text)
    linkedin_match = re.search(r"(linkedin\.com/[^\s]+)", text, re.IGNORECASE)
    github_match = re.search(r"(github\.com/[^\s]+)", text, re.IGNORECASE)
    return Contact(
        email=email_match.group(0) if email_match else "",
        phone=phone_match.group(0).strip() if phone_match else None,
        linkedin=f"https://{linkedin_match.group(0)}" if linkedin_match else None,
        github=f"https://{github_match.group(0)}" if github_match else None,
        portfolio=None,
    )


def _infer_targets_regex(text: str) -> List[str]:
    """Fallback regex-based target inference."""
    lowered = text.lower()
    targets: List[str] = []
    if any(term in lowered for term in ["data scientist", "data science"]):
        targets.append("data_scientist")
    if any(term in lowered for term in ["ml engineer", "machine learning engineer"]):
        targets.append("machine_learning_engineer")
    if "ai engineer" in lowered:
        targets.append("ai_engineer")
    if "software engineer" in lowered and "ml" in lowered:
        targets.append("software_engineer_ml_ai")
    return targets


def _extract_bullets_regex(text: str) -> List[Bullet]:
    """Extract bullet points from resume text."""
    bullets: List[Bullet] = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("•", "-", "▪", "*", "○")):
            clean = stripped.lstrip("•-▪*○").strip()
            if len(clean) > 20:  # Skip very short bullets
                bullets.append(
                    Bullet(
                        id=f"auto_bullet_{idx}",
                        text=clean,
                        skills=[],
                        domain=None,
                        metrics={},
                        keywords=_extract_keywords(clean),
                    )
                )
    return bullets


def _extract_keywords(text: str) -> List[str]:
    """Extract keywords from bullet text."""
    lower = (text or "").lower()
    found: List[str] = []
    seen = set()
    for term in TECH_TERMS:
        t = (term or "").lower()
        if not t or t in seen:
            continue
        if t in lower:
            found.append(t)
            seen.add(t)
    return found


def parse_resume_with_gpt(text: str) -> Optional[ParsedResume]:
    """Use GPT to parse resume into structured data (validated)."""
    client = OpenAI()
    
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": RESUME_PARSE_PROMPT},
                {"role": "user", "content": f"Parse this resume:\n\n{text[:15000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        raw = json.loads(content)
        return ParsedResume.model_validate(raw)
    except (ValidationError, json.JSONDecodeError, Exception):
        return None


def _canon_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _dedupe_skills(skills: List[Skill]) -> List[Skill]:
    seen = set()
    out: List[Skill] = []
    for s in skills:
        key = _canon_key(s.name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _skills_in_text(text: str, skills: List[Skill]) -> List[str]:
    t = (text or "").lower()
    t_compact = _canon_key(t)
    found: List[str] = []
    for s in skills:
        name = (s.name or "").strip()
        if not name:
            continue
        n = name.lower()
        if n and n in t:
            found.append(s.name)
            continue
        key = _canon_key(n)
        if len(key) >= 4 and key in t_compact:
            found.append(s.name)

    # Dedupe while preserving order.
    seen = set()
    out: List[str] = []
    for name in found:
        key = _canon_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _extract_metrics(text: str) -> Dict[str, object]:
    raw = text or ""
    metrics: Dict[str, object] = {}
    pcts = re.findall(r"\b\d+(?:\.\d+)?%\b", raw)
    if pcts:
        metrics["percentages"] = pcts[:5]
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", raw)
    if nums:
        metrics["numbers"] = nums[:8]
    return metrics


def _build_bullet_bank(
    *,
    experience: List[Experience],
    projects: List[Project],
    skills: List[Skill],
    fallback: List[Bullet],
) -> List[Bullet]:
    bullets: List[Bullet] = []
    idx = 0

    for exp in experience:
        domain = exp.domains[0] if exp.domains else None
        for b in exp.bullets or []:
            text = (b or "").strip()
            if len(text) < 20:
                continue
            bullets.append(
                Bullet(
                    id=f"exp_bullet_{idx}",
                    text=text,
                    skills=_skills_in_text(text, skills),
                    domain=domain,
                    metrics=_extract_metrics(text),
                    keywords=_extract_keywords(text),
                )
            )
            idx += 1

    for proj in projects:
        domain = proj.domains[0] if proj.domains else None
        for b in proj.bullets or []:
            text = (b or "").strip()
            if len(text) < 20:
                continue
            bullets.append(
                Bullet(
                    id=f"proj_bullet_{idx}",
                    text=text,
                    skills=_skills_in_text(text, skills),
                    domain=domain,
                    metrics=_extract_metrics(text),
                    keywords=_extract_keywords(text),
                )
            )
            idx += 1

    if bullets:
        return bullets

    # Fallback bullets from regex extraction, enriched with skill/metric tags.
    enriched: List[Bullet] = []
    for b in fallback:
        enriched.append(
            Bullet(
                id=b.id,
                text=b.text,
                skills=_skills_in_text(b.text, skills),
                domain=b.domain,
                metrics=_extract_metrics(b.text),
                keywords=_extract_keywords(b.text),
            )
        )
    return enriched


def build_profile_from_resume_text(
    text: str,
    *,
    base_profile: Optional[Profile] = None,
    explicit_tag: Optional[str] = None,
    source_resume: Optional[str] = None,
) -> Profile:
    """
    Create a Profile from extracted resume text. Uses GPT for structured parsing,
    but builds a deterministic bullet bank from extracted bullets.
    """
    tag = explicit_tag or generate_tag()
    parsed = parse_resume_with_gpt(text) if text else None

    contact_regex = _parse_contacts_regex(text)
    inferred_targets = _infer_targets_regex(text)
    bullets_regex = _extract_bullets_regex(text)
    name_fallback = text.strip().split("\n")[0][:50] if text else "Unknown"

    base = base_profile.model_copy(deep=True) if base_profile else None
    base_contact = base.identity.contact if base and base.identity.contact else None

    name = (parsed.name if parsed and parsed.name else None) or (base.identity.name if base else None) or name_fallback
    email = (parsed.email if parsed and parsed.email else None) or contact_regex.email or (base_contact.email if base_contact else "")
    phone = (parsed.phone if parsed and parsed.phone else None) or contact_regex.phone or (base_contact.phone if base_contact else None)
    linkedin = (parsed.linkedin if parsed and parsed.linkedin else None) or contact_regex.linkedin or (base_contact.linkedin if base_contact else None)
    github = (parsed.github if parsed and parsed.github else None) or contact_regex.github or (base_contact.github if base_contact else None)
    portfolio = (parsed.portfolio if parsed and parsed.portfolio else None) or (base_contact.portfolio if base_contact else None)

    contact = Contact(
        email=(email or "").strip(),
        phone=(phone or None),
        linkedin=(linkedin or None),
        github=(github or None),
        portfolio=(portfolio or None),
    )

    targets = (parsed.target_roles if parsed and parsed.target_roles else None) or inferred_targets or (base.identity.targets if base else [])
    seniority = (parsed.seniority if parsed and parsed.seniority else None) or (base.identity.seniority if base else None)

    # Skills
    extracted_skills: List[Skill] = []
    if parsed:
        for s in parsed.skills:
            name_s = (s.name or "").strip()
            if not name_s:
                continue
            extracted_skills.append(
                Skill(
                    name=name_s,
                    category=s.category,
                    level=s.level,
                    evidence=[],
                    source="resume",
                )
            )
    skills = _dedupe_skills(extracted_skills) or (base.skills if base else [])

    # Education / experience / projects (prefer extracted, fall back to base)
    extracted_edu: List[Education] = []
    extracted_exp: List[Experience] = []
    extracted_proj: List[Project] = []

    if parsed:
        for e in parsed.education:
            if (e.institution or "").strip():
                extracted_edu.append(
                    Education(
                        institution=e.institution,
                        degree=e.degree,
                        focus=e.focus,
                        start=e.start,
                        end=e.end,
                        gpa=e.gpa,
                        coursework=[],
                    )
                )
        for exp in parsed.experience:
            if (exp.company or "").strip() or (exp.role or "").strip():
                extracted_exp.append(
                    Experience(
                        company=exp.company,
                        role=exp.role,
                        location=exp.location,
                        start=exp.start,
                        end=exp.end,
                        domains=exp.domains or [],
                        bullets=[b for b in (exp.bullets or []) if (b or "").strip()],
                    )
                )
        for i, p in enumerate(parsed.projects):
            if (p.name or "").strip():
                extracted_proj.append(
                    Project(
                        id=f"project_{i}",
                        name=p.name,
                        url=p.url,
                        bullets=[b for b in (p.bullets or []) if (b or "").strip()],
                        domains=p.domains or [],
                    )
                )

    education = extracted_edu or (base.education if base else [])
    experience = extracted_exp or (base.experience if base else [])
    projects = extracted_proj or (base.projects if base else [])

    # Constraints (default sensible values when missing)
    constraints = base.constraints.model_copy(deep=True) if base else Constraints()
    if not constraints.location:
        constraints.location = ["us_any", "remote"]
    if not constraints.remote_policy:
        constraints.remote_policy = ["remote", "hybrid"]
    work_auth = (parsed.work_authorization if parsed and parsed.work_authorization else None) or constraints.work_authorization
    constraints.work_authorization = work_auth

    industries = base.identity.industries if base else []
    notes = base.identity.notes if base else None
    identity = Identity(
        name=name,
        contact=contact,
        targets=targets,
        industries=industries,
        notes=notes,
        seniority=seniority,
    )

    bullet_bank = _build_bullet_bank(experience=experience, projects=projects, skills=skills, fallback=bullets_regex)
    if not bullet_bank and base:
        bullet_bank = base.bullet_bank

    publications = base.publications if base else []

    profile = Profile(
        tag=tag,
        identity=identity,
        constraints=constraints,
        skills=skills,
        education=education,
        experience=experience,
        projects=projects,
        publications=publications,
        bullet_bank=bullet_bank,
        raw_resume_text=text or None,
        source_resume=source_resume,
        missing_fields=[],
    )

    missing = set(parsed.missing_fields if parsed else [])
    if not profile.identity.seniority:
        missing.add("seniority")
    if not profile.constraints.work_authorization:
        missing.add("work_authorization")
    if not profile.identity.targets:
        missing.add("target_roles")
    if not profile.skills:
        missing.add("skills")
    if not profile.experience:
        missing.add("experience")
    if not profile.education:
        missing.add("education")
    if not profile.identity.contact.email:
        missing.add("email")
    profile.missing_fields = sorted(missing)

    return profile


def build_profile_from_resume(
    pdf_path: Path,
    base_profile: Optional[Profile] = None,
    explicit_tag: Optional[str] = None,
) -> Profile:
    """Create a Profile from a PDF resume."""
    text = extract_text_from_pdf(pdf_path)
    return build_profile_from_resume_text(
        text,
        base_profile=base_profile,
        explicit_tag=explicit_tag,
        source_resume=str(pdf_path),
    )
