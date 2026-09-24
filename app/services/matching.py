"""Skill and experience matching utilities (deterministic)."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

from app.core.constants import NOT_SPECIFIED, SKILL_ALIASES
from app.models.schemas import ExperienceMatchResult, SkillMatchResult
from app.utils.helpers import normalize_skill


def expand_aliases(skill: str) -> Set[str]:
    norm = normalize_skill(skill)
    aliases = {norm}
    for canonical, group in SKILL_ALIASES.items():
        if norm == canonical or norm in group:
            aliases |= {normalize_skill(x) for x in group}
            aliases.add(canonical)
    return aliases


def skill_matches(required: str, candidate_skills: Iterable[str]) -> bool:
    """Exact/alias match only — no unsupported inference (Python ≠ Django)."""
    req_aliases = expand_aliases(required)
    for skill in candidate_skills:
        cand_aliases = expand_aliases(skill)
        if req_aliases & cand_aliases:
            return True
        # Also allow substring only for multi-word exact phrase presence
        req = normalize_skill(required)
        cand = normalize_skill(skill)
        if req and cand and (req == cand):
            return True
    return False


def match_skills(
    required_list: List[str], candidate_skills: List[str]
) -> SkillMatchResult:
    matched: List[str] = []
    missing: List[str] = []
    for skill in required_list:
        if skill_matches(skill, candidate_skills):
            matched.append(skill)
        else:
            missing.append(skill)
    total = len(required_list)
    score = (len(matched) / total) if total else 1.0
    return SkillMatchResult(matched=matched, missing=missing, score=round(score, 4))


def match_experience(
    required_years: Optional[float], candidate_years: Optional[float]
) -> ExperienceMatchResult:
    if required_years is None:
        return ExperienceMatchResult(
            required_years=None,
            candidate_years=candidate_years,
            score=1.0 if candidate_years is not None else 0.5,
            note="No experience requirement stated in JD.",
        )
    if candidate_years is None:
        return ExperienceMatchResult(
            required_years=required_years,
            candidate_years=None,
            score=0.0,
            note="Candidate experience not specified.",
        )
    if candidate_years >= required_years:
        # Slight bonus capped at 1.0 for meeting/exceeding
        score = 1.0
        note = "Meets or exceeds required experience."
    else:
        score = max(0.0, candidate_years / required_years)
        note = "Below required experience."
    return ExperienceMatchResult(
        required_years=required_years,
        candidate_years=candidate_years,
        score=round(score, 4),
        note=note,
    )


def responsibility_relevance(
    responsibilities: List[str], resume_text: str
) -> Tuple[float, List[str]]:
    """
    Lightweight lexical overlap for responsibility relevance.
    Returns (score 0-1, evidence snippets).
    """
    if not responsibilities:
        return 0.5, []
    if not resume_text.strip():
        return 0.0, []

    lower = resume_text.lower()
    hits = 0
    evidence: List[str] = []
    for duty in responsibilities:
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z]{4,}", duty.lower())
            if t
            not in {
                "with",
                "from",
                "that",
                "this",
                "have",
                "will",
                "your",
                "their",
                "about",
                "using",
                "into",
                "team",
                "work",
            }
        ]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in lower)
        if overlap >= max(1, len(tokens) // 3):
            hits += 1
            # Pull a short evidence line containing one token
            for line in resume_text.splitlines():
                if any(t in line.lower() for t in tokens[:3]):
                    snippet = line.strip()
                    if snippet and snippet not in evidence:
                        evidence.append(snippet[:220])
                    break
    score = hits / len(responsibilities)
    return round(min(1.0, score), 4), evidence[:5]


def education_match(
    jd_education: List[str],
    jd_certs: List[str],
    candidate_education: List[str],
    candidate_certs: List[str],
) -> Tuple[float, List[str]]:
    requirements = [*(jd_education or []), *(jd_certs or [])]
    if not requirements:
        return 0.7, []  # neutral-positive when JD silent
    haystack = " ".join(candidate_education + candidate_certs).lower()
    matched = 0
    evidence = []
    for req in requirements:
        tokens = [t for t in re.findall(r"[a-zA-Z0-9+]{3,}", req.lower())]
        if tokens and all(t in haystack for t in tokens[:2]):
            matched += 1
            evidence.append(req)
    return round(matched / len(requirements), 4), evidence


def format_experience_display(
    required: Optional[float], candidate: Optional[float]
) -> str:
    req = f"{required:g} years" if required is not None else NOT_SPECIFIED
    cand = f"{candidate:g} years" if candidate is not None else NOT_SPECIFIED
    return f"Required: {req} | Candidate: {cand}"
