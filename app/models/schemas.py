"""Pydantic schemas for JD, resumes, and evaluation results."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.constants import (
    NOT_SPECIFIED,
    RECOMMENDATION_DOES_NOT_MEET,
    RECOMMENDATION_REVIEW,
    RECOMMENDATION_SHORTLIST,
    VALID_RECOMMENDATIONS,
)


class Recommendation(str, Enum):
    SHORTLIST = RECOMMENDATION_SHORTLIST
    REVIEW = RECOMMENDATION_REVIEW
    DOES_NOT_MEET = RECOMMENDATION_DOES_NOT_MEET


class JobDescription(BaseModel):
    """Structured representation of a job description."""

    title: str = NOT_SPECIFIED
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience_years: Optional[float] = None
    education_requirements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    other_requirements: List[str] = Field(default_factory=list)
    raw_text: str = ""
    parsing_notes: List[str] = Field(default_factory=list)

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def clean_skill_lists(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return [str(s).strip() for s in value if str(s).strip()]


class CandidateProfile(BaseModel):
    """Structured candidate information extracted from a resume."""

    name: str = NOT_SPECIFIED
    total_experience_years: Optional[float] = None
    skills: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    cloud_technologies: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    previous_roles: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    domain_experience: List[str] = Field(default_factory=list)
    raw_text: str = ""
    source_filename: str = ""
    extraction_warnings: List[str] = Field(default_factory=list)

    def all_skill_tokens(self) -> List[str]:
        """Flatten all skill-like fields for matching."""
        buckets = [
            self.skills,
            self.programming_languages,
            self.frameworks,
            self.cloud_technologies,
            self.databases,
            self.certifications,
        ]
        seen: set[str] = set()
        result: List[str] = []
        for bucket in buckets:
            for item in bucket:
                key = item.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item.strip())
        return result


class SkillMatchResult(BaseModel):
    matched: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class ExperienceMatchResult(BaseModel):
    required_years: Optional[float] = None
    candidate_years: Optional[float] = None
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    note: str = NOT_SPECIFIED


class ComponentScores(BaseModel):
    required_skills: float = Field(ge=0.0, le=1.0)
    preferred_skills: float = Field(ge=0.0, le=1.0)
    experience: float = Field(ge=0.0, le=1.0)
    responsibilities: float = Field(ge=0.0, le=1.0)
    education: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)


class CandidateEvaluation(BaseModel):
    """Structured evaluation result for one candidate."""

    candidate: str
    overall_match: float = Field(ge=0.0, le=100.0)
    recommendation: str
    experience: ExperienceMatchResult
    required_skills: SkillMatchResult
    preferred_skills: SkillMatchResult
    relevant_experience: str = NOT_SPECIFIED
    potential_gaps: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    explanation: str = NOT_SPECIFIED
    component_scores: ComponentScores
    validation_warnings: List[str] = Field(default_factory=list)
    source_filename: str = ""

    @field_validator("recommendation")
    @classmethod
    def validate_recommendation(cls, value: str) -> str:
        if value not in VALID_RECOMMENDATIONS:
            raise ValueError(
                f"Invalid recommendation '{value}'. "
                f"Must be one of: {sorted(VALID_RECOMMENDATIONS)}"
            )
        return value


class ComparisonRow(BaseModel):
    candidate: str
    match: float
    experience: str
    required_skills_matched: int
    required_skills_total: int
    missing_skills: List[str]
    recommendation: str
    source_filename: str = ""


class ScreeningResult(BaseModel):
    job_title: str
    candidates: List[CandidateEvaluation]
    comparison: List[ComparisonRow]
    processing_notes: List[str] = Field(default_factory=list)
    llm_provider: str = "mock"


class ScreenRequestMeta(BaseModel):
    """Metadata for a screening run (non-file fields)."""

    job_description_text: str = Field(min_length=1)
    use_llm_extraction: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "application_error"
