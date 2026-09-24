"""Candidate evaluation and scoring."""

from __future__ import annotations

from typing import List

from app.core.config import Settings, get_settings
from app.core.constants import (
    NOT_SPECIFIED,
    RECOMMENDATION_DOES_NOT_MEET,
    RECOMMENDATION_REVIEW,
    RECOMMENDATION_SHORTLIST,
)
from app.core.logging import get_logger
from app.models.schemas import (
    CandidateEvaluation,
    CandidateProfile,
    ComponentScores,
    JobDescription,
)
from app.services.matching import (
    education_match,
    format_experience_display,
    match_experience,
    match_skills,
    responsibility_relevance,
)
from app.services.validation import OutputValidator

logger = get_logger(__name__)


class CandidateEvaluator:
    """Deterministic weighted scoring with transparent methodology."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.validator = OutputValidator()

    def evaluate(
        self, jd: JobDescription, profile: CandidateProfile
    ) -> CandidateEvaluation:
        logger.info(
            "Evaluating candidate source=%s",
            profile.source_filename or "inline",
        )
        weights = self.settings.matching_weights()
        candidate_skills = profile.all_skill_tokens()

        required = match_skills(jd.required_skills, candidate_skills)
        preferred = match_skills(jd.preferred_skills, candidate_skills)
        experience = match_experience(
            jd.min_experience_years, profile.total_experience_years
        )
        resp_score, resp_evidence = responsibility_relevance(
            jd.responsibilities, profile.raw_text
        )
        edu_score, edu_evidence = education_match(
            jd.education_requirements,
            jd.certifications,
            profile.education,
            profile.certifications,
        )

        component = ComponentScores(
            required_skills=required.score,
            preferred_skills=preferred.score,
            experience=experience.score,
            responsibilities=resp_score,
            education=edu_score,
            weights=weights,
        )

        # Prefer JD required skills weight; if JD has zero required skills, redistribute
        overall_unit = self._weighted_score(component, weights)
        overall = round(overall_unit * 100, 2)
        recommendation = self._recommend(overall)

        gaps: List[str] = []
        gaps.extend([f"Missing required skill: {s}" for s in required.missing])
        if experience.candidate_years is None and jd.min_experience_years is not None:
            gaps.append("Experience years not specified on resume.")
        elif (
            experience.candidate_years is not None
            and jd.min_experience_years is not None
            and experience.candidate_years < jd.min_experience_years
        ):
            gaps.append(
                f"Experience below requirement ({experience.candidate_years:g} < {jd.min_experience_years:g} years)."
            )
        if preferred.missing:
            gaps.append(
                "Missing preferred skills: " + ", ".join(preferred.missing[:5])
            )

        evidence: List[str] = []
        evidence.extend(resp_evidence)
        evidence.extend(edu_evidence)
        # Add grounded skill mentions
        for skill in required.matched[:5]:
            for line in profile.raw_text.splitlines():
                if skill.lower() in line.lower():
                    evidence.append(line.strip()[:220])
                    break

        evidence = self.validator.filter_ungrounded_evidence(
            evidence, profile.raw_text
        )[:8]

        relevant = self._relevant_summary(profile)
        explanation = self._explanation(
            overall, recommendation, required, preferred, experience
        )

        name = profile.name if profile.name != NOT_SPECIFIED else (
            profile.source_filename or "Unknown Candidate"
        )

        payload = {
            "candidate": name,
            "overall_match": overall,
            "recommendation": recommendation,
            "experience": experience.model_dump(),
            "required_skills": required.model_dump(),
            "preferred_skills": preferred.model_dump(),
            "relevant_experience": relevant,
            "potential_gaps": gaps,
            "evidence": evidence,
            "explanation": explanation,
            "component_scores": component.model_dump(),
            "validation_warnings": [],
            "source_filename": profile.source_filename,
        }

        evaluation, warnings = self.validator.validate_evaluation(payload, profile)
        if evaluation is None:
            # Safe fallback — should be rare since we build valid payloads
            logger.error("Built evaluation failed validation; using safe fallback")
            return CandidateEvaluation(
                candidate=name,
                overall_match=0.0,
                recommendation=RECOMMENDATION_DOES_NOT_MEET,
                experience=experience,
                required_skills=required,
                preferred_skills=preferred,
                relevant_experience=NOT_SPECIFIED,
                potential_gaps=["Evaluation failed schema validation."],
                evidence=[],
                explanation="Evaluation could not be validated safely.",
                component_scores=component,
                validation_warnings=warnings,
                source_filename=profile.source_filename,
            )
        evaluation.validation_warnings = list(
            dict.fromkeys(evaluation.validation_warnings + profile.extraction_warnings)
        )
        return evaluation

    def _weighted_score(self, component: ComponentScores, weights: dict) -> float:
        total_w = sum(weights.values()) or 1.0
        score = (
            component.required_skills * weights["required_skills"]
            + component.preferred_skills * weights["preferred_skills"]
            + component.experience * weights["experience"]
            + component.responsibilities * weights["responsibilities"]
            + component.education * weights["education"]
        )
        return score / total_w

    def _recommend(self, overall: float) -> str:
        if overall >= self.settings.threshold_shortlist:
            return RECOMMENDATION_SHORTLIST
        if overall >= self.settings.threshold_review:
            return RECOMMENDATION_REVIEW
        return RECOMMENDATION_DOES_NOT_MEET

    def _relevant_summary(self, profile: CandidateProfile) -> str:
        parts = []
        if profile.previous_roles:
            parts.append("Roles: " + "; ".join(profile.previous_roles[:3]))
        if profile.projects:
            parts.append("Projects: " + "; ".join(profile.projects[:2]))
        if profile.total_experience_years is not None:
            parts.append(f"Stated experience: {profile.total_experience_years:g} years")
        return " | ".join(parts) if parts else NOT_SPECIFIED

    def _explanation(
        self, overall, recommendation, required, preferred, experience
    ) -> str:
        return (
            f"Overall match {overall:.1f}/100 → {recommendation}. "
            f"Required skills {len(required.matched)}/{len(required.matched)+len(required.missing)} matched; "
            f"preferred {len(preferred.matched)}/{len(preferred.matched)+len(preferred.missing)} matched. "
            f"Experience: {format_experience_display(experience.required_years, experience.candidate_years)}. "
            f"Score uses configurable weights (required/preferred/experience/responsibilities/education)."
        )
