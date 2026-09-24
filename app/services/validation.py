"""Output validation and LLM failure-mode checks."""

from __future__ import annotations

import re
from typing import List

from pydantic import ValidationError

from app.core.constants import INJECTION_PATTERNS, NOT_SPECIFIED, VALID_RECOMMENDATIONS
from app.core.logging import get_logger
from app.models.schemas import CandidateEvaluation, CandidateProfile

logger = get_logger(__name__)


class OutputValidator:
    """Validate structured evaluation outputs and detect unsafe patterns."""

    def detect_injection(self, resume_text: str) -> List[str]:
        warnings: List[str] = []
        lower = resume_text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lower, flags=re.IGNORECASE):
                warnings.append(
                    "Potential prompt-injection / instruction-override text detected in resume; "
                    "content treated as data only."
                )
                break
        return warnings

    def evidence_is_grounded(self, evidence: List[str], resume_text: str) -> List[str]:
        warnings: List[str] = []
        lower = resume_text.lower()
        for item in evidence:
            # Require at least one meaningful token from evidence to appear in resume
            tokens = [t for t in re.findall(r"[a-zA-Z0-9+#.]{4,}", item.lower())]
            if not tokens:
                continue
            if not any(t in lower for t in tokens):
                warnings.append(
                    f"Evidence not grounded in resume text and was flagged: '{item[:80]}'"
                )
        return warnings

    def filter_ungrounded_evidence(
        self, evidence: List[str], resume_text: str
    ) -> List[str]:
        lower = resume_text.lower()
        kept: List[str] = []
        for item in evidence:
            tokens = [t for t in re.findall(r"[a-zA-Z0-9+#.]{4,}", item.lower())]
            if not tokens or any(t in lower for t in tokens):
                kept.append(item)
        return kept

    def skills_supported_by_resume(
        self, claimed_skills: List[str], profile: CandidateProfile
    ) -> List[str]:
        """Return claimed skills that are NOT present in extracted profile or raw text."""
        known = {s.lower() for s in profile.all_skill_tokens()}
        raw = profile.raw_text.lower()
        unsupported = []
        for skill in claimed_skills:
            s = skill.lower()
            if s in known:
                continue
            if re.search(rf"\b{re.escape(s)}\b", raw):
                continue
            unsupported.append(skill)
        return unsupported

    def validate_evaluation(
        self, payload: dict, profile: CandidateProfile
    ) -> tuple[CandidateEvaluation | None, List[str]]:
        warnings: List[str] = []
        try:
            evaluation = CandidateEvaluation.model_validate(payload)
        except ValidationError as exc:
            logger.error("Schema validation failed: %s", exc)
            warnings.append(f"Schema validation failed: {exc.errors()[0]['msg']}")
            return None, warnings

        if evaluation.recommendation not in VALID_RECOMMENDATIONS:
            warnings.append("Invalid recommendation value.")
            return None, warnings

        if not (0 <= evaluation.overall_match <= 100):
            warnings.append("overall_match out of range.")
            return None, warnings

        grounded = self.filter_ungrounded_evidence(
            evaluation.evidence, profile.raw_text
        )
        if len(grounded) < len(evaluation.evidence):
            warnings.append("Removed ungrounded evidence snippets.")
            evaluation.evidence = grounded

        # Hallucination check on matched required skills
        unsupported = self.skills_supported_by_resume(
            evaluation.required_skills.matched, profile
        )
        if unsupported:
            warnings.append(
                "Matched skills not evidenced in resume were removed: "
                + ", ".join(unsupported)
            )
            evaluation.required_skills.matched = [
                s
                for s in evaluation.required_skills.matched
                if s not in unsupported
            ]

        injection = self.detect_injection(profile.raw_text)
        warnings.extend(injection)

        if evaluation.candidate in {"", NOT_SPECIFIED} and profile.name != NOT_SPECIFIED:
            evaluation.candidate = profile.name

        evaluation.validation_warnings = warnings
        return evaluation, warnings
