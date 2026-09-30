"""End-to-end screening orchestration."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    CandidateEvaluation,
    ComparisonRow,
    ScreeningResult,
)
from app.services.document_processor import DocumentProcessingError, DocumentProcessor
from app.services.evaluator import CandidateEvaluator
from app.services.jd_processor import JDProcessor
from app.services.llm_client import (
    extract_jd_with_llm,
    extract_resume_with_llm,
    get_llm_client,
)
from app.services.matching import format_experience_display
from app.services.resume_processor import ResumeProcessor
from app.services.validation import OutputValidator
from app.utils.helpers import timed_operation

logger = get_logger(__name__)


class ScreeningService:
    """Coordinates JD/resume processing, evaluation, and comparison."""

    def __init__(
        self, settings: Settings | None = None, prompt_version: str | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.prompt_version = prompt_version
        self.documents = DocumentProcessor(self.settings)
        self.jd_processor = JDProcessor()
        self.resume_processor = ResumeProcessor()
        self.evaluator = CandidateEvaluator(self.settings)
        self.validator = OutputValidator()
        self.llm = get_llm_client(self.settings)

    def screen(
        self,
        job_description_text: str,
        resumes: Sequence[Tuple[str, bytes]],
        use_llm: bool = True,
    ) -> ScreeningResult:
        notes: List[str] = []
        if not job_description_text or not job_description_text.strip():
            raise ValueError("Job description is required.")
        if not resumes:
            raise ValueError("At least one resume is required.")
        if len(resumes) > self.settings.max_resumes_per_request:
            raise ValueError(
                f"Too many resumes. Max allowed: {self.settings.max_resumes_per_request}"
            )

        with timed_operation("screening_run", logger):
            llm_jd = (
                extract_jd_with_llm(
                    self.llm, job_description_text, self.prompt_version
                )
                if use_llm
                else None
            )
            jd = self.jd_processor.process(job_description_text, llm_structured=llm_jd)
            notes.extend(jd.parsing_notes)

            evaluations: List[CandidateEvaluation] = []
            for filename, content in resumes:
                try:
                    text, warnings = self.documents.extract_text(filename, content)
                except DocumentProcessingError as exc:
                    logger.error("Document processing failed for upload: %s", exc)
                    notes.append(f"Skipped '{filename}': {exc}")
                    continue

                injection_warnings = self.validator.detect_injection(text)
                warnings.extend(injection_warnings)

                llm_resume = (
                    extract_resume_with_llm(self.llm, text, self.prompt_version)
                    if use_llm
                    else None
                )
                # Offline regression demo: simulate a bad prompt that invents skills
                if self.prompt_version == "v_regress_bad" and llm_resume is None:
                    llm_resume = {
                        "name": None,
                        "total_experience_years": None,
                        "skills": ["Python", "Django", "FastAPI", "AWS", "Azure", "GCP"],
                        "programming_languages": ["Python"],
                        "frameworks": ["Django", "FastAPI"],
                        "cloud_technologies": ["AWS", "Azure", "GCP"],
                        "databases": [],
                        "certifications": [],
                        "education": [],
                        "previous_roles": [],
                        "projects": [],
                        "domain_experience": [],
                    }
                    notes.append(
                        "v_regress_bad offline simulation: invented related skills for regression demo."
                    )
                profile = self.resume_processor.process(
                    text,
                    filename=filename,
                    extraction_warnings=warnings,
                    llm_structured=llm_resume,
                    strict_grounding=self.prompt_version != "v_regress_bad",
                )
                evaluation = self.evaluator.evaluate(jd, profile)
                evaluations.append(evaluation)

            if not evaluations:
                raise ValueError("No resumes could be processed successfully.")

            comparison = self.build_comparison(evaluations)
            evaluations_sorted = sorted(
                evaluations, key=lambda e: e.overall_match, reverse=True
            )
            comparison_sorted = sorted(
                comparison, key=lambda r: r.match, reverse=True
            )

            return ScreeningResult(
                job_title=jd.title,
                candidates=evaluations_sorted,
                comparison=comparison_sorted,
                processing_notes=notes,
                llm_provider=self.settings.llm_provider,
            )

    def screen_text_resumes(
        self,
        job_description_text: str,
        resume_texts: Sequence[Tuple[str, str]],
        use_llm: bool = False,
    ) -> ScreeningResult:
        """Convenience for tests / evaluation harness (no file I/O)."""
        payloads = [(name, text.encode("utf-8")) for name, text in resume_texts]
        # Force .txt names
        normalized = []
        for name, raw in payloads:
            fname = name if name.lower().endswith((".txt", ".md")) else f"{name}.txt"
            normalized.append((fname, raw))
        return self.screen(job_description_text, normalized, use_llm=use_llm)

    @staticmethod
    def build_comparison(
        evaluations: List[CandidateEvaluation],
    ) -> List[ComparisonRow]:
        rows: List[ComparisonRow] = []
        for ev in evaluations:
            total = len(ev.required_skills.matched) + len(ev.required_skills.missing)
            rows.append(
                ComparisonRow(
                    candidate=ev.candidate,
                    match=ev.overall_match,
                    experience=format_experience_display(
                        ev.experience.required_years, ev.experience.candidate_years
                    ),
                    required_skills_matched=len(ev.required_skills.matched),
                    required_skills_total=total,
                    missing_skills=list(ev.required_skills.missing),
                    recommendation=ev.recommendation,
                    source_filename=ev.source_filename,
                )
            )
        return rows
