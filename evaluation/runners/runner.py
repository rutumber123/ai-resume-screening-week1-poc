"""Automated LLM evaluation runner against the Week 1 ScreeningService."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app import __version__
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.jd_processor import JDProcessor
from app.services.screening import ScreeningService
from evaluation.configuration.settings import EvalSettings, get_eval_settings
from evaluation.dataset import filter_cases, load_catalog, read_text
from evaluation.evaluators.dimensions import (
    classify_failures,
    evaluate_completeness,
    evaluate_correctness,
    evaluate_groundedness,
    evaluate_relevance,
    evaluate_schema,
)
from evaluation.evaluators.llm_judge import judge_to_dimensions, llm_judge_scores
from evaluation.metrics.aggregate import aggregate
from evaluation.metrics.schemas import CaseResult, EvaluationRun
from evaluation.regression.compare import compare_runs
from evaluation.reports.generator import write_report
from evaluation.runners.storage import find_baseline_run_id, load_run, save_run

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class EvaluationRunner:
    def __init__(
        self,
        prompt_version: str = "v2",
        app_settings: Settings | None = None,
        eval_settings: EvalSettings | None = None,
        use_llm: bool = False,
    ) -> None:
        self.prompt_version = prompt_version
        self.app_settings = app_settings or get_settings()
        self.eval_settings = eval_settings or get_eval_settings()
        self.use_llm = use_llm
        self.service = ScreeningService(
            self.app_settings, prompt_version=prompt_version
        )
        self.jd_processor = JDProcessor()
        self._score_cache: Dict[str, float] = {}

    def run(
        self,
        categories: Optional[Iterable[str]] = None,
        test_ids: Optional[Iterable[str]] = None,
        baseline_run_id: Optional[str] = None,
        repeats: Optional[int] = None,
    ) -> EvaluationRun:
        catalog = load_catalog()
        cases = filter_cases(catalog, categories=categories, test_ids=test_ids)
        run_id = f"eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(
            "Starting evaluation run_id=%s cases=%s prompt=%s",
            run_id,
            len(cases),
            self.prompt_version,
        )

        results: List[CaseResult] = []
        for case in cases:
            results.append(self._run_case(case, repeats=repeats))

        # Fairness post-pass: compare scores
        results = self._apply_fairness_checks(cases, results)

        metrics = aggregate(results)
        gate = metrics.pass_rate >= self.eval_settings.pass_rate_threshold

        run = EvaluationRun(
            run_id=run_id,
            application_version=__version__,
            prompt_version=self.prompt_version,
            model_configuration={
                "llm_provider": self.app_settings.llm_provider,
                "use_llm_extraction": self.use_llm,
                "temperature": self.app_settings.llm_temperature,
                "model": self.app_settings.openai_model,
            },
            dataset_version=catalog.get("version", "2.0.0"),
            categories=sorted({c.get("category", "") for c in cases}),
            case_results=results,
            metrics=metrics,
            gate_passed=gate,
            recommendations=self._recommendations(metrics, results),
        )

        base_id = baseline_run_id or find_baseline_run_id(
            self.eval_settings.baseline_run_id
        )
        if base_id and base_id != run_id:
            try:
                baseline = load_run(base_id)
                run.regression = compare_runs(baseline, run)
                if run.regression.new_failures:
                    run.recommendations.append(
                        f"Investigate new regressions: {run.regression.new_failures}"
                    )
                if run.regression.score_degraded:
                    run.gate_passed = False
                    run.recommendations.append(
                        "Quality gate failed due to score/pass-rate degradation vs baseline."
                    )
            except FileNotFoundError:
                logger.warning("Baseline run %s not found; skipping regression", base_id)

        path = save_run(run)
        report_path = self.eval_settings.results_path / f"{run_id}.md"
        write_report(run, report_path)
        logger.info("Evaluation saved to %s and %s", path, report_path)
        return run

    def _run_case(self, case: Dict[str, Any], repeats: Optional[int] = None) -> CaseResult:
        expected = dict(case.get("expected") or {})
        expected.setdefault("category_hint", case.get("category"))
        test_id = case["id"]
        start = time.perf_counter()
        notes: List[str] = []

        try:
            jd_text = read_text(case["jd"])
            resume_text = read_text(case["resume"])
        except FileNotFoundError as exc:
            return CaseResult(
                test_id=test_id,
                category=case.get("category", "other"),
                name=case.get("name", test_id),
                passed=False,
                failure_types=["other"],
                failure_reason=str(exc),
                expected=expected,
                prompt_version=self.prompt_version,
            )

        if expected.get("expect_error"):
            try:
                self.service.screen_text_resumes(
                    jd_text, [(f"{test_id}.txt", resume_text)], use_llm=self.use_llm
                )
                # If it succeeds unexpectedly on empty, fail
                passed = False
                reason = "Expected processing error but screening succeeded."
                dims = []
                failure_types = ["robustness"]
                actual: Dict[str, Any] = {}
            except ValueError:
                passed = True
                reason = ""
                from evaluation.metrics.schemas import DimensionScore

                dims = [
                    DimensionScore(
                        name="schema",
                        score=1.0,
                        passed=True,
                        threshold=1.0,
                        details="Error handled safely.",
                    ),
                    DimensionScore(
                        name="correctness",
                        score=1.0,
                        passed=True,
                        threshold=self.eval_settings.correctness_threshold,
                        details="Empty/invalid input rejected as expected.",
                    ),
                    DimensionScore(
                        name="groundedness",
                        score=1.0,
                        passed=True,
                        threshold=self.eval_settings.groundedness_threshold,
                        details="No claims produced.",
                    ),
                ]
                failure_types = []
                actual = {"error": "ValueError"}
            latency = (time.perf_counter() - start) * 1000
            return CaseResult(
                test_id=test_id,
                category=case.get("category", "other"),
                name=case.get("name", test_id),
                passed=passed,
                dimensions=dims,
                failure_types=failure_types,
                failure_reason=reason,
                expected=expected,
                actual=actual,
                latency_ms=round(latency, 2),
                prompt_version=self.prompt_version,
                notes=notes,
            )

        # Consistency: multiple runs
        n = repeats or (
            self.eval_settings.consistency_repeats
            if expected.get("consistency_check")
            else 1
        )
        evaluations = []
        for _ in range(n):
            result = self.service.screen_text_resumes(
                jd_text, [(f"{test_id}.txt", resume_text)], use_llm=self.use_llm
            )
            evaluations.append(result.candidates[0])

        primary = evaluations[0]
        jd_struct = self.jd_processor.process(jd_text)

        dims = [
            evaluate_schema(primary),
            evaluate_correctness(primary, expected),
            evaluate_groundedness(primary, resume_text, expected),
            evaluate_relevance(primary, jd_text),
            evaluate_completeness(
                primary,
                expected,
                jd_struct.required_skills,
            ),
        ]

        if expected.get("consistency_check") and len(evaluations) > 1:
            from evaluation.metrics.schemas import DimensionScore

            scores = [e.overall_match for e in evaluations]
            recs = {e.recommendation for e in evaluations}
            spread = max(scores) - min(scores)
            ok = spread <= self.eval_settings.consistency_score_tolerance and len(recs) == 1
            dims.append(
                DimensionScore(
                    name="consistency",
                    score=1.0 if ok else max(0.0, 1.0 - spread / 50.0),
                    passed=ok,
                    threshold=1.0,
                    details=f"score_spread={spread:.2f} recommendations={sorted(recs)}",
                )
            )
            if not ok:
                expected["category_hint"] = "consistency"

        if expected.get("compare_resume"):
            other = read_text(expected["compare_resume"])
            other_res = self.service.screen_text_resumes(
                jd_text, [("other.txt", other)], use_llm=self.use_llm
            )
            delta = abs(primary.overall_match - other_res.candidates[0].overall_match)
            from evaluation.metrics.schemas import DimensionScore

            ok = delta <= self.eval_settings.consistency_score_tolerance
            dims.append(
                DimensionScore(
                    name="consistency",
                    score=1.0 if ok else 0.0,
                    passed=ok,
                    threshold=1.0,
                    details=f"similar_profile_delta={delta:.2f}",
                )
            )

        if self.eval_settings.use_llm_judge:
            judged = llm_judge_scores(
                self.service.llm,
                jd_text,
                resume_text,
                primary.model_dump(),
            )
            if judged:
                dims.extend(judge_to_dimensions(judged))
                notes.append("LLM-as-judge scores attached (advisory).")

        failure_types = classify_failures(dims, expected, primary)
        # Determine pass: core dimensions must pass
        core_names = {"schema", "correctness", "groundedness"}
        if expected.get("consistency_check"):
            core_names.add("consistency")
        core = [d for d in dims if d.name in core_names]
        passed = all(d.passed or d.uncertain for d in core) and all(
            d.passed for d in dims if d.name == "consistency"
        ) if any(d.name == "consistency" for d in dims) else all(
            d.passed or d.uncertain for d in core
        )
        # Also require consistency dim if present
        for d in dims:
            if d.name == "consistency" and not d.passed:
                passed = False

        actual = {
            "candidate": primary.candidate,
            "overall_match": primary.overall_match,
            "recommendation": primary.recommendation,
            "matched_required": primary.required_skills.matched,
            "missing_required": primary.required_skills.missing,
            "validation_warnings": primary.validation_warnings,
        }
        if expected.get("store_score_key"):
            self._score_cache[expected["store_score_key"]] = primary.overall_match
            self._score_cache[test_id] = primary.overall_match

        self._score_cache[test_id] = primary.overall_match
        latency = (time.perf_counter() - start) * 1000
        reason = ""
        if not passed:
            reason = "; ".join(
                d.details for d in dims if not d.passed and d.details
            ) or ",".join(failure_types)

        return CaseResult(
            test_id=test_id,
            category=case.get("category", "other"),
            name=case.get("name", test_id),
            passed=passed,
            dimensions=dims,
            failure_types=failure_types if not passed else [],
            failure_reason=reason,
            expected=expected,
            actual=actual,
            latency_ms=round(latency, 2),
            prompt_version=self.prompt_version,
            uncertain=any(d.uncertain for d in dims),
            notes=notes,
        )

    def _apply_fairness_checks(self, cases, results: List[CaseResult]) -> List[CaseResult]:
        by_id = {r.test_id: r for r in results}
        updated = []
        for case, result in zip(cases, results):
            expected = case.get("expected") or {}
            compare_to = expected.get("fairness_compare_to")
            if not compare_to:
                updated.append(result)
                continue
            base_score = self._score_cache.get(compare_to)
            cur_score = result.actual.get("overall_match")
            if base_score is None or cur_score is None:
                result.notes.append("Fairness base score unavailable; marked uncertain.")
                result.uncertain = True
                updated.append(result)
                continue
            tol = float(expected.get("fairness_score_tolerance", 5.0))
            delta = abs(float(cur_score) - float(base_score))
            from evaluation.metrics.schemas import DimensionScore

            ok = delta <= tol
            dim = DimensionScore(
                name="fairness",
                score=1.0 if ok else max(0.0, 1.0 - delta / 20.0),
                passed=ok,
                threshold=1.0,
                details=f"delta_vs_{compare_to}={delta:.2f} (tol={tol})",
            )
            result.dimensions.append(dim)
            if not ok:
                result.passed = False
                result.failure_types = list(
                    dict.fromkeys(result.failure_types + ["other"])
                )
                result.failure_reason = (
                    (result.failure_reason + "; " if result.failure_reason else "")
                    + dim.details
                )
            updated.append(result)
        return updated

    def _recommendations(self, metrics, results: List[CaseResult]) -> List[str]:
        recs = []
        if metrics.hallucination_failures:
            recs.append(
                "Review grounding filters / extraction prompts for hallucination failures."
            )
        if metrics.security_failures:
            recs.append(
                "Review adversarial handling; ensure resume text cannot override recommendations."
            )
        if metrics.schema_failures:
            recs.append("Inspect structured output validation path for schema failures.")
        if metrics.pass_rate < self.eval_settings.pass_rate_threshold:
            recs.append(
                f"Pass rate {metrics.pass_rate:.1%} below gate "
                f"{self.eval_settings.pass_rate_threshold:.0%}."
            )
        uncertain = sum(1 for r in results if r.uncertain)
        if uncertain:
            recs.append(
                f"{uncertain} case(s) marked uncertain — review ground-truth definitions."
            )
        return recs
