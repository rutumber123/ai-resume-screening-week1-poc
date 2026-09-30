"""Regression comparison between evaluation runs."""

from __future__ import annotations

from evaluation.metrics.schemas import EvaluationRun, RegressionSummary


def compare_runs(baseline: EvaluationRun, current: EvaluationRun) -> RegressionSummary:
    base_fail = {c.test_id for c in baseline.case_results if not c.passed}
    cur_fail = {c.test_id for c in current.case_results if not c.passed}
    new_failures = sorted(cur_fail - base_fail)
    resolved = sorted(base_fail - cur_fail)
    unchanged = sorted(base_fail & cur_fail)

    score_degraded = (
        current.metrics.average_evaluation_score + 0.02
        < baseline.metrics.average_evaluation_score
        or current.metrics.pass_rate + 0.02 < baseline.metrics.pass_rate
    )

    return RegressionSummary(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        total_cases=current.metrics.total_tests,
        new_failures=new_failures,
        resolved_failures=resolved,
        unchanged_failures=unchanged,
        baseline_pass_rate=baseline.metrics.pass_rate,
        current_pass_rate=current.metrics.pass_rate,
        baseline_avg_score=baseline.metrics.average_evaluation_score,
        current_avg_score=current.metrics.average_evaluation_score,
        baseline_hallucination_failures=baseline.metrics.hallucination_failures,
        current_hallucination_failures=current.metrics.hallucination_failures,
        baseline_schema_failures=baseline.metrics.schema_failures,
        current_schema_failures=current.metrics.schema_failures,
        score_degraded=score_degraded,
    )
