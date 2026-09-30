"""Markdown / text evaluation reports."""

from __future__ import annotations

from pathlib import Path

from evaluation.metrics.schemas import EvaluationRun


def render_report(run: EvaluationRun) -> str:
    m = run.metrics
    lines = [
        f"# Evaluation Report — {run.run_id}",
        "",
        "## Summary",
        "",
        f"- **Timestamp:** {run.timestamp}",
        f"- **Prompt version:** {run.prompt_version}",
        f"- **Dataset version:** {run.dataset_version}",
        f"- **Model config:** `{run.model_configuration}`",
        f"- **Total tests:** {m.total_tests}",
        f"- **Passed:** {m.passed_tests}",
        f"- **Failed:** {m.failed_tests}",
        f"- **Pass rate:** {m.pass_rate:.1%}",
        f"- **Average evaluation score:** {m.average_evaluation_score:.3f}",
        f"- **Quality gate:** {'PASS' if run.gate_passed else 'FAIL'}",
        "",
        "## Metric Summary",
        "",
        f"| Dimension | Average |",
        f"|-----------|---------|",
        f"| Correctness | {m.avg_correctness:.3f} |",
        f"| Groundedness | {m.avg_groundedness:.3f} |",
        f"| Relevance | {m.avg_relevance:.3f} |",
        f"| Completeness | {m.avg_completeness:.3f} |",
        f"| Schema | {m.avg_schema:.3f} |",
        "",
        "## Failure Counts",
        "",
        f"- Hallucination: {m.hallucination_failures}",
        f"- Grounding: {m.grounding_failures}",
        f"- Schema: {m.schema_failures}",
        f"- Consistency: {m.consistency_failures}",
        f"- Robustness: {m.robustness_failures}",
        f"- Security/Adversarial: {m.security_failures}",
        f"- Correctness-related: {m.correctness_failures}",
        "",
        "### Failure Distribution",
        "",
    ]
    if m.failure_distribution:
        for k, v in sorted(m.failure_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")

    failed = [c for c in run.case_results if not c.passed]
    lines += ["", "## Failed Tests", ""]
    if not failed:
        lines.append("No failed tests.")
    else:
        for c in failed[:25]:
            lines.append(
                f"- **{c.test_id}** ({c.category}) — {c.failure_reason or ', '.join(c.failure_types)}"
            )
        if len(failed) > 25:
            lines.append(f"- … and {len(failed) - 25} more")

    if run.regression:
        r = run.regression
        lines += [
            "",
            "## Regression Summary",
            "",
            f"- Baseline: `{r.baseline_run_id}`",
            f"- Current: `{r.current_run_id}`",
            f"- New failures: {len(r.new_failures)} → {r.new_failures}",
            f"- Resolved failures: {len(r.resolved_failures)} → {r.resolved_failures}",
            f"- Unchanged failures: {len(r.unchanged_failures)}",
            f"- Pass rate: {r.baseline_pass_rate:.1%} → {r.current_pass_rate:.1%}",
            f"- Avg score: {r.baseline_avg_score:.3f} → {r.current_avg_score:.3f}",
            f"- Hallucination failures: {r.baseline_hallucination_failures} → {r.current_hallucination_failures}",
            f"- Schema failures: {r.baseline_schema_failures} → {r.current_schema_failures}",
            f"- Score degraded: {r.score_degraded}",
        ]

    lines += ["", "## Recommendations for Investigation", ""]
    if run.recommendations:
        for rec in run.recommendations:
            lines.append(f"- {rec}")
    else:
        lines.append("- No automated recommendations.")

    lines += [
        "",
        "## Notes",
        "",
        "- Dimension scores are defined in `docs/WEEK2_EVALUATION.md`.",
        "- LLM-as-judge scores (if present) are advisory and uncertain.",
        "- Do not treat pass rate as proof of real-world hiring fairness.",
        "",
    ]
    return "\n".join(lines)


def write_report(run: EvaluationRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run), encoding="utf-8")
    return path
