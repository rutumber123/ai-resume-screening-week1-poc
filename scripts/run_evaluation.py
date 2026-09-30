#!/usr/bin/env python
"""
Single-command Week 2 evaluation entrypoint.

Examples:
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --categories positive,hallucination
  python scripts/run_evaluation.py --prompt-version v1 --compare-prompts v2,v3
  python scripts/run_evaluation.py --prompt-version v_regress_bad --baseline-run-id <id>

Exit codes:
  0 = quality gate passed
  1 = quality gate failed or errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.core.logging import setup_logging
from evaluation.runners.runner import EvaluationRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM evaluation suite")
    p.add_argument("--categories", type=str, default="", help="Comma-separated categories")
    p.add_argument("--test-ids", type=str, default="", help="Comma-separated test IDs")
    p.add_argument("--prompt-version", type=str, default="v2")
    p.add_argument("--baseline-run-id", type=str, default="")
    p.add_argument("--use-llm", action="store_true", help="Use live LLM extraction")
    p.add_argument(
        "--compare-prompts",
        type=str,
        default="",
        help="Comma-separated extra prompt versions to compare after primary run",
    )
    p.add_argument("--set-baseline", action="store_true", help="Mark this run as baseline file")
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    test_ids = [c.strip() for c in args.test_ids.split(",") if c.strip()] or None

    app_settings = Settings(llm_provider="mock") if not args.use_llm else Settings()
    runner = EvaluationRunner(
        prompt_version=args.prompt_version,
        app_settings=app_settings,
        use_llm=args.use_llm,
    )
    run = runner.run(
        categories=categories,
        test_ids=test_ids,
        baseline_run_id=args.baseline_run_id or None,
    )

    print("\n=== Evaluation Summary ===")
    print(f"Run ID:     {run.run_id}")
    print(f"Prompt:     {run.prompt_version}")
    print(f"Total:      {run.metrics.total_tests}")
    print(f"Passed:     {run.metrics.passed_tests}")
    print(f"Failed:     {run.metrics.failed_tests}")
    print(f"Pass rate:  {run.metrics.pass_rate:.1%}")
    print(f"Avg score:  {run.metrics.average_evaluation_score:.3f}")
    print(f"Gate:       {'PASS' if run.gate_passed else 'FAIL'}")
    if run.regression:
        print(
            f"Regression: new={len(run.regression.new_failures)} "
            f"resolved={len(run.regression.resolved_failures)} "
            f"degraded={run.regression.score_degraded}"
        )
    print(f"Report:     results/{run.run_id}.md")

    if args.set_baseline:
        baseline_ptr = ROOT / "results" / "BASELINE_RUN_ID.txt"
        baseline_ptr.parent.mkdir(parents=True, exist_ok=True)
        baseline_ptr.write_text(run.run_id, encoding="utf-8")
        print(f"Baseline pointer set to {run.run_id}")

    # Optional multi-prompt comparison table
    if args.compare_prompts:
        rows = [
            {
                "prompt": run.prompt_version,
                "pass_rate": run.metrics.pass_rate,
                "avg": run.metrics.average_evaluation_score,
                "hallucination_failures": run.metrics.hallucination_failures,
                "schema_failures": run.metrics.schema_failures,
                "run_id": run.run_id,
            }
        ]
        for ver in [v.strip() for v in args.compare_prompts.split(",") if v.strip()]:
            r2 = EvaluationRunner(
                prompt_version=ver,
                app_settings=app_settings,
                use_llm=args.use_llm,
            ).run(categories=categories, test_ids=test_ids, baseline_run_id=run.run_id)
            rows.append(
                {
                    "prompt": ver,
                    "pass_rate": r2.metrics.pass_rate,
                    "avg": r2.metrics.average_evaluation_score,
                    "hallucination_failures": r2.metrics.hallucination_failures,
                    "schema_failures": r2.metrics.schema_failures,
                    "run_id": r2.run_id,
                }
            )
        print("\n=== Prompt Comparison ===")
        print(json.dumps(rows, indent=2))
        (ROOT / "results" / "prompt_comparison_latest.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    return 0 if run.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
