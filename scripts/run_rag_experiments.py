#!/usr/bin/env python
"""Chunking + Top-K experiments for Week 3 RAG."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.logging import setup_logging
from evaluation.rag.runner import RAGEvaluationRunner


def main() -> int:
    setup_logging()
    catalog = ROOT / "evaluation_data" / "rag" / "rag_catalog.json"
    if not catalog.exists():
        from scripts.generate_rag_catalog import main as gen

        gen()

    results = []
    for profile in ["A_small", "B_medium", "C_large"]:
        for k in [3, 5, 8]:
            runner = RAGEvaluationRunner(chunking_profile=profile, top_k=k)
            # Use a representative subset for speed
            run = runner.run(
                test_ids=[
                    "D01",
                    "D02",
                    "D04",
                    "D06",
                    "U01",
                    "X01",
                    "I01",
                    "M01",
                    "H01",
                    "C01",
                ]
            )
            row = {
                "chunking_profile": profile,
                "top_k": k,
                "pass_rate": run.metrics.pass_rate,
                "passed": run.metrics.passed_tests,
                "failed": run.metrics.failed_tests,
                "avg_score": run.metrics.average_evaluation_score,
                "run_id": run.run_id,
            }
            # Dimension avgs
            for name in ["context_precision", "context_recall", "faithfulness"]:
                vals = [
                    d.score
                    for c in run.case_results
                    for d in c.dimensions
                    if d.name == name
                ]
                row[f"avg_{name}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
            results.append(row)
            print(
                f"{profile:10s} k={k} pass={row['pass_rate']:.0%} "
                f"faith={row['avg_faithfulness']:.2f} recall={row['avg_context_recall']:.2f}"
            )

    out_dir = ROOT / "experiments" / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "chunking_topk_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Pick recommended config: best pass_rate then faithfulness then recall
    best = sorted(
        results,
        key=lambda r: (r["pass_rate"], r["avg_faithfulness"], r["avg_context_recall"]),
        reverse=True,
    )[0]
    summary = {
        "results": results,
        "recommended": best,
        "notes": [
            "Recommendation balances pass rate, faithfulness, and context recall.",
            "Higher K can improve recall but may reduce precision.",
            "Smaller chunks can help precise leave-policy facts; larger chunks help multi-sentence answers.",
        ],
    }
    (out_dir / "chunking_topk_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nRecommended:", json.dumps(best, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
