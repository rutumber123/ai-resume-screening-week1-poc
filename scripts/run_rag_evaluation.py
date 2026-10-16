#!/usr/bin/env python
"""Run Week 3 RAG evaluation suite (CI-ready exit codes)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.logging import setup_logging
from evaluation.rag.runner import RAGEvaluationRunner


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Run RAG evaluation")
    p.add_argument("--chunking-profile", default="B_medium", choices=["A_small", "B_medium", "C_large"])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--categories", default="")
    p.add_argument("--test-ids", default="")
    args = p.parse_args()

    # Ensure catalog exists
    catalog = ROOT / "evaluation_data" / "rag" / "rag_catalog.json"
    if not catalog.exists():
        from scripts.generate_rag_catalog import main as gen

        gen()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    ids = [c.strip() for c in args.test_ids.split(",") if c.strip()] or None
    runner = RAGEvaluationRunner(chunking_profile=args.chunking_profile, top_k=args.top_k)
    run = runner.run(categories=cats, test_ids=ids)
    print("\n=== RAG Evaluation Summary ===")
    print(f"Run ID:    {run.run_id}")
    print(f"Profile:   {args.chunking_profile} | top_k={args.top_k}")
    print(f"Total:     {run.metrics.total_tests}")
    print(f"Passed:    {run.metrics.passed_tests}")
    print(f"Failed:    {run.metrics.failed_tests}")
    print(f"Pass rate: {run.metrics.pass_rate:.1%}")
    print(f"Gate:      {'PASS' if run.gate_passed else 'FAIL'}")
    print(f"Report:    results/{run.run_id}_rag.md")
    return 0 if run.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
