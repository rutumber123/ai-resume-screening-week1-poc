#!/usr/bin/env python
"""Run a quick demo screening against sample data (mock LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.core.logging import setup_logging
from app.services.screening import ScreeningService


def main() -> None:
    setup_logging()
    jd = (ROOT / "sample_data/job_descriptions/senior_python_genai.txt").read_text(
        encoding="utf-8"
    )
    resumes_dir = ROOT / "sample_data" / "resumes"
    selected = [
        "01_excellent_match_alex_rivera.txt",
        "02_poor_match_jordan_lee.txt",
        "03_partial_match_sam_patel.txt",
        "18_adversarial_prompt_injection_harper_quinn.txt",
    ]
    payloads = []
    for name in selected:
        payloads.append((name, (resumes_dir / name).read_bytes()))

    service = ScreeningService(Settings(llm_provider="mock"))
    result = service.screen(jd, payloads, use_llm=False)
    print(f"Job: {result.job_title}")
    print(f"Candidates: {len(result.candidates)}")
    print("\nComparison:")
    for row in result.comparison:
        print(
            f"  {row.candidate:20s}  match={row.match:5.1f}  "
            f"{row.recommendation:28s}  missing={', '.join(row.missing_skills[:3])}"
        )
    out = ROOT / "scripts" / "demo_output.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull JSON written to {out}")


if __name__ == "__main__":
    main()
