#!/usr/bin/env python
"""Execute baseline evaluation scenarios and print a summary table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.evaluation.dataset import load_baseline, read_text, resolve_jd
from app.services.screening import ScreeningService


def main() -> None:
    dataset = load_baseline()
    service = ScreeningService(Settings(llm_provider="mock"))
    print(f"Dataset: {dataset['dataset_name']} v{dataset['version']}")
    print(f"Scenarios: {len(dataset['scenarios'])}\n")

    for scenario in dataset["scenarios"]:
        jd = resolve_jd(dataset, scenario["jd"])
        if "resumes" in scenario:
            resumes = [(Path(p).name, read_text(p)) for p in scenario["resumes"]]
            result = service.screen_text_resumes(jd, resumes, use_llm=False)
            print(
                f"{scenario['id']:4s} {scenario['name'][:42]:42s} "
                f"candidates={len(result.candidates)}"
            )
            continue
        try:
            result = service.screen_text_resumes(
                jd,
                [(Path(scenario["resume"]).name, read_text(scenario["resume"]))],
                use_llm=False,
            )
            ev = result.candidates[0]
            print(
                f"{scenario['id']:4s} {scenario['name'][:42]:42s} "
                f"match={ev.overall_match:5.1f}  {ev.recommendation}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{scenario['id']:4s} {scenario['name'][:42]:42s} ERROR: {exc}")


if __name__ == "__main__":
    main()
