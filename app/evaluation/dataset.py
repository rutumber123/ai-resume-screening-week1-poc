"""Evaluation helpers for baseline dataset execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_baseline() -> dict[str, Any]:
    path = ROOT / "evaluation_data" / "baseline_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def resolve_jd(dataset: dict[str, Any], jd_key: str) -> str:
    rel = dataset["job_descriptions"][jd_key]
    return read_text(rel)
