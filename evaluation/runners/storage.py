"""Persist and load evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from evaluation.configuration.settings import get_eval_settings
from evaluation.metrics.schemas import EvaluationRun


def save_run(run: EvaluationRun) -> Path:
    settings = get_eval_settings()
    path = settings.results_path / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    latest = settings.results_path / "latest.json"
    latest.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    # index
    index_path = settings.results_path / "index.json"
    index: List[dict] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index = [i for i in index if i.get("run_id") != run.run_id]
    index.append(
        {
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "prompt_version": run.prompt_version,
            "pass_rate": run.metrics.pass_rate,
            "gate_passed": run.gate_passed,
            "path": path.name,
        }
    )
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def load_run(run_id: str) -> EvaluationRun:
    settings = get_eval_settings()
    path = settings.results_path / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {path}")
    return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


def load_latest() -> Optional[EvaluationRun]:
    settings = get_eval_settings()
    path = settings.results_path / "latest.json"
    if not path.exists():
        return None
    return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


def find_baseline_run_id(explicit: str = "") -> Optional[str]:
    if explicit:
        return explicit
    settings = get_eval_settings()
    if settings.baseline_run_id:
        return settings.baseline_run_id
    ptr = settings.results_path / "BASELINE_RUN_ID.txt"
    if ptr.exists():
        value = ptr.read_text(encoding="utf-8").strip()
        if value:
            return value
    index_path = settings.results_path / "index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if len(index) < 2:
        return None
    return index[-2]["run_id"]
