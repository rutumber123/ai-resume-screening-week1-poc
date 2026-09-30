"""
Observability foundation for evaluation traces (LangSmith-ready hooks).

Week 2 captures local JSON traces. Week 3 can forward these to LangSmith.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

from evaluation.configuration.settings import get_eval_settings


@contextmanager
def evaluation_span(name: str, metadata: Dict[str, Any] | None = None) -> Iterator[Dict[str, Any]]:
    start = time.perf_counter()
    span: Dict[str, Any] = {
        "name": name,
        "metadata": metadata or {},
        "events": [],
    }
    try:
        yield span
        span["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        _persist_span(span)


def _persist_span(span: Dict[str, Any]) -> None:
    settings = get_eval_settings()
    trace_dir = settings.results_path / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    # Avoid storing full resume text
    safe = dict(span)
    meta = dict(safe.get("metadata") or {})
    for key in ("resume_text", "jd_text", "raw_response"):
        if key in meta and isinstance(meta[key], str) and len(meta[key]) > 200:
            meta[key] = meta[key][:200] + "…[truncated]"
    safe["metadata"] = meta
    path = trace_dir / f"trace_{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
