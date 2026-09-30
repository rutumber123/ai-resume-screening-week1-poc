"""
DeepEval integration with offline + optional local-Ollama paths.

Modes:
1) offline (default) — deterministic groundedness check, NO API key, NO LLM judge
2) ollama — optional FaithfulnessMetric via local Ollama (no commercial key)
3) openai — commercial API (requires OPENAI_API_KEY)

Usage:
  python -m evaluation.tools.deepeval_bridge
  python -m evaluation.tools.deepeval_bridge --mode offline
  python -m evaluation.tools.deepeval_bridge --mode ollama
  python -m evaluation.tools.deepeval_bridge --mode openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def deterministic_faithfulness(
    actual_output: str, context_chunks: List[str], threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Offline stand-in for DeepEval FaithfulnessMetric.

    Extracts capitalized / known tech tokens claimed in the output and checks
    each appears in the retrieval context. No LLM judge required.
    """
    context = " ".join(context_chunks).lower()
    # Claim-like tech tokens in the output
    candidates = set(
        re.findall(
            r"\b(Python|FastAPI|Django|Kubernetes|AWS|Azure|GCP|Docker|Excel|RAG|LangChain)\b",
            actual_output,
            flags=re.IGNORECASE,
        )
    )
    # Ignore tokens that are clearly marked missing
    missing_marked = set(
        re.findall(
            r"(?:missing|no|not)\s+([A-Za-z][A-Za-z0-9+.#]*)",
            actual_output,
            flags=re.IGNORECASE,
        )
    )
    missing_marked = {m.lower() for m in missing_marked}

    claimed = [c for c in candidates if c.lower() not in missing_marked]
    if not claimed:
        score = 1.0
        unsupported: List[str] = []
    else:
        unsupported = [c for c in claimed if c.lower() not in context]
        score = 1.0 - (len(unsupported) / len(claimed))

    return {
        "status": "passed" if score >= threshold else "failed",
        "mode": "offline_deterministic",
        "metric": "DeterministicFaithfulness",
        "score": round(score, 4),
        "threshold": threshold,
        "claimed": claimed,
        "unsupported": unsupported,
        "note": (
            "No API key used. This is a local groundedness proxy, not DeepEval GEval."
        ),
    }


def run_offline() -> Dict[str, Any]:
    # Positive case — claims supported by context
    ok = deterministic_faithfulness(
        actual_output=(
            "Candidate matched Python and FastAPI. Missing Kubernetes. "
            "Recommendation: Review."
        ),
        context_chunks=[
            "Skills: Python, FastAPI, Docker. 5 years of experience. No Kubernetes."
        ],
    )
    # Negative case — hallucinated Kubernetes as matched
    bad = deterministic_faithfulness(
        actual_output="Candidate has Kubernetes and AWS experience.",
        context_chunks=["Skills: Python, Docker. No cloud vendors named."],
    )
    return {
        "status": "passed" if ok["status"] == "passed" and bad["status"] == "failed" else "failed",
        "mode": "offline",
        "positive_case": ok,
        "negative_case": bad,
        "verification": (
            "Offline DeepEval bridge OK: supported claims pass, unsupported claims fail."
            if ok["status"] == "passed" and bad["status"] == "failed"
            else "Offline verification unexpected — inspect positive/negative cases."
        ),
    }


def run_deepeval_ollama() -> Dict[str, Any]:
    try:
        from deepeval import assert_test
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        from deepeval.models import OllamaModel
    except ImportError:
        return {
            "status": "skipped",
            "reason": "deepeval not installed. pip install deepeval",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed_or_unconfigured", "error": str(exc)}

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        model = OllamaModel(model=model_name)
        test_case = LLMTestCase(
            input="Screen resume against Python/FastAPI JD",
            actual_output=(
                "Candidate matched Python and FastAPI. Missing Kubernetes. "
                "Recommendation: Review."
            ),
            retrieval_context=[
                "Skills: Python, FastAPI, Docker. 5 years of experience. No Kubernetes."
            ],
        )
        metric = FaithfulnessMetric(threshold=0.7, model=model)
        assert_test(test_case, [metric])
        return {
            "status": "passed",
            "mode": "ollama",
            "metric": "FaithfulnessMetric",
            "score": getattr(metric, "score", None),
            "model": model_name,
            "note": "Used local Ollama — no commercial API key.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed_or_unconfigured",
            "mode": "ollama",
            "error": str(exc),
            "hint": (
                "Start Ollama locally (https://ollama.com), pull a model, e.g. "
                "`ollama pull llama3.2`, then retry."
            ),
        }


def run_deepeval_openai() -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY not set. Use --mode offline or --mode ollama.",
        }
    try:
        from deepeval import assert_test
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return {
            "status": "skipped",
            "reason": "deepeval not installed. pip install deepeval",
        }

    test_case = LLMTestCase(
        input="Screen resume against Python/FastAPI JD",
        actual_output=(
            "Candidate matched Python and FastAPI. Missing Kubernetes. "
            "Recommendation: Review."
        ),
        retrieval_context=[
            "Skills: Python, FastAPI, Docker. 5 years of experience. No Kubernetes."
        ],
    )
    metric = FaithfulnessMetric(threshold=0.7)
    try:
        assert_test(test_case, [metric])
        return {
            "status": "passed",
            "mode": "openai",
            "metric": "FaithfulnessMetric",
            "score": getattr(metric, "score", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed_or_unconfigured", "mode": "openai", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepEval bridge (offline/ollama/openai)")
    parser.add_argument(
        "--mode",
        choices=["offline", "ollama", "openai"],
        default=os.getenv("DEEPEVAL_MODE", "offline"),
    )
    args = parser.parse_args()

    if args.mode == "offline":
        result = run_offline()
    elif args.mode == "ollama":
        result = run_deepeval_ollama()
    else:
        result = run_deepeval_openai()

    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
