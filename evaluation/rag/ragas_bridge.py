"""Optional RAGAS bridge — offline proxies + live RAGAS when installed/configured."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from app.rag.models import RAGAnswer
from evaluation.rag.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def offline_ragas_proxy(answer: RAGAnswer, expected: Dict[str, Any]) -> Dict[str, float]:
    """
    Local proxies for RAGAS-like metrics (no API key).
    These are not official RAGAS scores.
    """
    keywords = (
        expected.get("expected_keywords")
        or expected.get("expected_information_keywords")
        or []
    )
    if isinstance(keywords, str):
        keywords = [keywords]
    return {
        "context_precision": round(context_precision(answer.retrieved, keywords), 4),
        "context_recall": round(
            context_recall(
                answer.retrieved,
                expected.get("expected_document"),
                keywords,
            ),
            4,
        ),
        "faithfulness": round(faithfulness(answer), 4),
        "answer_relevancy": round(answer_relevancy(answer.question, answer.answer), 4),
    }


def try_ragas_evaluate(
    question: str, answer: str, contexts: List[str], ground_truth: str = ""
) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy as ragas_ar
        from ragas.metrics import context_precision as ragas_cp
        from ragas.metrics import context_recall as ragas_cr
        from ragas.metrics import faithfulness as ragas_faith
    except ImportError:
        return {
            "status": "skipped",
            "reason": "ragas/datasets not installed. Using offline proxies.",
        }

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "status": "skipped",
            "reason": "No OPENAI_API_KEY; RAGAS LLM judge unavailable. Using offline proxies.",
        }

    try:
        data = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth or answer],
            }
        )
        result = evaluate(
            data,
            metrics=[ragas_faith, ragas_ar, ragas_cp, ragas_cr],
        )
        return {"status": "passed", "scores": dict(result)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed_or_unconfigured", "error": str(exc)}


if __name__ == "__main__":
    print(json.dumps({"hint": "Use scripts/run_rag_evaluation.py"}, indent=2))
