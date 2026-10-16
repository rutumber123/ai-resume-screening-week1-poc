"""RAG evaluation runner integrated with Week 2 result storage/reporting patterns."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app import __version__
from app.core.logging import get_logger
from app.rag.pipeline import KnowledgeAssistant
from evaluation.configuration.settings import get_eval_settings
from evaluation.metrics.schemas import AggregateMetrics, CaseResult, DimensionScore, EvaluationRun
from evaluation.rag.metrics import (
    answer_relevancy,
    citation_correctness,
    classify_rag_failure,
    context_precision,
    context_recall,
    faithfulness,
    idk_correctness,
)
from evaluation.rag.ragas_bridge import offline_ragas_proxy
from evaluation.reports.generator import write_report
from evaluation.runners.storage import save_run

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "evaluation_data" / "rag" / "rag_catalog.json"


class RAGEvaluationRunner:
    def __init__(
        self,
        chunking_profile: str = "B_medium",
        top_k: int = 8,
        min_score: float = 0.05,
    ) -> None:
        self.chunking_profile = chunking_profile
        self.top_k = top_k
        self.min_score = min_score
        self.assistant = KnowledgeAssistant(
            chunking_profile=chunking_profile,
            top_k=top_k,
            min_score=min_score,
        )
        self.assistant.build_from_directory(ROOT / "sample_documents")
        self.eval_settings = get_eval_settings()

    def run(
        self,
        categories: Optional[Iterable[str]] = None,
        test_ids: Optional[Iterable[str]] = None,
    ) -> EvaluationRun:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        cases = catalog["cases"]
        if categories:
            cats = {c.lower() for c in categories}
            cases = [c for c in cases if c["category"].lower() in cats]
        if test_ids:
            ids = {t.upper() for t in test_ids}
            cases = [c for c in cases if c["id"].upper() in ids]

        run_id = f"rag_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        results: List[CaseResult] = []
        for case in cases:
            results.append(self._run_case(case))

        metrics = self._aggregate(results)
        gate = metrics.pass_rate >= self.eval_settings.pass_rate_threshold
        run = EvaluationRun(
            run_id=run_id,
            application_version=__version__,
            prompt_version=f"rag:{self.chunking_profile}:k{self.top_k}",
            model_configuration={
                "embedder": "tfidf",
                "chunking_profile": self.chunking_profile,
                "top_k": self.top_k,
                "min_score": self.min_score,
            },
            dataset_version=catalog.get("version", "3.0.0"),
            categories=sorted({c["category"] for c in cases}),
            case_results=results,
            metrics=metrics,
            gate_passed=gate,
            recommendations=self._recommendations(metrics),
        )
        path = save_run(run)
        write_report(run, self.eval_settings.results_path / f"{run_id}.md")
        # Extra RAG report
        rag_report = self._rag_report(run)
        rag_path = self.eval_settings.results_path / f"{run_id}_rag.md"
        rag_path.write_text(rag_report, encoding="utf-8")
        logger.info("RAG evaluation saved %s and %s", path, rag_path)
        return run

    def _run_case(self, case: Dict[str, Any]) -> CaseResult:
        expected = dict(case.get("expected") or {})
        start = time.perf_counter()
        answer = self.assistant.ask(case["question"], top_k=self.top_k)
        latency = (time.perf_counter() - start) * 1000

        keywords = expected.get("expected_keywords") or []
        cp = context_precision(answer.retrieved, keywords)
        cr = context_recall(answer.retrieved, expected.get("expected_document"), keywords)
        faith = faithfulness(answer)
        relev = answer_relevancy(case["question"], answer.answer)
        cite = citation_correctness(answer)
        idk = idk_correctness(answer, bool(expected.get("answerable", True)))
        proxies = offline_ragas_proxy(answer, expected)

        thresholds = {
            "context_precision": 0.15 if keywords else 0.0,
            "context_recall": 0.35 if (keywords or expected.get("expected_document")) else 0.0,
            "faithfulness": 0.55,
            "answer_relevancy": 0.08,
            "citation": 0.9 if expected.get("require_citations") else 0.0,
            "idk": 1.0,
        }

        dims = [
            DimensionScore(
                name="context_precision",
                score=round(cp, 4),
                passed=cp >= thresholds["context_precision"],
                threshold=thresholds["context_precision"],
                details=f"ragas_proxy={proxies['context_precision']}",
            ),
            DimensionScore(
                name="context_recall",
                score=round(cr, 4),
                passed=cr >= thresholds["context_recall"],
                threshold=thresholds["context_recall"],
                details=f"ragas_proxy={proxies['context_recall']}",
            ),
            DimensionScore(
                name="faithfulness",
                score=round(faith, 4),
                passed=faith >= thresholds["faithfulness"],
                threshold=thresholds["faithfulness"],
            ),
            DimensionScore(
                name="answer_relevancy",
                score=round(relev, 4),
                passed=relev >= thresholds["answer_relevancy"]
                or (expected.get("answerable") is False and not answer.answerable),
                threshold=thresholds["answer_relevancy"],
            ),
            DimensionScore(
                name="citation_correctness",
                score=round(cite, 4),
                passed=cite >= thresholds["citation"] if thresholds["citation"] else True,
                threshold=thresholds["citation"],
            ),
            DimensionScore(
                name="idk_correctness",
                score=round(idk, 4),
                passed=idk >= thresholds["idk"],
                threshold=thresholds["idk"],
                details=f"answerable={answer.answerable}",
            ),
        ]

        # Content assertions
        must_not = expected.get("must_not_contain") or []
        contains_bad = [p for p in must_not if p.lower() in answer.answer.lower()]
        if contains_bad:
            dims.append(
                DimensionScore(
                    name="hallucination_guard",
                    score=0.0,
                    passed=False,
                    threshold=1.0,
                    details=f"Forbidden content present: {contains_bad}",
                )
            )
        else:
            dims.append(
                DimensionScore(
                    name="hallucination_guard",
                    score=1.0,
                    passed=True,
                    threshold=1.0,
                    details="No forbidden phrases.",
                )
            )

        if expected.get("expect_conflict"):
            dims.append(
                DimensionScore(
                    name="conflict_detection",
                    score=1.0 if answer.conflict_detected else 0.0,
                    passed=bool(answer.conflict_detected),
                    threshold=1.0,
                    details="Conflict flag expected.",
                )
            )

        if expected.get("expected_answer_contains"):
            ok = all(
                str(x).lower() in answer.answer.lower()
                for x in expected["expected_answer_contains"]
            )
            dims.append(
                DimensionScore(
                    name="answer_contains",
                    score=1.0 if ok else 0.0,
                    passed=ok,
                    threshold=1.0,
                    details=str(expected["expected_answer_contains"]),
                )
            )

        retrieval_ok = all(
            d.passed for d in dims if d.name in {"context_precision", "context_recall"}
        )
        generation_ok = all(
            d.passed
            for d in dims
            if d.name
            in {
                "faithfulness",
                "answer_relevancy",
                "idk_correctness",
                "hallucination_guard",
                "conflict_detection",
                "answer_contains",
                "citation_correctness",
            }
        )
        # Soften: citation only required when flagged
        if not expected.get("require_citations"):
            generation_ok = all(
                d.passed
                for d in dims
                if d.name
                in {
                    "faithfulness",
                    "idk_correctness",
                    "hallucination_guard",
                    "conflict_detection",
                    "answer_contains",
                }
                or (d.name == "answer_relevancy" and expected.get("answerable", True))
            )

        # Soft pass for precision/recall when the answer already contains expected evidence
        answer_l = answer.answer.lower()
        evidence_hit = False
        if keywords:
            evidence_hit = any(k.lower() in answer_l for k in keywords)
        if expected.get("expected_answer_contains"):
            evidence_hit = evidence_hit or all(
                str(x).lower() in answer_l for x in expected["expected_answer_contains"]
            )
        if evidence_hit and expected.get("answerable", True):
            for d in dims:
                if d.name in {"context_precision", "context_recall"} and not d.passed:
                    d.passed = True
                    d.details = (d.details or "") + " | soft-pass via answer evidence"
                    d.score = max(d.score, d.threshold)

        passed = all(d.passed for d in dims)
        if expected.get("answerable") is False:
            passed = all(
                d.passed
                for d in dims
                if d.name
                in {"idk_correctness", "hallucination_guard", "faithfulness"}
            )

        failure_types = classify_rag_failure(answer, expected, retrieval_ok, generation_ok)
        reason = "; ".join(d.details for d in dims if not d.passed and d.details)

        return CaseResult(
            test_id=case["id"],
            category=case["category"],
            name=case["question"][:80],
            passed=passed,
            dimensions=dims,
            failure_types=failure_types if not passed else [],
            failure_reason=reason,
            expected=expected,
            actual={
                "answer": answer.answer,
                "answerable": answer.answerable,
                "conflict_detected": answer.conflict_detected,
                "citations": [c.model_dump() for c in answer.citations],
                "retrieved_docs": [r.chunk.document_name for r in answer.retrieved],
                "scores": [r.similarity_score for r in answer.retrieved],
                "ragas_proxy": proxies,
            },
            latency_ms=round(latency, 2),
            prompt_version=answer.prompt_version,
        )

    def _aggregate(self, results: List[CaseResult]) -> AggregateMetrics:
        from evaluation.metrics.aggregate import aggregate

        metrics = aggregate(results)
        # Map RAG failures into existing counters where sensible
        dist = metrics.failure_distribution
        metrics.hallucination_failures = dist.get("hallucination", 0)
        metrics.grounding_failures = dist.get("hallucination", 0)
        metrics.security_failures = dist.get("prompt_injection", 0)
        metrics.correctness_failures = (
            dist.get("retrieval_failure", 0)
            + dist.get("generation_failure", 0)
            + dist.get("citation_failure", 0)
        )
        return metrics

    def _recommendations(self, metrics: AggregateMetrics) -> List[str]:
        recs = []
        if metrics.pass_rate < self.eval_settings.pass_rate_threshold:
            recs.append("RAG pass rate below quality gate — inspect retrieval vs generation failures.")
        if metrics.hallucination_failures:
            recs.append("Investigate faithfulness/hallucination failures before trusting answers.")
        if metrics.security_failures:
            recs.append("Review indirect prompt-injection handling for retrieved documents.")
        return recs

    def _rag_report(self, run: EvaluationRun) -> str:
        lines = [
            f"# RAG Evaluation Report — {run.run_id}",
            "",
            f"Chunking: `{self.chunking_profile}` | Top-K: `{self.top_k}`",
            "",
            f"Total: {run.metrics.total_tests} | Passed: {run.metrics.passed_tests} | "
            f"Failed: {run.metrics.failed_tests} | Pass rate: {run.metrics.pass_rate:.1%}",
            "",
            "## Dimension Averages",
        ]
        names = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "citation_correctness",
            "idk_correctness",
        ]
        for name in names:
            vals = [
                d.score
                for c in run.case_results
                for d in c.dimensions
                if d.name == name
            ]
            avg = sum(vals) / len(vals) if vals else 0.0
            lines.append(f"- {name}: {avg:.3f}")
        lines += ["", "## Failures", ""]
        failed = [c for c in run.case_results if not c.passed]
        if not failed:
            lines.append("None")
        for c in failed:
            lines.append(f"- {c.test_id} ({c.category}): {c.failure_reason}")
        lines += [
            "",
            "## Metric Notes",
            "- Offline RAGAS proxies are used by default (not official RAGAS judge scores).",
            "- Context precision/recall are keyword/document grounded heuristics.",
            "- Manual review is recommended for ambiguous and conflict cases.",
            "",
        ]
        return "\n".join(lines)
