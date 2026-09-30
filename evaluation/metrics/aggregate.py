"""Aggregate metric calculation — definitions are explicit, not opaque."""

from __future__ import annotations

from collections import Counter
from typing import List

from evaluation.metrics.schemas import AggregateMetrics, CaseResult, DimensionScore


def case_average_score(dimensions: List[DimensionScore]) -> float:
    if not dimensions:
        return 0.0
    return sum(d.score for d in dimensions) / len(dimensions)


def aggregate(case_results: List[CaseResult]) -> AggregateMetrics:
    total = len(case_results)
    passed = sum(1 for c in case_results if c.passed)
    failed = total - passed
    pass_rate = (passed / total) if total else 0.0

    def avg_dim(name: str) -> float:
        vals = []
        for c in case_results:
            for d in c.dimensions:
                if d.name == name:
                    vals.append(d.score)
        return (sum(vals) / len(vals)) if vals else 0.0

    avg_eval = (
        sum(case_average_score(c.dimensions) for c in case_results) / total
        if total
        else 0.0
    )

    dist: Counter[str] = Counter()
    for c in case_results:
        if not c.passed:
            if c.failure_types:
                for ft in c.failure_types:
                    dist[ft] += 1
            else:
                dist["other"] += 1

    return AggregateMetrics(
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        pass_rate=round(pass_rate, 4),
        average_evaluation_score=round(avg_eval, 4),
        avg_correctness=round(avg_dim("correctness"), 4),
        avg_groundedness=round(avg_dim("groundedness"), 4),
        avg_relevance=round(avg_dim("relevance"), 4),
        avg_completeness=round(avg_dim("completeness"), 4),
        avg_schema=round(avg_dim("schema"), 4),
        hallucination_failures=dist.get("hallucination", 0),
        schema_failures=dist.get("schema_violation", 0),
        grounding_failures=dist.get("grounding", 0),
        consistency_failures=dist.get("inconsistent_response", 0),
        robustness_failures=dist.get("robustness", 0),
        security_failures=dist.get("prompt_injection", 0)
        + dist.get("instruction_override", 0),
        correctness_failures=dist.get("incorrect_matching", 0)
        + dist.get("incorrect_extraction", 0)
        + dist.get("missing_requirement", 0)
        + dist.get("unsupported_inference", 0),
        other_failures=dist.get("other", 0),
        failure_distribution=dict(dist),
    )
