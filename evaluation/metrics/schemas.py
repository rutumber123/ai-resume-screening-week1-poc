"""Evaluation result schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    threshold: float
    details: str = ""
    uncertain: bool = False


class CaseResult(BaseModel):
    test_id: str
    category: str
    name: str
    passed: bool
    dimensions: List[DimensionScore] = Field(default_factory=list)
    failure_types: List[str] = Field(default_factory=list)
    failure_reason: str = ""
    expected: Dict[str, Any] = Field(default_factory=dict)
    actual: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_version: str = ""
    uncertain: bool = False
    notes: List[str] = Field(default_factory=list)


class AggregateMetrics(BaseModel):
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    average_evaluation_score: float = 0.0
    avg_correctness: float = 0.0
    avg_groundedness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_schema: float = 0.0
    hallucination_failures: int = 0
    schema_failures: int = 0
    grounding_failures: int = 0
    consistency_failures: int = 0
    robustness_failures: int = 0
    security_failures: int = 0
    correctness_failures: int = 0
    other_failures: int = 0
    failure_distribution: Dict[str, int] = Field(default_factory=dict)


class RegressionSummary(BaseModel):
    baseline_run_id: str = ""
    current_run_id: str = ""
    total_cases: int = 0
    new_failures: List[str] = Field(default_factory=list)
    resolved_failures: List[str] = Field(default_factory=list)
    unchanged_failures: List[str] = Field(default_factory=list)
    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    baseline_avg_score: float = 0.0
    current_avg_score: float = 0.0
    baseline_hallucination_failures: int = 0
    current_hallucination_failures: int = 0
    baseline_schema_failures: int = 0
    current_schema_failures: int = 0
    score_degraded: bool = False


class EvaluationRun(BaseModel):
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    application_version: str = "0.1.0"
    prompt_version: str = "v2"
    model_configuration: Dict[str, Any] = Field(default_factory=dict)
    dataset_version: str = "2.0.0"
    categories: List[str] = Field(default_factory=list)
    case_results: List[CaseResult] = Field(default_factory=list)
    metrics: AggregateMetrics = Field(default_factory=AggregateMetrics)
    regression: Optional[RegressionSummary] = None
    gate_passed: bool = False
    recommendations: List[str] = Field(default_factory=list)
