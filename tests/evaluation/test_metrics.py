"""Unit tests for Week 2 evaluation metrics/evaluators."""

from app.models.schemas import (
    CandidateEvaluation,
    ComponentScores,
    ExperienceMatchResult,
    SkillMatchResult,
)
from evaluation.evaluators.dimensions import (
    evaluate_correctness,
    evaluate_groundedness,
    evaluate_schema,
)
from evaluation.metrics.aggregate import aggregate
from evaluation.metrics.schemas import CaseResult, DimensionScore


def _eval(**kwargs) -> CandidateEvaluation:
    base = dict(
        candidate="X",
        overall_match=80,
        recommendation="Shortlist",
        experience=ExperienceMatchResult(
            required_years=5, candidate_years=6, score=1.0, note="ok"
        ),
        required_skills=SkillMatchResult(
            matched=["Python", "FastAPI"], missing=["RAG"], score=0.66
        ),
        preferred_skills=SkillMatchResult(matched=[], missing=[], score=1.0),
        relevant_experience="Built FastAPI",
        potential_gaps=["Missing RAG"],
        evidence=["Skills Python FastAPI"],
        explanation="Strong Python FastAPI match",
        component_scores=ComponentScores(
            required_skills=0.66,
            preferred_skills=1,
            experience=1,
            responsibilities=0.5,
            education=0.7,
            weights={},
        ),
    )
    base.update(kwargs)
    return CandidateEvaluation(**base)


def test_schema_ok() -> None:
    d = evaluate_schema(_eval())
    assert d.passed and d.score == 1.0


def test_correctness_detects_bad_match() -> None:
    ev = _eval(
        required_skills=SkillMatchResult(
            matched=["Python", "Django"], missing=[], score=1.0
        )
    )
    d = evaluate_correctness(ev, {"must_not_match_skills": ["Django"]})
    assert not d.passed


def test_groundedness_flags_ungrounded_skill() -> None:
    ev = _eval(
        required_skills=SkillMatchResult(
            matched=["Kubernetes"], missing=[], score=1.0
        ),
        evidence=[],
    )
    d = evaluate_groundedness(ev, "Skills: Python only", {"strict_no_hallucination": True})
    assert not d.passed


def test_aggregate_pass_rate() -> None:
    results = [
        CaseResult(
            test_id="A",
            category="positive",
            name="a",
            passed=True,
            dimensions=[
                DimensionScore(name="correctness", score=1, passed=True, threshold=0.7)
            ],
        ),
        CaseResult(
            test_id="B",
            category="negative",
            name="b",
            passed=False,
            failure_types=["hallucination"],
            dimensions=[
                DimensionScore(name="correctness", score=0.2, passed=False, threshold=0.7)
            ],
        ),
    ]
    m = aggregate(results)
    assert m.total_tests == 2
    assert m.pass_rate == 0.5
    assert m.hallucination_failures == 1
