"""Unit tests for matching and scoring."""

from app.core.config import Settings
from app.core.constants import RECOMMENDATION_DOES_NOT_MEET, RECOMMENDATION_SHORTLIST
from app.models.schemas import CandidateProfile, JobDescription
from app.services.evaluator import CandidateEvaluator
from app.services.matching import match_experience, match_skills, skill_matches


def test_skill_alias_match() -> None:
    assert skill_matches("AWS", ["Amazon Web Services"])
    assert skill_matches("k8s", ["Kubernetes"])
    assert not skill_matches("Django", ["Python"])  # no unsupported inference


def test_match_skills_scores() -> None:
    result = match_skills(["Python", "FastAPI", "RAG"], ["Python", "FastAPI"])
    assert result.matched == ["Python", "FastAPI"]
    assert result.missing == ["RAG"]
    assert abs(result.score - round(2 / 3, 4)) < 1e-9


def test_experience_below_and_above() -> None:
    low = match_experience(5, 2)
    assert low.score < 1
    high = match_experience(5, 10)
    assert high.score == 1.0
    missing = match_experience(5, None)
    assert missing.score == 0.0


def test_evaluator_excellent_vs_poor() -> None:
    settings = Settings(LLM_PROVIDER="mock")
    evaluator = CandidateEvaluator(settings)
    jd = JobDescription(
        title="Engineer",
        required_skills=["Python", "FastAPI", "Docker"],
        preferred_skills=["Kubernetes"],
        min_experience_years=5,
        responsibilities=["Build FastAPI services with Docker"],
        education_requirements=["Bachelor"],
        raw_text="Python FastAPI Docker Kubernetes Bachelor 5 years",
    )
    good = CandidateProfile(
        name="Good Cand",
        total_experience_years=7,
        skills=["Python", "FastAPI", "Docker", "Kubernetes"],
        education=["Bachelor of Science"],
        previous_roles=["Built FastAPI services with Docker"],
        raw_text="Good Cand\n7 years of experience\nPython FastAPI Docker Kubernetes\nBuilt FastAPI services with Docker\nBachelor of Science",
    )
    bad = CandidateProfile(
        name="Bad Cand",
        total_experience_years=1,
        skills=["Excel"],
        raw_text="Bad Cand\n1 years of experience\nExcel",
    )
    good_eval = evaluator.evaluate(jd, good)
    bad_eval = evaluator.evaluate(jd, bad)
    assert good_eval.overall_match > bad_eval.overall_match
    assert good_eval.recommendation in {RECOMMENDATION_SHORTLIST, "Review"}
    assert bad_eval.recommendation == RECOMMENDATION_DOES_NOT_MEET
    assert 0 <= good_eval.overall_match <= 100
