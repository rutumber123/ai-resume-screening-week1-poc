"""
LLM behavior tests / failure-mode catalog.

These tests encode expected *application safeguards* and documented LLM failure modes.
They run without a live LLM (mock provider). Live-model probes are marked and skipped
unless LLM_PROVIDER is openai/azure with credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.constants import RECOMMENDATION_SHORTLIST
from app.services.resume_processor import ResumeProcessor
from app.services.screening import ScreeningService
from app.services.validation import OutputValidator

ROOT = Path(__file__).resolve().parents[2]


def _jd() -> str:
    return (ROOT / "sample_data/job_descriptions/senior_python_genai.txt").read_text(
        encoding="utf-8"
    )


def _resume(name: str) -> str:
    return (ROOT / "sample_data/resumes" / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Failure mode: Hallucination
# ---------------------------------------------------------------------------
def test_hallucination_guard_strips_unevidenced_skills() -> None:
    """Model must not claim skills absent from resume; processor strips them."""
    text = "Pat Example\nSkills: Python\n2 years of experience"
    llm_payload = {
        "name": "Pat Example",
        "total_experience_years": 2,
        "skills": ["Python", "Kubernetes", "RAG"],
        "programming_languages": [],
        "frameworks": [],
        "cloud_technologies": ["Kubernetes"],
        "databases": [],
        "certifications": [],
        "education": [],
        "previous_roles": [],
        "projects": [],
        "domain_experience": [],
    }
    profile = ResumeProcessor().process(text, llm_structured=llm_payload)
    tokens = {t.lower() for t in profile.all_skill_tokens()}
    assert "python" in tokens
    assert "kubernetes" not in tokens
    assert "rag" not in tokens


# ---------------------------------------------------------------------------
# Failure mode: Missing information
# ---------------------------------------------------------------------------
def test_missing_information_marked_not_specified() -> None:
    profile = ResumeProcessor().process("Just a name line\nSam Onlyname\n", filename="x.txt")
    # Experience absent
    assert profile.total_experience_years is None
    assert any("not specified" in w.lower() or "experience" in w.lower() for w in profile.extraction_warnings)


# ---------------------------------------------------------------------------
# Failure mode: Prompt injection / instruction override
# ---------------------------------------------------------------------------
def test_prompt_injection_does_not_override_recommendation() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [("adv.txt", _resume("18_adversarial_prompt_injection_harper_quinn.txt"))],
        use_llm=False,
    )
    ev = result.candidates[0]
    assert ev.recommendation != RECOMMENDATION_SHORTLIST
    assert OutputValidator().detect_injection(_resume("18_adversarial_prompt_injection_harper_quinn.txt"))


# ---------------------------------------------------------------------------
# Failure mode: Unsupported inference
# ---------------------------------------------------------------------------
def test_unsupported_inference_python_not_django() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [("priya.txt", _resume("19_python_no_django_inference_priya_shah.txt"))],
        use_llm=False,
    )
    matched = {s.lower() for s in result.candidates[0].required_skills.matched}
    assert "django" not in matched
    assert "fastapi" not in matched


# ---------------------------------------------------------------------------
# Failure mode: Long context
# ---------------------------------------------------------------------------
def test_long_resume_completes() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [("long.txt", _resume("13_very_long_elliot_nash.txt"))],
        use_llm=False,
    )
    assert len(result.candidates) == 1
    assert "Python" in result.candidates[0].required_skills.matched or any(
        "python" in s.lower() for s in result.candidates[0].required_skills.matched
    )


# ---------------------------------------------------------------------------
# Failure mode: Inconsistent output (determinism of scoring layer)
# ---------------------------------------------------------------------------
def test_scoring_is_deterministic_for_same_inputs() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    payload = [("alex.txt", _resume("01_excellent_match_alex_rivera.txt"))]
    r1 = service.screen_text_resumes(_jd(), payload, use_llm=False)
    r2 = service.screen_text_resumes(_jd(), payload, use_llm=False)
    assert r1.candidates[0].overall_match == r2.candidates[0].overall_match
    assert r1.candidates[0].recommendation == r2.candidates[0].recommendation


# ---------------------------------------------------------------------------
# Failure mode: Ambiguous requirements
# ---------------------------------------------------------------------------
def test_ambiguous_jd_still_returns_structured_result() -> None:
    jd = (ROOT / "sample_data/job_descriptions/fullstack_ambiguous.txt").read_text(
        encoding="utf-8"
    )
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        jd,
        [("sam.txt", _resume("03_partial_match_sam_patel.txt"))],
        use_llm=False,
    )
    assert result.candidates
    assert 0 <= result.candidates[0].overall_match <= 100


# ---------------------------------------------------------------------------
# Optional live LLM probe (skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.getenv("LLM_PROVIDER", "mock").lower() == "mock"
    or not os.getenv("OPENAI_API_KEY"),
    reason="Live LLM probe requires LLM_PROVIDER!=mock and OPENAI_API_KEY",
)
def test_live_llm_extraction_optional() -> None:
    service = ScreeningService()
    result = service.screen_text_resumes(
        _jd(),
        [("alex.txt", _resume("01_excellent_match_alex_rivera.txt"))],
        use_llm=True,
    )
    assert result.candidates[0].overall_match >= 50
