"""Integration tests for multi-candidate screening."""

from pathlib import Path

from app.core.config import Settings
from app.core.constants import RECOMMENDATION_DOES_NOT_MEET
from app.services.screening import ScreeningService

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> str:
    return (ROOT / "sample_data" / "resumes" / name).read_text(encoding="utf-8")


def _jd() -> str:
    return (ROOT / "sample_data/job_descriptions/senior_python_genai.txt").read_text(
        encoding="utf-8"
    )


def test_multi_candidate_comparison_order() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [
            ("alex.txt", _load("01_excellent_match_alex_rivera.txt")),
            ("jordan.txt", _load("02_poor_match_jordan_lee.txt")),
            ("sam.txt", _load("03_partial_match_sam_patel.txt")),
        ],
        use_llm=False,
    )
    assert len(result.candidates) == 3
    assert len(result.comparison) == 3
    by_name = {c.candidate: c for c in result.candidates}
    # Excellent should outrank poor
    assert by_name["Alex Rivera"].overall_match > by_name["Jordan Lee"].overall_match
    assert by_name["Jordan Lee"].recommendation == RECOMMENDATION_DOES_NOT_MEET
    # Comparison derived from structured results
    assert result.comparison[0].match >= result.comparison[-1].match


def test_adversarial_resume_not_forced_shortlist() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [
            (
                "adversarial.txt",
                _load("18_adversarial_prompt_injection_harper_quinn.txt"),
            )
        ],
        use_llm=False,
    )
    ev = result.candidates[0]
    assert ev.recommendation != "Shortlist"
    assert ev.overall_match < 75
    assert any(
        "injection" in w.lower() or "instruction" in w.lower()
        for w in ev.validation_warnings
    )


def test_python_does_not_imply_django_or_fastapi() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [("priya.txt", _load("19_python_no_django_inference_priya_shah.txt"))],
        use_llm=False,
    )
    ev = result.candidates[0]
    matched = [s.lower() for s in ev.required_skills.matched]
    assert "django" not in matched
    assert "fastapi" not in matched
    assert any(s.lower() == "python" for s in ev.required_skills.matched)


def test_similar_profiles_similar_scores() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    result = service.screen_text_resumes(
        _jd(),
        [
            ("nina.txt", _load("11_similar_profile_nina_volkov.txt")),
            ("nora.txt", _load("12_similar_profile_nora_volkov.txt")),
        ],
        use_llm=False,
    )
    scores = [c.overall_match for c in result.candidates]
    assert abs(scores[0] - scores[1]) <= 1.0


def test_empty_resume_rejected() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    try:
        service.screen_text_resumes(_jd(), [("empty.txt", "")], use_llm=False)
        assert False, "Expected ValueError for empty resume"
    except ValueError as exc:
        assert "No resumes could be processed" in str(exc)


def test_empty_resume_raises_or_low_score() -> None:
    service = ScreeningService(Settings(LLM_PROVIDER="mock"))
    # Provide whitespace-only which decodes but strips empty
    try:
        result = service.screen(
            _jd(),
            [("empty.txt", b"   ")],
            use_llm=False,
        )
        assert result.candidates[0].overall_match < 50
    except ValueError:
        # Also acceptable: no resumes processed
        pass
