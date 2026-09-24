"""Unit tests for JD processing."""

from pathlib import Path

from app.services.jd_processor import JDProcessor

ROOT = Path(__file__).resolve().parents[2]


def test_parse_structured_jd() -> None:
    text = (ROOT / "sample_data/job_descriptions/senior_python_genai.txt").read_text(
        encoding="utf-8"
    )
    jd = JDProcessor().process(text)
    assert "Python" in jd.required_skills or any(
        "python" in s.lower() for s in jd.required_skills
    )
    assert jd.min_experience_years == 5 or jd.min_experience_years == 5.0
    assert jd.title != "Not specified"
    assert jd.responsibilities


def test_ambiguous_jd_notes() -> None:
    text = (ROOT / "sample_data/job_descriptions/fullstack_ambiguous.txt").read_text(
        encoding="utf-8"
    )
    jd = JDProcessor().process(text)
    assert jd.raw_text
    # Experience may be unclear ("several years")
    assert isinstance(jd.parsing_notes, list)


def test_empty_jd_raises() -> None:
    try:
        JDProcessor().process("   ")
        assert False, "Expected ValueError"
    except ValueError:
        pass
