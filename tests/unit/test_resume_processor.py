"""Unit tests for resume extraction."""

from pathlib import Path

from app.core.constants import NOT_SPECIFIED
from app.services.resume_processor import ResumeProcessor

ROOT = Path(__file__).resolve().parents[2]
PROC = ResumeProcessor()


def test_extract_excellent_resume() -> None:
    text = (ROOT / "sample_data/resumes/01_excellent_match_alex_rivera.txt").read_text(
        encoding="utf-8"
    )
    profile = PROC.process(text, filename="01.txt")
    assert "Alex" in profile.name
    assert profile.total_experience_years == 7
    tokens = [t.lower() for t in profile.all_skill_tokens()]
    assert "python" in tokens
    assert "fastapi" in tokens


def test_empty_resume_not_specified() -> None:
    profile = PROC.process("", filename="empty.txt")
    assert profile.name == NOT_SPECIFIED
    assert profile.total_experience_years is None
    assert any("empty" in w.lower() for w in profile.extraction_warnings)


def test_no_skills_section_still_finds_tech() -> None:
    text = (ROOT / "sample_data/resumes/05_no_skills_section_riley_brooks.txt").read_text(
        encoding="utf-8"
    )
    profile = PROC.process(text, filename="05.txt")
    tokens = [t.lower() for t in profile.all_skill_tokens()]
    assert "python" in tokens
    assert any("skills section" in w.lower() for w in profile.extraction_warnings)


def test_llm_hallucinated_skill_stripped() -> None:
    text = "Jamie Example\nSkills: Python\n3 years of experience"
    llm = {
        "name": "Jamie Example",
        "total_experience_years": 3,
        "skills": ["Python", "Django"],
        "programming_languages": ["Python"],
        "frameworks": ["Django"],
        "cloud_technologies": [],
        "databases": [],
        "certifications": [],
        "education": [],
        "previous_roles": [],
        "projects": [],
        "domain_experience": [],
    }
    profile = PROC.process(text, filename="x.txt", llm_structured=llm)
    tokens = [t.lower() for t in profile.all_skill_tokens()]
    assert "python" in tokens
    assert "django" not in tokens
