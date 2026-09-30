"""Security / adversarial evaluation tests (pytest layer)."""

from pathlib import Path

from app.core.config import Settings
from app.services.screening import ScreeningService
from app.services.validation import OutputValidator

ROOT = Path(__file__).resolve().parents[2]


def test_injection_variants_not_shortlisted() -> None:
    jd = (ROOT / "sample_data/job_descriptions/senior_python_genai.txt").read_text(
        encoding="utf-8"
    )
    service = ScreeningService(Settings(llm_provider="mock"))
    files = [
        "sample_data/resumes/18_adversarial_prompt_injection_harper_quinn.txt",
        "sample_data/resumes_extra/injection_in_skills.txt",
        "sample_data/resumes_extra/injection_footer.txt",
        "sample_data/resumes_extra/injection_in_experience.txt",
        "sample_data/resumes_extra/hidden_system_override.txt",
    ]
    for rel in files:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert OutputValidator().detect_injection(text)
        result = service.screen_text_resumes(jd, [(Path(rel).name, text)], use_llm=False)
        assert result.candidates[0].recommendation != "Shortlist"
        assert result.candidates[0].overall_match < 75
