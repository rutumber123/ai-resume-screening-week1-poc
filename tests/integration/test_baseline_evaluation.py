"""Baseline evaluation dataset runner (application-level assertions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.evaluation.dataset import load_baseline, read_text, resolve_jd
from app.services.screening import ScreeningService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def service() -> ScreeningService:
    return ScreeningService(Settings(LLM_PROVIDER="mock"))


@pytest.fixture(scope="module")
def dataset() -> dict:
    return load_baseline()


def test_baseline_scenario_count(dataset: dict) -> None:
    assert len(dataset["scenarios"]) >= 20


def test_run_core_baseline_scenarios(service: ScreeningService, dataset: dict) -> None:
    failures = []
    for scenario in dataset["scenarios"]:
        sid = scenario["id"]
        expected = scenario["expected"]
        jd = resolve_jd(dataset, scenario["jd"])

        if "resumes" in scenario:
            resumes = [
                (Path(p).name, read_text(p)) for p in scenario["resumes"]
            ]
            result = service.screen_text_resumes(jd, resumes, use_llm=False)
            if expected.get("comparison_row_count"):
                if len(result.comparison) != expected["comparison_row_count"]:
                    failures.append(f"{sid}: comparison count")
            if expected.get("excellent_outranks_poor"):
                by = {c.candidate: c.overall_match for c in result.candidates}
                if by.get("Alex Rivera", 0) <= by.get("Jordan Lee", 100):
                    failures.append(f"{sid}: ranking invariant")
            continue

        resume_path = scenario["resume"]
        resume_text = read_text(resume_path)
        try:
            result = service.screen_text_resumes(
                jd, [(Path(resume_path).name, resume_text)], use_llm=False
            )
        except ValueError:
            if expected.get("completes_or_skips_gracefully") or expected.get("handles_empty"):
                continue
            failures.append(f"{sid}: crashed with ValueError")
            continue

        ev = result.candidates[0]
        if "recommendation_in" in expected and ev.recommendation not in expected["recommendation_in"]:
            failures.append(
                f"{sid}: recommendation {ev.recommendation} not in {expected['recommendation_in']}"
            )
        if "min_overall_match" in expected and ev.overall_match < expected["min_overall_match"]:
            failures.append(f"{sid}: match {ev.overall_match} < min")
        if "max_overall_match" in expected and ev.overall_match > expected["max_overall_match"]:
            failures.append(f"{sid}: match {ev.overall_match} > max")
        if expected.get("must_not_force_shortlist") and ev.recommendation == "Shortlist":
            failures.append(f"{sid}: injection forced shortlist")
        if expected.get("validation_warning_mentions_injection"):
            joined = " ".join(ev.validation_warnings).lower()
            if "injection" not in joined and "instruction" not in joined:
                failures.append(f"{sid}: missing injection warning")
        if "must_not_match_skills" in expected:
            matched = {s.lower() for s in ev.required_skills.matched}
            for skill in expected["must_not_match_skills"]:
                if skill.lower() in matched:
                    failures.append(f"{sid}: unexpectedly matched {skill}")
        if "must_match_skills_any" in expected:
            matched = {s.lower() for s in ev.required_skills.matched}
            if not any(s.lower() in matched for s in expected["must_match_skills_any"]):
                # Also accept presence via preferred in partial cases? No — required only
                # For resumes that have the skill, matcher should find it if JD requires it
                failures.append(f"{sid}: missing expected skill matches")
        if expected.get("experience_below_required"):
            if (
                ev.experience.required_years is not None
                and ev.experience.candidate_years is not None
                and not (ev.experience.candidate_years < ev.experience.required_years)
            ):
                failures.append(f"{sid}: expected below experience")

    assert not failures, "Baseline failures:\n" + "\n".join(failures)
