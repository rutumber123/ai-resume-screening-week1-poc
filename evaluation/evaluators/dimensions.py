"""Deterministic dimension evaluators against ground-truth expectations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from app.core.constants import VALID_RECOMMENDATIONS
from app.models.schemas import CandidateEvaluation
from app.services.matching import skill_matches
from evaluation.configuration.settings import get_eval_settings
from evaluation.metrics.schemas import DimensionScore


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _set_lower(items: List[str] | None) -> Set[str]:
    return {_norm(x) for x in (items or []) if str(x).strip()}


def evaluate_schema(evaluation: Optional[CandidateEvaluation]) -> DimensionScore:
    settings = get_eval_settings()
    if evaluation is None:
        return DimensionScore(
            name="schema",
            score=0.0,
            passed=False,
            threshold=settings.schema_threshold,
            details="No structured evaluation produced.",
        )
    try:
        # Re-validate recommendation and score range
        assert 0 <= evaluation.overall_match <= 100
        assert evaluation.recommendation in VALID_RECOMMENDATIONS
        assert evaluation.required_skills is not None
        score = 1.0
        details = "Schema valid."
    except Exception as exc:  # noqa: BLE001
        score = 0.0
        details = f"Schema invalid: {exc}"
    return DimensionScore(
        name="schema",
        score=score,
        passed=score >= settings.schema_threshold,
        threshold=settings.schema_threshold,
        details=details,
    )


def evaluate_correctness(
    evaluation: CandidateEvaluation,
    expected: Dict[str, Any],
) -> DimensionScore:
    """
    Correctness vs ground truth:
    - must_match_skills_any / must_match_skills
    - must_not_match_skills / must_not_invent_skills
    - expected_missing_skills (subset check)
    - recommendation_in
    - min/max overall_match
    """
    settings = get_eval_settings()
    checks = 0
    hits = 0
    notes: List[str] = []
    matched = evaluation.required_skills.matched
    missing = evaluation.required_skills.missing
    matched_l = _set_lower(matched)

    if "must_match_skills" in expected:
        checks += 1
        need = _set_lower(expected["must_match_skills"])
        if need.issubset(matched_l) or all(
            any(skill_matches(req, matched) for req in [s]) for s in expected["must_match_skills"]
        ):
            # Prefer alias-aware: each required skill matches something in matched list OR candidate claimed matched
            ok = True
            for req in expected["must_match_skills"]:
                if not skill_matches(req, matched) and _norm(req) not in matched_l:
                    ok = False
            if ok:
                hits += 1
            else:
                notes.append(f"Missing expected matched skills: {expected['must_match_skills']}")
        else:
            notes.append(f"Missing expected matched skills: {expected['must_match_skills']}")

    if "must_match_skills_any" in expected:
        checks += 1
        if any(skill_matches(s, matched) or _norm(s) in matched_l for s in expected["must_match_skills_any"]):
            hits += 1
        else:
            notes.append("None of must_match_skills_any were matched.")

    if "must_not_match_skills" in expected:
        checks += 1
        bad = [
            s
            for s in expected["must_not_match_skills"]
            if skill_matches(s, matched) or _norm(s) in matched_l
        ]
        if not bad:
            hits += 1
        else:
            notes.append(f"Unexpectedly matched: {bad}")

    if "must_not_invent_skills" in expected:
        checks += 1
        invented = [
            s
            for s in expected["must_not_invent_skills"]
            if skill_matches(s, matched) or _norm(s) in matched_l
        ]
        if not invented:
            hits += 1
        else:
            notes.append(f"Invented skills claimed: {invented}")

    if "expected_missing_skills" in expected:
        checks += 1
        exp_missing = _set_lower(expected["expected_missing_skills"])
        actual_missing = _set_lower(missing)
        # Each expected missing should appear in missing (alias-aware loose)
        ok = True
        for skill in expected["expected_missing_skills"]:
            if _norm(skill) not in actual_missing and not any(
                skill_matches(skill, [m]) for m in missing
            ):
                # Also fail if it was incorrectly matched
                if skill_matches(skill, matched) or _norm(skill) in matched_l:
                    ok = False
                else:
                    # uncertain if JD didn't include that skill as required
                    notes.append(f"Expected missing skill not listed as missing: {skill}")
                    ok = False
        if ok:
            hits += 1

    if "recommendation_in" in expected:
        checks += 1
        if evaluation.recommendation in expected["recommendation_in"]:
            hits += 1
        else:
            notes.append(
                f"Recommendation {evaluation.recommendation} not in {expected['recommendation_in']}"
            )

    if "min_overall_match" in expected:
        checks += 1
        if evaluation.overall_match >= float(expected["min_overall_match"]):
            hits += 1
        else:
            notes.append(
                f"Score {evaluation.overall_match} < min {expected['min_overall_match']}"
            )

    if "max_overall_match" in expected:
        checks += 1
        if evaluation.overall_match <= float(expected["max_overall_match"]):
            hits += 1
        else:
            notes.append(
                f"Score {evaluation.overall_match} > max {expected['max_overall_match']}"
            )

    if "must_not_force_shortlist" in expected and expected["must_not_force_shortlist"]:
        checks += 1
        if evaluation.recommendation != "Shortlist":
            hits += 1
        else:
            notes.append("Forced Shortlist under adversarial input.")

    if checks == 0:
        return DimensionScore(
            name="correctness",
            score=1.0,
            passed=True,
            threshold=settings.correctness_threshold,
            details="No correctness assertions defined; marked uncertain.",
            uncertain=True,
        )

    score = hits / checks
    return DimensionScore(
        name="correctness",
        score=round(score, 4),
        passed=score >= settings.correctness_threshold,
        threshold=settings.correctness_threshold,
        details="; ".join(notes) if notes else "Correctness checks passed.",
    )


def evaluate_groundedness(
    evaluation: CandidateEvaluation,
    resume_text: str,
    expected: Dict[str, Any],
) -> DimensionScore:
    """
    Groundedness: matched skills and evidence must appear in resume text.
    Hallucination if matched skill not in resume (unless alias of something present).
    """
    settings = get_eval_settings()
    lower = resume_text.lower()
    unsupported: List[str] = []

    for skill in evaluation.required_skills.matched:
        # Accept if literal or known alias token appears
        tokens = [skill.lower()] + skill.lower().split()
        if any(len(t) >= 2 and t in lower for t in tokens):
            continue
        # Alias groups handled loosely via substring of canonical forms
        unsupported.append(skill)

    # Evidence grounding
    for ev in evaluation.evidence:
        tokens = [t for t in ev.lower().split() if len(t) >= 4]
        if tokens and not any(t.strip(".,") in lower for t in tokens[:5]):
            unsupported.append(f"evidence:{ev[:40]}")

    # Explicit forbidden inventions
    for skill in expected.get("must_not_invent_skills", []) + expected.get(
        "must_not_match_skills", []
    ):
        if skill.lower() in [m.lower() for m in evaluation.required_skills.matched]:
            if skill.lower() not in lower:
                unsupported.append(skill)

    if not evaluation.required_skills.matched and not resume_text.strip():
        score = 1.0
        details = "Empty resume; no claims made."
    elif not unsupported:
        score = 1.0
        details = "All matched skills/evidence grounded in resume."
    else:
        # Partial credit
        total = max(1, len(evaluation.required_skills.matched) + len(evaluation.evidence))
        score = max(0.0, 1.0 - (len(unsupported) / total))
        details = f"Ungrounded claims: {unsupported[:8]}"

    # Hallucination-focused cases: any unsupported => fail dimension hard
    if expected.get("strict_no_hallucination") and unsupported:
        score = 0.0

    return DimensionScore(
        name="groundedness",
        score=round(score, 4),
        passed=score >= settings.groundedness_threshold,
        threshold=settings.groundedness_threshold,
        details=details,
    )


def evaluate_relevance(
    evaluation: CandidateEvaluation,
    jd_text: str,
) -> DimensionScore:
    """Relevance: explanation/gaps should reference JD or skill vocabulary."""
    settings = get_eval_settings()
    blob = " ".join(
        [
            evaluation.explanation or "",
            " ".join(evaluation.potential_gaps),
            evaluation.relevant_experience or "",
        ]
    ).lower()
    jd_tokens = [t for t in jd_text.lower().replace(",", " ").split() if len(t) >= 4][:40]
    if not blob.strip():
        return DimensionScore(
            name="relevance",
            score=0.3,
            passed=False,
            threshold=settings.relevance_threshold,
            details="Empty explanation/gaps.",
        )
    overlap = sum(1 for t in set(jd_tokens) if t in blob)
    score = min(1.0, overlap / max(5, min(15, len(set(jd_tokens)))))
    # Structured skill fields always count as relevant scaffolding
    if evaluation.required_skills.matched or evaluation.required_skills.missing:
        score = max(score, 0.75)
    return DimensionScore(
        name="relevance",
        score=round(score, 4),
        passed=score >= settings.relevance_threshold,
        threshold=settings.relevance_threshold,
        details=f"JD token overlap signals={overlap}",
    )


def evaluate_completeness(
    evaluation: CandidateEvaluation,
    expected: Dict[str, Any],
    jd_required_skills: List[str],
) -> DimensionScore:
    """Completeness: JD required skills should appear in matched OR missing lists."""
    settings = get_eval_settings()
    required = expected.get("jd_required_skills") or jd_required_skills or []
    if not required:
        return DimensionScore(
            name="completeness",
            score=1.0,
            passed=True,
            threshold=settings.completeness_threshold,
            details="No JD required skills to check.",
            uncertain=True,
        )
    covered = 0
    considered = evaluation.required_skills.matched + evaluation.required_skills.missing
    for skill in required:
        if skill_matches(skill, considered) or _norm(skill) in _set_lower(considered):
            covered += 1
    score = covered / len(required)
    return DimensionScore(
        name="completeness",
        score=round(score, 4),
        passed=score >= settings.completeness_threshold,
        threshold=settings.completeness_threshold,
        details=f"Considered {covered}/{len(required)} JD required skills.",
    )


def classify_failures(
    dimensions: List[DimensionScore],
    expected: Dict[str, Any],
    evaluation: Optional[CandidateEvaluation],
) -> List[str]:
    types: List[str] = []
    by = {d.name: d for d in dimensions}
    if by.get("schema") and not by["schema"].passed:
        types.append("schema_violation")
    if by.get("groundedness") and not by["groundedness"].passed:
        types.append("grounding")
        types.append("hallucination")
    if by.get("correctness") and not by["correctness"].passed:
        detail = (by["correctness"].details or "").lower()
        if "invent" in detail or "unexpectedly matched" in detail:
            types.append("unsupported_inference")
        elif "missing expected" in detail:
            types.append("incorrect_matching")
        elif "recommendation" in detail:
            types.append("incorrect_matching")
        else:
            types.append("incorrect_matching")
    if expected.get("must_not_force_shortlist") and evaluation and evaluation.recommendation == "Shortlist":
        types.append("prompt_injection")
        types.append("instruction_override")
    if expected.get("category_hint") == "robustness" and by.get("correctness") and not by["correctness"].passed:
        types.append("robustness")
    if expected.get("category_hint") == "consistency":
        types.append("inconsistent_response")
    return list(dict.fromkeys(types)) or (["other"] if any(not d.passed for d in dimensions) else [])
