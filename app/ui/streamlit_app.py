"""Streamlit demo UI for the AI Resume Screening Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path when launched via `streamlit run`
# streamlit_app.py lives at app/ui/ → parents[2] is the project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.screening import ScreeningService

setup_logging()
settings = get_settings()

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📋",
    layout="wide",
)

st.title("AI Resume Screening Assistant")
st.caption(
    "Week 1 GenAI POC — structured pipeline: JD → Resume → Extract → Match → Score → Compare"
)

with st.sidebar:
    st.header("Configuration")
    st.write(f"**LLM provider:** `{settings.llm_provider}`")
    st.write(
        f"**Weights:** req={settings.weight_required_skills}, "
        f"pref={settings.weight_preferred_skills}, "
        f"exp={settings.weight_experience}, "
        f"resp={settings.weight_responsibilities}, "
        f"edu={settings.weight_education}"
    )
    st.write(
        f"**Thresholds:** Shortlist ≥ {settings.threshold_shortlist}, "
        f"Review ≥ {settings.threshold_review}"
    )
    use_llm = st.checkbox(
        "Use LLM extraction (if provider ≠ mock)",
        value=settings.llm_provider != "mock",
    )
    st.markdown("---")
    st.markdown(
        "Tip: load sample JD from `sample_data/job_descriptions/` "
        "and resumes from `sample_data/resumes/`."
    )

sample_jd_path = ROOT / "sample_data" / "job_descriptions" / "senior_python_genai.txt"
default_jd = ""
if sample_jd_path.exists():
    default_jd = sample_jd_path.read_text(encoding="utf-8")

jd_text = st.text_area(
    "Job Description",
    value=default_jd,
    height=260,
    placeholder="Paste a natural-language job description…",
)

uploads = st.file_uploader(
    "Upload candidate resumes (PDF, DOCX, TXT, MD)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 3])
with col_a:
    run = st.button("Start Screening", type="primary", use_container_width=True)

if run:
    if not jd_text.strip():
        st.error("Please provide a Job Description.")
    elif not uploads:
        st.error("Please upload at least one resume.")
    else:
        progress = st.progress(0, text="Starting screening…")
        status = st.empty()
        try:
            status.info("Processing JD and resumes…")
            payloads = [(f.name, f.getvalue()) for f in uploads]
            progress.progress(30, text="Extracting and matching…")
            service = ScreeningService(settings)
            result = service.screen(jd_text, payloads, use_llm=use_llm)
            progress.progress(100, text="Done")
            st.session_state["screening_result"] = result
            status.success(
                f"Screened {len(result.candidates)} candidate(s) "
                f"for **{result.job_title}** (provider: {result.llm_provider})."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Screening failed: {exc}")
            progress.empty()

result = st.session_state.get("screening_result")
if result:
    st.subheader("Candidate Comparison")
    if result.processing_notes:
        with st.expander("Processing notes"):
            for note in result.processing_notes:
                st.write(f"- {note}")

    table_rows = []
    for row in result.comparison:
        table_rows.append(
            {
                "Candidate": row.candidate,
                "Match": row.match,
                "Experience": row.experience,
                "Required Skills": f"{row.required_skills_matched}/{row.required_skills_total}",
                "Missing Skills": ", ".join(row.missing_skills) or "—",
                "Recommendation": row.recommendation,
            }
        )
    st.dataframe(table_rows, use_container_width=True)

    st.subheader("Detailed Candidate Analysis")
    names = [c.candidate for c in result.candidates]
    selected = st.selectbox("Select candidate", names)
    detail = next(c for c in result.candidates if c.candidate == selected)

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Match", f"{detail.overall_match:.1f}")
    m2.metric("Recommendation", detail.recommendation)
    m3.metric(
        "Required Skills",
        f"{len(detail.required_skills.matched)}/"
        f"{len(detail.required_skills.matched)+len(detail.required_skills.missing)}",
    )

    st.markdown("#### Experience")
    st.write(
        f"Required: {detail.experience.required_years if detail.experience.required_years is not None else 'Not specified'} years  \n"
        f"Candidate: {detail.experience.candidate_years if detail.experience.candidate_years is not None else 'Not specified'} years  \n"
        f"Note: {detail.experience.note}"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Required Skills — Matched")
        st.write(detail.required_skills.matched or ["—"])
        st.markdown("#### Preferred Skills — Matched")
        st.write(detail.preferred_skills.matched or ["—"])
    with c2:
        st.markdown("#### Required Skills — Missing")
        st.write(detail.required_skills.missing or ["—"])
        st.markdown("#### Preferred Skills — Missing")
        st.write(detail.preferred_skills.missing or ["—"])

    st.markdown("#### Relevant Experience")
    st.write(detail.relevant_experience)

    st.markdown("#### Potential Gaps")
    st.write(detail.potential_gaps or ["—"])

    st.markdown("#### Evidence")
    st.write(detail.evidence or ["—"])

    st.markdown("#### Explanation")
    st.write(detail.explanation)

    st.markdown("#### Component Scores")
    st.json(detail.component_scores.model_dump())

    if detail.validation_warnings:
        st.warning("Validation warnings:\n\n- " + "\n- ".join(detail.validation_warnings))
