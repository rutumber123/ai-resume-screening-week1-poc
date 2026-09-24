# Week 1 Manager Status Summary

**Initiative:** GenAI Applications → LLM Testing → Agentic QA (12 weeks)  
**Week:** 1 — AI Resume Screening Assistant POC  
**Status:** Complete (runnable POC with tests, sample data, evaluation baseline, docs)

## What was delivered

- End-to-end screening pipeline (JD → extract → match → score → compare → validate)
- FastAPI application layer + Streamlit demo UI
- Configurable weighted scoring with documented methodology
- Synthetic sample data: 3 JDs, 20 resumes (including adversarial)
- ~25 baseline evaluation scenarios
- Automated unit, integration, and LLM failure-mode tests
- README + architecture, scoring, edge-case, and Week 2 docs

## What works

- Multi-resume screening with structured JSON results
- Transparent scores and recommendations
- Prompt-injection resume does **not** force Shortlist
- Python does **not** imply Django/FastAPI
- Runs offline with `LLM_PROVIDER=mock` (no API key)
- Optional OpenAI/Azure extraction behind the same interfaces

## Issues / limitations identified

- Rule-based skill lexicon coverage is finite
- Compound JD skill phrases can under-match
- Contradiction handling is shallow
- Live LLM quality not measured until Week 2 harness

## What remains / Week 2

- Automated LLM evaluation metrics & prompt regression
- Model comparison and CI eval gates
- Expanded security corpus
- Path to Agentic QA around this SUT

## Demo checklist

1. `pip install -r requirements.txt`
2. `streamlit run app/ui/streamlit_app.py`
3. Screen Alex / Jordan / Sam / Harper against GenAI JD
4. `pytest -q`
