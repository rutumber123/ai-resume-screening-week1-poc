# AI Resume Screening Assistant — Week 1 + Week 2 POC

GenAI application that screens multiple candidate resumes against a job description, produces **structured evaluations**, transparent **matching scores**, and a **multi-candidate comparison** view.

**Week 2** adds an automated **LLM testing & evaluation framework** around this application (dataset → runner → metrics → reports → regression).

This is **not** a chatbot. It is a pipeline:

```
Job Description → JD Processing → Resume Upload → Document Extraction
→ Candidate Extraction → Requirement Matching → Scoring → Structured Results
→ Comparison → Validation
→ (Week 2) Evaluation Dataset → Test Runner → Metrics → Report → Regression
```

Default mode uses a **deterministic extractor + scorer** (`LLM_PROVIDER=mock`) so the POC runs without API keys. Swap to OpenAI/Azure for LLM-assisted extraction without changing the pipeline.

---

## Problem Statement

Manual resume screening is slow, inconsistent, and hard to audit. Hiring teams need:

- Structured match/gap analysis against JD requirements
- Transparent scores (not opaque “AI vibes”)
- Evidence grounded in the resume
- Resilience to missing data, noise, and prompt-injection attempts
- A foundation for later **LLM evaluation / Agentic QA** work

---

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component details.

| Layer | Responsibility |
|-------|----------------|
| `app/ui` | Streamlit demo UI |
| `app/api` | FastAPI endpoints |
| `app/services` | Document, JD, resume, LLM, matching, evaluation, validation, screening |
| `app/models` | Pydantic schemas |
| `app/core` | Config, logging, constants |
| `app/evaluation` | Baseline dataset loaders |
| `tests/unit` | Application logic tests |
| `tests/llm` | LLM failure-mode / behavior tests |
| `evaluation_data` | Baseline scenarios (20–25+) |
| `sample_data` | Synthetic JDs & resumes |

---

## Setup

### Prerequisites

- Python 3.10+
- Windows / macOS / Linux

### Install

```bash
cd "Week 1 POC"
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
```

By default `.env` uses `LLM_PROVIDER=mock` (no API key required).

### Optional: live LLM extraction

Set in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Or Azure:

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT
```

**Never commit `.env` or API keys.**

---

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | `mock` / `openai` / `azure` | `mock` |
| `LLM_TEMPERATURE` | Generation temperature | `0.0` |
| `WEIGHT_REQUIRED_SKILLS` | Score weight | `0.40` |
| `WEIGHT_PREFERRED_SKILLS` | Score weight | `0.15` |
| `WEIGHT_EXPERIENCE` | Score weight | `0.20` |
| `WEIGHT_RESPONSIBILITIES` | Score weight | `0.15` |
| `WEIGHT_EDUCATION` | Score weight | `0.10` |
| `THRESHOLD_SHORTLIST` | Score ≥ → Shortlist | `75` |
| `THRESHOLD_REVIEW` | Score ≥ → Review | `50` |
| `MAX_FILE_SIZE_MB` | Upload limit | `5` |
| `SUPPORTED_EXTENSIONS` | Allowed types | `.pdf,.docx,.txt,.md` |

Scoring methodology: [docs/SCORING.md](docs/SCORING.md).

---

## Usage

### Streamlit UI (recommended demo)

```bash
streamlit run app/ui/streamlit_app.py
```

1. Paste/edit the Job Description  
2. Upload multiple resumes from `sample_data/resumes/`  
3. Click **Start Screening**  
4. Inspect comparison table + detailed analysis / evidence / warnings  

### FastAPI

```bash
uvicorn app.api.main:app --reload --port 8000
```

- Health: `GET http://localhost:8000/health`  
- Screen: `POST http://localhost:8000/api/v1/screen` (multipart: `job_description`, `resumes`)  
- Docs: `http://localhost:8000/docs`

### CLI demo

```bash
python scripts/run_demo.py
python scripts/run_baseline_eval.py
```

### Week 2 — LLM evaluation suite

```bash
python scripts/generate_week2_catalog.py
python scripts/run_evaluation.py --set-baseline
```

See [docs/WEEK2_EVALUATION.md](docs/WEEK2_EVALUATION.md) and [docs/WEEK2_STATUS.md](docs/WEEK2_STATUS.md).

---

## Evaluation

**Week 1 baseline:** `evaluation_data/baseline_scenarios.json` (~25 scenarios).

**Week 2 catalog:** `evaluation_data/week2_catalog.json` (50+ categorized cases with ground truth).

```bash
pytest tests/integration/test_baseline_evaluation.py -q
python scripts/run_baseline_eval.py
python scripts/run_evaluation.py
```

---

## Testing

```bash
# All automated tests
pytest -q

# Application only
pytest tests/unit tests/integration -q

# LLM behavior / failure-mode tests
pytest tests/llm -q
```

| Suite | Purpose |
|-------|---------|
| `tests/unit` | File validation, JD/resume parsing, matching, scoring, schema validation |
| `tests/integration` | Multi-candidate screening, adversarial cases, baseline dataset |
| `tests/llm` | Hallucination guards, injection, long context, determinism, optional live probe |

---

## Test Scenarios (sample data)

| File | Scenario |
|------|----------|
| `01_excellent_match_*` | Strong GenAI match |
| `02_poor_match_*` | Marketing profile vs eng JD |
| `03_partial_match_*` | Partial skill overlap |
| `04_missing_info_*` | Sparse resume |
| `05_no_skills_section_*` | Skills only in prose |
| `06_irrelevant_skills_*` | Noise skills |
| `07_less_experience_*` | Below min years |
| `08_more_experience_*` | Far above min years |
| `09_exact_skills_*` | Near-exact stack |
| `10_related_skills_*` | Java/Spring ≠ Python/GenAI |
| `11` / `12` | Similar profiles |
| `13_very_long_*` | Long-context stress |
| `14_empty_resume` | Empty file |
| `15_malformed_*` | Malformed content |
| `16_contradictory_*` | Conflicting claims |
| `17_unusual_formatting_*` | Odd separators |
| `18_adversarial_*` | Prompt injection |
| `19_python_no_django_*` | No unsupported inference |
| `20_data_engineer_*` | Alternate JD domain |

Observed behaviors: [docs/EDGE_CASE_RESULTS.md](docs/EDGE_CASE_RESULTS.md).

---

## Known Limitations

- Rule-based extraction is lexicon-limited; exotic skill names may be missed without LLM mode  
- Compound JD lines (“LangChain or LlamaIndex”) may score as a single unmatched skill token  
- Responsibility relevance uses lexical overlap, not deep semantic understanding  
- PDF/DOCX quality depends on source encoding; scanned PDFs are out of scope  
- Mock mode does not call a real LLM; live LLM variance needs Week 2 eval harness  
- Contradiction detection is limited (warnings / first-number extraction)  
- No auth, persistence, or multi-tenant isolation (POC only)

---

## Future Extension (Week 2+)

This POC is the **system under test** for a broader GenAI QA program:

1. **Prompt regression testing** — golden JD/resume pairs, snapshot structured outputs  
2. **Automated LLM evaluation** — faithfulness, hallucination rate, injection resistance metrics  
3. **Model comparison** — same dataset across providers/models  
4. **RAG evaluation** — if JD/resume knowledge bases are added  
5. **Security testing** — expanded injection corpus, jailbreak suites  
6. **Observability** — traces, token usage, latency SLOs  
7. **CI/CD evaluation gates** — fail builds on eval regressions  
8. **Agentic QA** — agents that generate cases, execute screening, and file defects  

See [docs/WEEK2_EXTENSIONS.md](docs/WEEK2_EXTENSIONS.md) and [docs/WEEK1_STATUS.md](docs/WEEK1_STATUS.md).

---

## Project Structure

```
Week 1 POC/
├── app/
│   ├── api/            # FastAPI
│   ├── core/           # config, logging, constants
│   ├── services/       # pipeline components
│   ├── models/         # schemas
│   ├── evaluation/     # dataset helpers
│   ├── ui/             # Streamlit
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── llm/
├── evaluation_data/
├── sample_data/
│   ├── resumes/
│   └── job_descriptions/
├── docs/
├── scripts/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Demo Flow (for stakeholders)

1. Start UI with mock provider  
2. Load Senior Python/GenAI JD  
3. Upload Alex (strong), Jordan (poor), Sam (partial), Harper (adversarial)  
4. Show comparison table ranking  
5. Open Harper detail → injection warning, **not** Shortlist  
6. Open Alex detail → matched skills + grounded evidence  
7. Run `pytest -q` to show automated coverage  

---

## License / Data

Synthetic sample data only — no real personal information.
