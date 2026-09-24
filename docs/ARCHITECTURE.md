# Architecture — AI Resume Screening Assistant

## Design goals

1. **Pipeline, not chatbot** — each stage has a clear interface and structured I/O.
2. **Replaceable LLM** — `LLMClient` abstracts mock / OpenAI / Azure.
3. **Deterministic scoring** — weights and thresholds live in configuration.
4. **Safe failure** — schema validation; ungrounded evidence removed; injection flagged.
5. **Eval-ready** — baseline dataset and separated app vs LLM test suites.

## Data flow

```
┌────────────┐     ┌─────────────┐     ┌──────────────────┐
│ Streamlit  │────▶│ FastAPI API │────▶│ ScreeningService │
│ UI (demo)  │     │ (optional)  │     └────────┬─────────┘
└────────────┘     └─────────────┘              │
                                                ▼
                    ┌───────────────────────────────────────────┐
                    │ JDProcessor  (+ optional LLM extract)     │
                    │ DocumentProcessor (pdf/docx/txt)          │
                    │ ResumeProcessor (+ optional LLM extract)  │
                    │ CandidateEvaluator (weighted match)       │
                    │ OutputValidator (schema + grounding)      │
                    │ Comparison builder (from structured rows) │
                    └───────────────────────────────────────────┘
                                                │
                                                ▼
                                        ScreeningResult JSON
```

## Component interfaces

| Component | Input | Output |
|-----------|-------|--------|
| `DocumentProcessor` | filename + bytes | text + warnings |
| `JDProcessor` | raw JD text (+ optional LLM JSON) | `JobDescription` |
| `ResumeProcessor` | raw resume text (+ optional LLM JSON) | `CandidateProfile` |
| `CandidateEvaluator` | JD + profile | `CandidateEvaluation` |
| `ScreeningService` | JD + files | `ScreeningResult` |
| `LLMClient` | system + user prompts | JSON dict |

## Anti-hallucination controls

- LLM-extracted skills must appear in resume text (literal / catalog check).
- Evidence snippets must be grounded in resume tokens.
- Matched required skills re-checked against profile/raw text.
- Resume content wrapped as untrusted `<<<DATA>>>` for live LLM calls.
- Injection patterns detected and surfaced as validation warnings; scoring remains requirement-based.

## Extension seams (Week 2+)

- Swap `LLMClient` for additional providers.
- Add `app/evaluation/metrics.py` for faithfulness / consistency scores.
- Hook `ScreeningService.screen` into CI as a regression target.
- Add tracing middleware around API without changing services.
