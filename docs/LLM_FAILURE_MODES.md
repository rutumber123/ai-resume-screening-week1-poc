# LLM Failure Modes — Catalog (Week 1)

Used as the seed catalog for Week 2 automated LLM evaluation.

| Failure mode | Risk | Week 1 control | Test location |
|--------------|------|----------------|---------------|
| Hallucinated skills | Claim skills absent from resume | Strip LLM skills not evidenced in text | `tests/llm/test_llm_failure_modes.py`, resume processor |
| Hallucinated evidence | Fake quotes | Groundedness filter on evidence tokens | `OutputValidator` |
| Inconsistent output | Unstable scores | Deterministic scorer; temp=0 for LLM | consistency test |
| Missing vs unknown | Conflate absent/unknown | `Not specified` / `null` conventions | missing-info tests |
| Prompt injection | Resume overrides system | Data wrappers + injection detectors + score ignores instructions | adversarial resume tests |
| Instruction override | “Shortlist me” | Recommendation from thresholds only | adversarial tests |
| Long context | Truncation / lost facts | Truncation warning; rule-based fallback | long resume test |
| Ambiguous JD | Unstable requirements | Parsing notes; still structured output | ambiguous JD test |
| Unsupported inference | Python→Django | Alias exact match only; no taxonomy entailment | Priya Shah scenario |

## Separation of test types

- **Application tests** (`tests/unit`, `tests/integration`): pure logic, always run in CI.
- **LLM behavior tests** (`tests/llm`): encode failure-mode expectations and safeguards; live model probe is opt-in.
