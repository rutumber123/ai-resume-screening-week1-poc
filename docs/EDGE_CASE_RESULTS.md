# Edge Case & Adversarial Results (Week 1)

Observed with `LLM_PROVIDER=mock` (deterministic pipeline). Re-run via:

```bash
python scripts/run_baseline_eval.py
pytest tests/integration tests/llm -q
```

| # | Scenario | Observed behavior |
|---|----------|-------------------|
| 1 | Excellent match | High score; Shortlist/strong Review; many required skills matched; grounded evidence |
| 2 | Poor match | Low score; Does Not Meet; most required skills missing |
| 3 | Partial match | Mid score; some required matched (Python/FastAPI), GenAI stack missing |
| 4 | Missing information | Fields `Not specified` / null experience; low match; warnings |
| 5 | No skills section | Tech still extracted from prose; warning about missing skills section |
| 6 | Irrelevant skills | Noise skills ignored for scoring; relevant stack still matched |
| 7 | Less experience | Experience component < 1; gap called out |
| 8 | More experience | Experience component = 1; high overall when skills align |
| 9 | Exact skills | High required-skill ratio |
| 10 | Related different skills | Java/Spring does **not** count as Python/FastAPI |
| 11–12 | Similar profiles | Scores within ~1 point (deterministic) |
| 13 | Very long resume | Completes; real skills matched; Django not invented |
| 14 | Empty resume | Rejected or near-zero evaluation; no crash of process pool |
| 15 | Malformed doc | Parsed as text junk; low/no skill match; no hard crash |
| 16 | Contradictory info | Completes; first extracted experience may win; limitation documented |
| 17 | Unusual formatting | Core skills still detected |
| 18 | Ambiguous JD | Parsing notes; screening still returns structured result |
| 19 | JD no clear experience | `min_experience_years` often null; note recorded |
| 20 | Mixed required/preferred | Ambiguous JD puts mixed list into required-like extraction |
| 21 | Prompt injection | Injection warning; **not** forced Shortlist; low match |
| 22 | Unsupported inference | Python present ≠ Django/FastAPI matched |

## Failures / limitations noted

- Compound requirements (“LangChain or LlamaIndex”) may appear as one missing skill string.
- Contradiction resolution is naive (no conflict graph).
- Empty uploads fail validation by design (safer than fabricating a candidate).
- Live LLM behavior not asserted in default CI (optional skipped probe).
