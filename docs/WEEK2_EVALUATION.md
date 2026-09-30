# Week 2 — LLM Testing & Evaluation

## Question this week answers

> How do we know whether the resume screening LLM application produces reliable and acceptable results?

## Architecture

```text
Week 1 ScreeningService (SUT)
        ↓
evaluation_data/week2_catalog.json
        ↓
EvaluationRunner
        ↓
Dimension evaluators (correctness, groundedness, relevance, completeness, schema, consistency)
        ↓
Aggregate metrics + pass/fail thresholds
        ↓
results/<run_id>.json + .md
        ↓
Regression compare vs baseline
```

## Run evaluation (single command)

```powershell
cd "c:\Users\rutumber.nath\Documents\Week 1 POC"
.\.venv\Scripts\python.exe scripts\generate_week2_catalog.py
.\.venv\Scripts\python.exe scripts\run_evaluation.py --set-baseline
```

Exit code `0` = quality gate passed; `1` = failed.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--categories positive,hallucination` | Subset by category |
| `--test-ids P01,A01` | Single/multiple cases |
| `--prompt-version v2` | Prompt bundle under `prompts/` |
| `--compare-prompts v1,v3` | Multi-prompt comparison table |
| `--baseline-run-id <id>` | Explicit regression baseline |
| `--use-llm` | Live LLM extraction (needs API key) |
| `--set-baseline` | Save run id to `results/BASELINE_RUN_ID.txt` |

### Regression demo

```powershell
# 1) Establish baseline with good prompt
.\.venv\Scripts\python.exe scripts\run_evaluation.py --prompt-version v2 --set-baseline

# 2) Introduce bad prompt → detect regressions
.\.venv\Scripts\python.exe scripts\run_evaluation.py --prompt-version v_regress_bad

# 3) Restore / compare improved prompts
.\.venv\Scripts\python.exe scripts\run_evaluation.py --prompt-version v3 --compare-prompts v1,v2
```

Note: with `LLM_PROVIDER=mock`, normal prompt versions use deterministic parsers.
The special prompt `v_regress_bad` simulates a degraded extractor offline (invents related skills)
so regression detection can be demonstrated without an API key. Live prompt diffs for v1/v2/v3
appear when `--use-llm` is enabled with a real provider.

## Metric definitions

| Metric | Definition |
|--------|------------|
| Correctness | Fraction of ground-truth assertions passed (skills, score bounds, recommendation) |
| Groundedness | Matched skills/evidence supported by resume text; unsupported → fail |
| Relevance | Explanation/gaps overlap with JD vocabulary + structured skill fields |
| Completeness | JD required skills appear in matched ∪ missing |
| Schema | CandidateEvaluation validates; score 0–100; recommendation enum |
| Consistency | Repeat-run score spread ≤ tolerance; recommendation stable |
| Pass rate | `#passed / #total` |
| Avg evaluation score | Mean of per-case dimension averages |

Thresholds: `evaluation/configuration/eval_defaults.env` (override via `.env`).

## Dataset

- Catalog: `evaluation_data/week2_catalog.json` (~50+ categorized cases)
- Version: `2.0.0`
- Categories: positive, negative, partial_match, hallucination, prompt_injection, robustness, consistency, schema_validation, fairness, edge_cases, completeness

## Tooling

| Tool | Role in Week 2 |
|------|----------------|
| **Custom runner** | Primary automated eval + regression + reports |
| **PromptFoo** | Optional prompt-unit tests (`evaluation/tools/promptfooconfig.yaml`) |
| **DeepEval** | Optional faithfulness bridge (`evaluation/tools/deepeval_bridge.py`) |

## Test pyramid

1. Application unit tests — `tests/unit`
2. Integration — `tests/integration`
3. LLM behavior — `tests/llm`
4. Evaluation metrics — `tests/evaluation`
5. Security/adversarial — `tests/security`
6. Full eval suite — `scripts/run_evaluation.py`

## Observability

Local spans under `results/traces/` via `evaluation/observability.py` (LangSmith-ready metadata shape for Week 3).

## Fairness note

Fairness cases only check that **irrelevant personal/hobby/name variants** do not swing scores for equivalent qualifications. This is a technical robustness check — **not** a claim about real-world hiring fairness.

## Path to Week 3

RAG assistant → retrieval tests → RAGAS → richer observability → CI regression gates.
