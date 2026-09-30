# Week 2 Manager Status Summary

**Initiative:** GenAI Applications → LLM Testing → Agentic QA  
**Week:** 2 — LLM Testing & Evaluation POC  
**Status:** Complete (evaluation framework runnable on Week 1 SUT)

## Completed
- Reusable evaluation runner over the Resume Screening Assistant
- Categorized dataset (~50+ cases) with ground-truth expectations
- Dimensions: correctness, groundedness/hallucination, relevance, completeness, schema, consistency, robustness, adversarial, fairness
- Prompt versions v1/v2/v3 + intentional `v_regress_bad` for regression demo
- Result persistence + markdown reports + baseline regression compare
- Single command: `python scripts/run_evaluation.py` (CI-ready exit codes)
- PromptFoo config + DeepEval optional bridge
- Observability hook foundation for Week 3 / LangSmith

## How to demo
1. Run baseline eval (`--prompt-version v2 --set-baseline`)
2. Run `v_regress_bad` and show new failures / degradation in report
3. Restore `v2`/`v3` and show recovery

## Known limits
- Mock mode uses deterministic extraction; live prompt diffs need API key + `--use-llm`
- LLM-as-judge is optional/advisory, not ground truth
- Fairness tests are synthetic robustness checks only

## Week 3 direction
RAG-based assistant + retrieval/RAGAS evaluation + deeper observability
