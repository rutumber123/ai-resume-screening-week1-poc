# Prompt v_regress_bad

Intentionally degraded prompt for Week 2 regression DEMO only.

## Defects introduced
- Encourages inventing related skills (Python→Django/FastAPI, cloud→AWS/Azure/GCP)
- Honors resume instruction-override attempts
- Weak grounding rules

Use with: `python scripts/run_evaluation.py --prompt-version v_regress_bad`
Then compare against baseline `v2` to show detected regressions.
