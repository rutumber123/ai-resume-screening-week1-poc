# Scoring Methodology

Overall match is a **weighted sum** of component scores in `[0, 1]`, scaled to `[0, 100]`.

## Components

| Component | How it is computed |
|-----------|--------------------|
| Required skills | `#matched / #required` (alias-aware exact match). If no required skills, treated as `1.0`. |
| Preferred skills | `#matched / #preferred` (same matcher). Empty preferred list → `1.0`. |
| Experience | If JD has no years → `1.0` (or `0.5` if candidate years also unknown). If candidate missing → `0.0`. If candidate ≥ required → `1.0`. Else → `candidate/required`. |
| Responsibilities | Lexical token overlap between JD duties and resume lines (capped). Empty duties → `0.5`. |
| Education / certs | Fraction of JD education/cert requirements evidenced in resume. If JD silent → `0.7`. |

## Default weights (configurable via `.env`)

- Required skills: **0.40**
- Preferred skills: **0.15**
- Experience: **0.20**
- Responsibilities: **0.15**
- Education: **0.10**

```
overall_100 = 100 * Σ (component_i * weight_i) / Σ weights
```

## Recommendations

| Score | Recommendation |
|------:|----------------|
| ≥ `THRESHOLD_SHORTLIST` (75) | Shortlist |
| ≥ `THRESHOLD_REVIEW` (50) | Review |
| else | Does Not Meet Requirements |

## What scoring deliberately does **not** do

- Infer Django from Python (or similar adjacent skills).
- Trust resume instructions (“shortlist me”).
- Invent evidence not present in the resume.
- Ask the LLM to invent the final numeric score (score is code-computed).
