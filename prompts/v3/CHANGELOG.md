# Prompt V3

Stronger output constraints and stricter grounding.

## Changes vs V2
- Atomic skill naming preference for JD
- Vague experience → null
- Stricter "omit if uncertain" for resume skills
- Explicit "JSON only" output constraint
- Stronger anti-vendor-expansion for vague cloud claims

## Note for demo regression
Intentionally more conservative; may miss skills that V2 captures (extraction recall tradeoff).
A deliberately weak variant for regression demo is available as `v3_weak` if needed —
V3 itself is strict, not broken.
