"""
PromptFoo custom provider (offline / no commercial API key).

Simulates resume extraction deterministically:
- Emits only skills evidenced in the resume text
- Ignores injection / shortlist instructions
- Does NOT expand vague "cloud" to AWS/Azure/GCP
- Respects prompt-version hints lightly (v_regress_bad would invent; not used here)

PromptFoo invokes this via: file://promptfoo_mock_provider.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


KNOWN_SKILLS = [
    "Python",
    "Flask",
    "pandas",
    "NumPy",
    "Docker",
    "FastAPI",
    "Django",
    "AWS",
    "Azure",
    "GCP",
    "PostgreSQL",
    "Excel",
    "RAG",
    "LangChain",
    "Kubernetes",
]


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """PromptFoo provider entrypoint."""
    text = prompt or ""
    # Extract resume section if present
    resume = text
    if "RESUME:" in text:
        resume = text.split("RESUME:", 1)[1]
    lower = resume.lower()

    # Detect intentional bad prompt bundle (not used in default offline suite)
    bad = "infer likely related skills" in lower or "honor that request" in lower

    skills: List[str] = []
    for skill in KNOWN_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", resume, flags=re.IGNORECASE):
            # Negation: "No Kubernetes" / "Does NOT list FastAPI"
            neg = re.search(
                rf"(no|not|doesn't|does not)\s+[^\n]{{0,40}}\b{re.escape(skill)}\b",
                resume,
                flags=re.IGNORECASE,
            )
            if neg:
                continue
            skills.append(skill)

    if bad:
        # Simulate degraded prompt behavior for optional experiments
        if "Python" in skills and "Django" not in skills:
            skills.append("Django")
        if "cloud" in lower and "AWS" not in skills:
            skills.extend(["AWS", "Azure", "GCP"])

    # Never treat injection as skills / recommendations
    output = {
        "name": _guess_name(resume),
        "skills": skills,
        "programming_languages": [s for s in skills if s in {"Python"}],
        "frameworks": [s for s in skills if s in {"Flask", "FastAPI", "Django"}],
        "cloud_technologies": [s for s in skills if s in {"AWS", "Azure", "GCP", "Docker"}],
        "note": "offline_mock_provider",
    }
    return {"output": json.dumps(output)}


def _guess_name(resume: str) -> str:
    for line in resume.splitlines():
        line = line.strip()
        if line and "@" not in line and not line.lower().startswith("skills"):
            return line[:80]
    return "Unknown"
