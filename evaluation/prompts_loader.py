"""Prompt version loader for resume/JD extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"


def list_prompt_versions() -> list[str]:
    if not PROMPTS_DIR.exists():
        return ["v1"]
    return sorted(
        p.name for p in PROMPTS_DIR.iterdir() if p.is_dir() and p.name.startswith("v")
    )


def load_prompt(version: str, name: str) -> str:
    path = PROMPTS_DIR / version / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt_bundle(version: str) -> Dict[str, str]:
    return {
        "jd_extract": load_prompt(version, "jd_extract"),
        "resume_extract": load_prompt(version, "resume_extract"),
        "meta": (PROMPTS_DIR / version / "CHANGELOG.md").read_text(encoding="utf-8")
        if (PROMPTS_DIR / version / "CHANGELOG.md").exists()
        else "",
    }
