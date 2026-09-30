"""Load Week 2 evaluation catalog cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "evaluation_data" / "week2_catalog.json"


def load_catalog(path: Path | None = None) -> Dict[str, Any]:
    p = path or CATALOG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Catalog missing at {p}. Run: python scripts/generate_week2_catalog.py"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def filter_cases(
    catalog: Dict[str, Any],
    categories: Optional[Iterable[str]] = None,
    test_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    cases = catalog["cases"]
    if categories:
        cats = {c.lower() for c in categories}
        cases = [c for c in cases if c.get("category", "").lower() in cats]
    if test_ids:
        ids = {t.upper() for t in test_ids}
        cases = [c for c in cases if c.get("id", "").upper() in ids]
    return cases
