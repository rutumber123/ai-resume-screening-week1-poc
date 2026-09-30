"""Week 2 evaluation configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(ROOT / "evaluation" / "configuration" / "eval_defaults.env")),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    correctness_threshold: float = Field(default=0.70, alias="EVAL_CORRECTNESS_THRESHOLD")
    groundedness_threshold: float = Field(default=0.80, alias="EVAL_GROUNDEDNESS_THRESHOLD")
    relevance_threshold: float = Field(default=0.70, alias="EVAL_RELEVANCE_THRESHOLD")
    completeness_threshold: float = Field(default=0.70, alias="EVAL_COMPLETENESS_THRESHOLD")
    schema_threshold: float = Field(default=1.0, alias="EVAL_SCHEMA_THRESHOLD")
    pass_rate_threshold: float = Field(default=0.80, alias="EVAL_PASS_RATE_THRESHOLD")
    consistency_score_tolerance: float = Field(
        default=5.0, alias="EVAL_CONSISTENCY_SCORE_TOLERANCE"
    )
    consistency_repeats: int = Field(default=3, alias="EVAL_CONSISTENCY_REPEATS")
    prompt_version: str = Field(default="v2", alias="PROMPT_VERSION")
    use_llm_judge: bool = Field(default=False, alias="EVAL_USE_LLM_JUDGE")
    results_dir: str = Field(default="results", alias="EVAL_RESULTS_DIR")
    baseline_run_id: str = Field(default="", alias="EVAL_BASELINE_RUN_ID")

    @property
    def results_path(self) -> Path:
        path = ROOT / self.results_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()
