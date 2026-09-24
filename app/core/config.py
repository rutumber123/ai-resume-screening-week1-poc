"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Resume Screening POC."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview", alias="AZURE_OPENAI_API_VERSION"
    )
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    weight_required_skills: float = Field(default=0.40, alias="WEIGHT_REQUIRED_SKILLS")
    weight_preferred_skills: float = Field(default=0.15, alias="WEIGHT_PREFERRED_SKILLS")
    weight_experience: float = Field(default=0.20, alias="WEIGHT_EXPERIENCE")
    weight_responsibilities: float = Field(default=0.15, alias="WEIGHT_RESPONSIBILITIES")
    weight_education: float = Field(default=0.10, alias="WEIGHT_EDUCATION")

    threshold_shortlist: float = Field(default=75.0, alias="THRESHOLD_SHORTLIST")
    threshold_review: float = Field(default=50.0, alias="THRESHOLD_REVIEW")

    max_file_size_mb: float = Field(default=5.0, alias="MAX_FILE_SIZE_MB")
    max_resumes_per_request: int = Field(default=20, alias="MAX_RESUMES_PER_REQUEST")
    supported_extensions: str = Field(
        default=".pdf,.docx,.txt,.md", alias="SUPPORTED_EXTENSIONS"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    @field_validator("llm_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return (value or "mock").strip().lower()

    @property
    def supported_extension_list(self) -> List[str]:
        return [
            ext.strip().lower()
            for ext in self.supported_extensions.split(",")
            if ext.strip()
        ]

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    def matching_weights(self) -> dict[str, float]:
        return {
            "required_skills": self.weight_required_skills,
            "preferred_skills": self.weight_preferred_skills,
            "experience": self.weight_experience,
            "responsibilities": self.weight_responsibilities,
            "education": self.weight_education,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
