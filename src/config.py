"""Global configuration management for Stochastic Marine Drift using pydantic-settings."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env or environment variables."""

    # LLM Host & Serving Configuration
    llm_host_url: str = "http://localhost:8000/v1"
    llm_api_key: Optional[str] = None
    research_model_name: str = "Qwen/Qwen3.8-27B"
    parser_model_name: str = "Qwen/Qwen3.5-4B"
    llm_timeout_seconds: float = 30.0

    # System-One Judge Configuration (Kev / Jev)
    judge_enabled: bool = True
    judge_provider: str = "kev"  # 'kev', 'jev', 'heuristic'
    judge_model_name: str = "jaredpalmer/kev-0.8b"
    judge_host_url: str = "http://localhost:8000/v1"
    judge_confidence_threshold: float = 0.85
    typesafe_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global singleton instance
settings = Settings()
