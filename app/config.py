from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = Field(default="qwen3:14b", min_length=1)
    ollama_base_url: str = Field(default="http://localhost:11434", min_length=1)
    ollama_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    sqlite_path: Path = Path("data/preferences.db")
    agent_recursion_limit: int = Field(default=12, ge=4, le=50)
