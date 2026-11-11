"""Application settings loaded from environment variables / .env."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ST-NLP"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    api_admin_key: str = "change-this-demo-key-before-use"
    database_url: str = "sqlite:///./stnlp.db"
    model_name: str = "svm"
    model_artifact_dir: str = "artifacts"
    confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
