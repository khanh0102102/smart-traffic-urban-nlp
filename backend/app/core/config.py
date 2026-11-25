"""Typed application settings loaded from environment variables and .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ST-NLP"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    database_url: str | None = None
    db_driver: str = "postgresql+psycopg"
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_name: str = "stnlp"
    db_user: str = "stnlp_user"
    db_password: str = "stnlp_dev_password"
    sql_echo: bool = False

    # Authentication secrets must be supplied through environment variables.
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_issuer: str = "st-nlp-api"
    jwt_audience: str = "st-nlp-client"

    google_client_id: str = ""
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Report attachment storage is outside PostgreSQL; mount this directory persistently.
    report_upload_dir: Path = PROJECT_ROOT / "backend" / "storage"
    report_max_upload_size_bytes: int = 5 * 1024 * 1024
    report_max_attachments: int = 5

    @property
    def sqlalchemy_url(self) -> URL:
        if self.database_url:
            return make_url(self.database_url)
        return URL.create(
            drivername=self.db_driver,
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


settings = Settings()
