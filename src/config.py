"""Application configuration loaded from .env via Pydantic settings."""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for API, infrastructure and middleware."""

    app_name: str = Field(default="Minsy", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    secret_key: str = Field(alias="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=1440,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    auth_rate_limit: int = Field(default=30, alias="AUTH_RATE_LIMIT")
    auth_rate_window: int = Field(default=60, alias="AUTH_RATE_WINDOW")
    openai_response_model: str = Field(default="gpt-5", alias="OPENAI_RESPONSE_MODEL")
    mcp_env: str = Field(default="prod", alias="MCP_ENV")
    mcp_server_url_dev: str = Field(
        default="https://dev.minsyai.com/mcp",
        alias="MCP_SERVER_URL_DEV",
    )
    mcp_server_url_prod: str = Field(
        default="https://mcp.minsyai.com/mcp",
        alias="MCP_SERVER_URL_PROD",
    )
    mcp_server_url_override: str | None = Field(
        default=None,
        alias="MCP_SERVER_URL",
    )

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="123456", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="minsy_pgsql", alias="POSTGRES_DB")
    postgres_pool_size: int = Field(default=10, alias="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(default=20, alias="POSTGRES_MAX_OVERFLOW")
    sqlalchemy_echo: bool = Field(default=False, alias="SQLALCHEMY_ECHO")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=50, alias="REDIS_MAX_CONNECTIONS")
    celery_broker_url: str | None = Field(default=None, alias="CELERY_BROKER_URL")
    celery_result_backend: str | None = Field(default=None, alias="CELERY_RESULT_BACKEND")
    celery_task_default_queue: str = Field(default="backtest", alias="CELERY_TASK_DEFAULT_QUEUE")
    celery_task_time_limit_seconds: int = Field(
        default=1800,
        alias="CELERY_TASK_TIME_LIMIT_SECONDS",
    )
    celery_task_soft_time_limit_seconds: int = Field(
        default=1740,
        alias="CELERY_TASK_SOFT_TIME_LIMIT_SECONDS",
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=1,
        alias="CELERY_WORKER_PREFETCH_MULTIPLIER",
    )
    celery_task_acks_late: bool = Field(default=True, alias="CELERY_TASK_ACKS_LATE")
    celery_task_always_eager: bool = Field(default=False, alias="CELERY_TASK_ALWAYS_EAGER")

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            if normalized.startswith("["):
                return json.loads(normalized)
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return value

    @property
    def database_url(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            password = quote_plus(self.redis_password)
            return (
                f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def effective_celery_broker_url(self) -> str:
        if isinstance(self.celery_broker_url, str) and self.celery_broker_url.strip():
            return self.celery_broker_url.strip()
        return self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        if isinstance(self.celery_result_backend, str) and self.celery_result_backend.strip():
            return self.celery_result_backend.strip()
        return self.redis_url

    @property
    def mcp_server_url(self) -> str:
        if (
            isinstance(self.mcp_server_url_override, str)
            and self.mcp_server_url_override.strip()
        ):
            return self.mcp_server_url_override.strip()

        mode = self.mcp_env.strip().lower()
        if mode in {"dev", "development", "local"}:
            return self.mcp_server_url_dev
        return self.mcp_server_url_prod

    @mcp_server_url.setter
    def mcp_server_url(self, value: str) -> None:
        self.mcp_server_url_override = value

    @property
    def strategy_mcp_server_url(self) -> str:
        """Backward-compatible alias for strategy MCP URL."""
        return self.mcp_server_url

    @strategy_mcp_server_url.setter
    def strategy_mcp_server_url(self, value: str) -> None:
        self.mcp_server_url = value

    @property
    def backtest_mcp_server_url(self) -> str:
        """Backward-compatible alias for backtest MCP URL."""
        return self.mcp_server_url

    @backtest_mcp_server_url.setter
    def backtest_mcp_server_url(self, value: str) -> None:
        self.mcp_server_url = value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
