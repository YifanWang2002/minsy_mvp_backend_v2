"""Application configuration loaded from .env via Pydantic settings."""

from __future__ import annotations

import json
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Runtime settings for API, infrastructure and middleware."""

    app_name: str = Field(default="Minsy", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    chat_debug_trace_enabled: bool = Field(
        default=False,
        alias="CHAT_DEBUG_TRACE_ENABLED",
    )
    chat_debug_trace_mode: str = Field(
        default="verbose",
        alias="CHAT_DEBUG_TRACE_MODE",
    )
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
    openai_response_model: str = Field(default="gpt-5.2", alias="OPENAI_RESPONSE_MODEL")
    mcp_server_url_dev: str = Field(
        default="https://dev.minsyai.com/mcp",
        alias="MCP_SERVER_URL_DEV",
    )
    mcp_server_url_prod: str = Field(
        default="https://mcp.minsyai.com/mcp",
        alias="MCP_SERVER_URL_PROD",
    )
    mcp_context_secret: str | None = Field(default=None, alias="MCP_CONTEXT_SECRET")
    mcp_context_ttl_seconds: int = Field(default=300, alias="MCP_CONTEXT_TTL_SECONDS")
    telegram_enabled: bool = Field(default=True, alias="TELEGRAM_ENABLED")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_bot_username: str = Field(default="", alias="TELEGRAM_BOT_USERNAME")
    telegram_webhook_secret_token: str = Field(default="", alias="TELEGRAM_WEBHOOK_SECRET_TOKEN")
    telegram_connect_ttl_seconds: int = Field(default=600, alias="TELEGRAM_CONNECT_TTL_SECONDS")
    telegram_test_batches_enabled: bool = Field(
        default=True,
        alias="TELEGRAM_TEST_BATCHES_ENABLED",
    )
    telegram_webapp_base_url: str = Field(
        default="https://api.minsyai.com",
        alias="TELEGRAM_WEBAPP_BASE_URL",
    )
    telegram_test_payment_provider_token: str = Field(
        default="",
        alias="TELEGRAM_TEST_PAYMENT_PROVIDER_TOKEN",
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
    celery_timezone: str = Field(default="UTC", alias="CELERY_TIMEZONE")

    postgres_backup_enabled: bool = Field(default=True, alias="POSTGRES_BACKUP_ENABLED")
    postgres_backup_dir: str = Field(default="backups/postgres", alias="POSTGRES_BACKUP_DIR")
    postgres_backup_retention_count: int = Field(
        default=14,
        alias="POSTGRES_BACKUP_RETENTION_COUNT",
    )
    postgres_backup_hour_utc: int = Field(default=3, alias="POSTGRES_BACKUP_HOUR_UTC")
    postgres_backup_minute_utc: int = Field(default=0, alias="POSTGRES_BACKUP_MINUTE_UTC")
    postgres_pg_dump_bin: str = Field(default="pg_dump", alias="POSTGRES_PG_DUMP_BIN")

    user_email_csv_export_enabled: bool = Field(
        default=True,
        alias="USER_EMAIL_CSV_EXPORT_ENABLED",
    )
    user_email_csv_path: str = Field(default="exports/user_emails.csv", alias="USER_EMAIL_CSV_PATH")
    user_email_csv_export_interval_minutes: int = Field(
        default=60,
        alias="USER_EMAIL_CSV_EXPORT_INTERVAL_MINUTES",
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "https://app.minsyai.com",
            "https://dev.minsyai.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        alias="CORS_ORIGINS",
    )
    cors_origin_regex: str | None = Field(
        default=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        alias="CORS_ORIGIN_REGEX",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use `.env` as the primary source over process environment variables."""
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
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

    @field_validator("cors_origin_regex", mode="before")
    @classmethod
    def _parse_cors_origin_regex(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("postgres_backup_retention_count")
    @classmethod
    def _validate_backup_retention(cls, value: int) -> int:
        if value < 1:
            raise ValueError("POSTGRES_BACKUP_RETENTION_COUNT must be >= 1.")
        return value

    @field_validator("postgres_backup_hour_utc")
    @classmethod
    def _validate_backup_hour(cls, value: int) -> int:
        if not 0 <= value <= 23:
            raise ValueError("POSTGRES_BACKUP_HOUR_UTC must be in [0, 23].")
        return value

    @field_validator("postgres_backup_minute_utc")
    @classmethod
    def _validate_backup_minute(cls, value: int) -> int:
        if not 0 <= value <= 59:
            raise ValueError("POSTGRES_BACKUP_MINUTE_UTC must be in [0, 59].")
        return value

    @field_validator("user_email_csv_export_interval_minutes")
    @classmethod
    def _validate_email_export_interval(cls, value: int) -> int:
        if value < 1:
            raise ValueError("USER_EMAIL_CSV_EXPORT_INTERVAL_MINUTES must be >= 1.")
        return value

    @field_validator("telegram_connect_ttl_seconds")
    @classmethod
    def _validate_telegram_connect_ttl_seconds(cls, value: int) -> int:
        if value < 60:
            raise ValueError("TELEGRAM_CONNECT_TTL_SECONDS must be >= 60.")
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
    def effective_cors_origins(self) -> list[str]:
        """CORS allow-list with required first-party web origins included."""
        required = ("https://app.minsyai.com", "https://dev.minsyai.com")
        merged: list[str] = []
        seen: set[str] = set()
        for origin in (*self.cors_origins, *required):
            normalized = origin.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    @property
    def runtime_env(self) -> str:
        """Normalized runtime environment key for cross-cutting feature switches."""
        raw = self.app_env.strip().lower()
        if raw in {"dev", "development", "local"}:
            return "dev"
        if raw in {"prod", "production"}:
            return "prod"
        if raw in {"test", "testing"}:
            return "test"
        return raw

    @property
    def is_dev_mode(self) -> bool:
        """Whether app runs in local/development mode."""
        return self.runtime_env in {"dev", "test"}

    @property
    def mcp_server_url(self) -> str:
        """Resolve MCP server URL from the unified APP_ENV mode switch."""
        if self.runtime_env == "prod":
            return self.mcp_server_url_prod
        return self.mcp_server_url_dev

    @property
    def strategy_mcp_server_url(self) -> str:
        """Backward-compatible alias for strategy MCP URL."""
        return self.mcp_server_url

    @property
    def backtest_mcp_server_url(self) -> str:
        """Backward-compatible alias for backtest MCP URL."""
        return self.mcp_server_url

    @property
    def effective_mcp_context_secret(self) -> str:
        """Signing secret for MCP context token propagation."""
        if isinstance(self.mcp_context_secret, str) and self.mcp_context_secret.strip():
            return self.mcp_context_secret.strip()
        return self.secret_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
