from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings

ALL_SETTING_KEYS = [
    "APP_NAME",
    "APP_ENV",
    "DEBUG",
    "HOST",
    "PORT",
    "API_V1_PREFIX",
    "LOG_LEVEL",
    "OPENAI_API_KEY",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_POOL_SIZE",
    "POSTGRES_MAX_OVERFLOW",
    "SQLALCHEMY_ECHO",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "REDIS_MAX_CONNECTIONS",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "CELERY_TASK_DEFAULT_QUEUE",
    "CELERY_TASK_TIME_LIMIT_SECONDS",
    "CELERY_TASK_SOFT_TIME_LIMIT_SECONDS",
    "CELERY_WORKER_PREFETCH_MULTIPLIER",
    "CELERY_TASK_ACKS_LATE",
    "CELERY_TASK_ALWAYS_EAGER",
    "CORS_ORIGINS",
]


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ALL_SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_reads_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=MinsyTest",
                "APP_ENV=test",
                "DEBUG=false",
                "HOST=127.0.0.1",
                "PORT=9999",
                "API_V1_PREFIX=/api/test",
                "LOG_LEVEL=DEBUG",
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "POSTGRES_HOST=localhost",
                "POSTGRES_PORT=5432",
                "POSTGRES_USER=postgres",
                "POSTGRES_PASSWORD=123456",
                "POSTGRES_DB=minsy_pgsql",
                "POSTGRES_POOL_SIZE=5",
                "POSTGRES_MAX_OVERFLOW=8",
                "SQLALCHEMY_ECHO=true",
                "REDIS_HOST=localhost",
                "REDIS_PORT=6379",
                "REDIS_DB=0",
                "REDIS_MAX_CONNECTIONS=15",
                "CORS_ORIGINS=[\"http://localhost:3000\"]",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.app_name == "MinsyTest"
    assert settings.openai_api_key == "test-key"
    assert settings.postgres_db == "minsy_pgsql"
    assert settings.redis_db == 0
    assert settings.sqlalchemy_echo is True
    assert settings.cors_origins == ["http://localhost:3000"]


def test_settings_missing_required_env_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=MinsyTest",
                "APP_ENV=test",
                "DEBUG=false",
                "POSTGRES_HOST=localhost",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)
