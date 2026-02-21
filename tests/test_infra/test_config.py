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
    "SENTRY_DSN",
    "SENTRY_ENV",
    "SENTRY_RELEASE",
    "SENTRY_TRACES_SAMPLE_RATE",
    "SENTRY_PROFILES_SAMPLE_RATE",
    "SENTRY_HTTP_STATUS_CAPTURE_ENABLED",
    "SENTRY_HTTP_STATUS_MIN_CODE",
    "SENTRY_HTTP_STATUS_MAX_CODE",
    "SENTRY_HTTP_STATUS_EXCLUDE_PATHS",
    "CHAT_DEBUG_TRACE_ENABLED",
    "OPENAI_API_KEY",
    "OPENAI_COST_TRACKING_ENABLED",
    "OPENAI_PRICING_JSON",
    "MCP_SERVER_URL_DEV",
    "MCP_SERVER_URL_PROD",
    "MCP_SERVER_URL",
    "MCP_SERVER_URL_STRATEGY_DEV",
    "MCP_SERVER_URL_STRATEGY_PROD",
    "MCP_SERVER_URL_BACKTEST_DEV",
    "MCP_SERVER_URL_BACKTEST_PROD",
    "MCP_SERVER_URL_MARKET_DATA_DEV",
    "MCP_SERVER_URL_MARKET_DATA_PROD",
    "MCP_SERVER_URL_STRESS_DEV",
    "MCP_SERVER_URL_STRESS_PROD",
    "MCP_SERVER_URL_TRADING_DEV",
    "MCP_SERVER_URL_TRADING_PROD",
    "MCP_CONTEXT_SECRET",
    "MCP_CONTEXT_TTL_SECONDS",
    "PAPER_TRADING_ENABLED",
    "PAPER_TRADING_ENQUEUE_ON_START",
    "PAPER_TRADING_EXECUTE_ORDERS",
    "PAPER_TRADING_LOOP_INTERVAL_SECONDS",
    "PAPER_TRADING_MAX_RETRIES",
    "PAPER_TRADING_KILL_SWITCH_GLOBAL",
    "PAPER_TRADING_KILL_SWITCH_USERS",
    "PAPER_TRADING_KILL_SWITCH_DEPLOYMENTS",
    "PAPER_TRADING_BROKER_RETRY_MAX_ATTEMPTS",
    "PAPER_TRADING_BROKER_RETRY_BACKOFF_SECONDS",
    "PAPER_TRADING_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "PAPER_TRADING_CIRCUIT_BREAKER_RECOVERY_SECONDS",
    "PAPER_TRADING_RUNTIME_HEALTH_STALE_SECONDS",
    "BACKTEST_MAX_BARS",
    "BACKTEST_STALE_JOB_CLEANUP_ENABLED",
    "BACKTEST_STALE_JOB_CLEANUP_INTERVAL_MINUTES",
    "BACKTEST_RUNNING_STALE_MINUTES",
    "BACKTEST_RESULT_MAX_TRADES",
    "BACKTEST_RESULT_MAX_EQUITY_POINTS",
    "BACKTEST_RESULT_MAX_RETURNS",
    "BACKTEST_RESULT_MAX_EVENTS",
    "TRADING_CREDENTIALS_SECRET",
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "ALPACA_TRADING_BASE_URL",
    "ALPACA_MARKET_DATA_BASE_URL",
    "ALPACA_MARKET_DATA_STREAM_URL",
    "ALPACA_STOCKS_FEED",
    "ALPACA_CRYPTO_FEED",
    "ALPACA_REQUEST_RATE_LIMIT_PER_MINUTE",
    "ALPACA_STREAM_RECONNECT_BASE_SECONDS",
    "ALPACA_STREAM_MAX_RETRIES",
    "MARKET_DATA_BACKFILL_LIMIT",
    "MARKET_DATA_AGGREGATE_TIMEFRAMES",
    "MARKET_DATA_AGGREGATE_TIMEZONE",
    "MARKET_DATA_RING_CAPACITY_1M",
    "MARKET_DATA_RING_CAPACITY_AGG",
    "MARKET_DATA_FACTOR_CACHE_MAX_ENTRIES",
    "MARKET_DATA_CHECKPOINT_TTL_SECONDS",
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
    "CELERY_WORKER_MAX_MEMORY_PER_CHILD",
    "CELERY_WORKER_PREFETCH_MULTIPLIER",
    "CELERY_TASK_ACKS_LATE",
    "CELERY_TASK_ALWAYS_EAGER",
    "CELERY_TIMEZONE",
    "FLOWER_ENABLED",
    "FLOWER_HOST",
    "FLOWER_PORT",
    "FLOWER_USER",
    "FLOWER_PASSWORD",
    "POSTGRES_BACKUP_ENABLED",
    "POSTGRES_BACKUP_DIR",
    "POSTGRES_BACKUP_RETENTION_COUNT",
    "POSTGRES_BACKUP_HOUR_UTC",
    "POSTGRES_BACKUP_MINUTE_UTC",
    "POSTGRES_PG_DUMP_BIN",
    "USER_EMAIL_CSV_EXPORT_ENABLED",
    "USER_EMAIL_CSV_PATH",
    "USER_EMAIL_CSV_EXPORT_INTERVAL_MINUTES",
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


def test_mcp_server_url_uses_env_file_values_over_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MCP_SERVER_URL_DEV", "https://process.dev/mcp")
    monkeypatch.setenv("MCP_SERVER_URL_PROD", "https://process.prod/mcp")
    monkeypatch.setenv("MCP_SERVER_URL", "https://process.override/mcp")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=dev",
                "MCP_SERVER_URL_DEV=https://dotenv.dev/mcp",
                "MCP_SERVER_URL_PROD=https://dotenv.prod/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.runtime_env == "dev"
    assert settings.mcp_server_url_dev == "https://dotenv.dev/mcp"
    assert settings.mcp_server_url_prod == "https://dotenv.prod/mcp"
    assert settings.mcp_server_url == "https://dotenv.dev/mcp"


def test_mcp_server_url_ignores_legacy_override_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_SERVER_URL", "https://process.override/mcp")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=prod",
                "MCP_SERVER_URL_DEV=https://dotenv.dev/mcp",
                "MCP_SERVER_URL_PROD=https://dotenv.prod/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.mcp_server_url == "https://dotenv.prod/mcp"


def test_mcp_server_url_is_controlled_by_app_env_not_legacy_mcp_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_ENV", "prod")

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=dev",
                "MCP_SERVER_URL_DEV=https://dotenv.dev/mcp",
                "MCP_SERVER_URL_PROD=https://dotenv.prod/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.mcp_server_url == "https://dotenv.dev/mcp"


def test_sentry_http_status_capture_settings_parse_and_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "SENTRY_HTTP_STATUS_CAPTURE_ENABLED=true",
                "SENTRY_HTTP_STATUS_MIN_CODE=401",
                "SENTRY_HTTP_STATUS_MAX_CODE=599",
                "SENTRY_HTTP_STATUS_EXCLUDE_PATHS=[\"/api/v1/health\",\"/api/v1/status\",\"/favicon.ico\"]",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.sentry_http_status_capture_enabled is True
    assert settings.sentry_http_status_min_code == 401
    assert settings.sentry_http_status_max_code == 599
    assert settings.sentry_http_status_exclude_paths == [
        "/api/v1/health",
        "/api/v1/status",
        "/favicon.ico",
    ]


def test_sentry_http_status_capture_invalid_range_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "SENTRY_HTTP_STATUS_MIN_CODE=599",
                "SENTRY_HTTP_STATUS_MAX_CODE=400",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_celery_worker_max_memory_per_child_reads_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "CELERY_WORKER_MAX_MEMORY_PER_CHILD=262144",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.celery_worker_max_memory_per_child == 262144


def test_celery_worker_max_memory_per_child_invalid_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "CELERY_WORKER_MAX_MEMORY_PER_CHILD=0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_backtest_limits_read_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "BACKTEST_MAX_BARS=4200000",
                "BACKTEST_STALE_JOB_CLEANUP_ENABLED=true",
                "BACKTEST_STALE_JOB_CLEANUP_INTERVAL_MINUTES=12",
                "BACKTEST_RUNNING_STALE_MINUTES=45",
                "BACKTEST_RESULT_MAX_TRADES=1234",
                "BACKTEST_RESULT_MAX_EQUITY_POINTS=2345",
                "BACKTEST_RESULT_MAX_RETURNS=3456",
                "BACKTEST_RESULT_MAX_EVENTS=456",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.backtest_max_bars == 4_200_000
    assert settings.backtest_stale_job_cleanup_enabled is True
    assert settings.backtest_stale_job_cleanup_interval_minutes == 12
    assert settings.backtest_running_stale_minutes == 45
    assert settings.backtest_result_max_trades == 1234
    assert settings.backtest_result_max_equity_points == 2345
    assert settings.backtest_result_max_returns == 3456
    assert settings.backtest_result_max_events == 456


def test_backtest_limits_invalid_values_raise_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "BACKTEST_MAX_BARS=0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)

    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "BACKTEST_STALE_JOB_CLEANUP_INTERVAL_MINUTES=0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)

    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "BACKTEST_RESULT_MAX_TRADES=-1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_domain_mcp_server_urls_fallback_to_legacy_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=dev",
                "MCP_SERVER_URL_DEV=https://dotenv.dev/mcp",
                "MCP_SERVER_URL_PROD=https://dotenv.prod/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.strategy_mcp_server_url == "https://dotenv.dev/mcp"
    assert settings.backtest_mcp_server_url == "https://dotenv.dev/mcp"
    assert settings.market_data_mcp_server_url == "https://dotenv.dev/mcp"
    assert settings.stress_mcp_server_url == "https://dotenv.dev/mcp"
    assert settings.trading_mcp_server_url == "https://dotenv.dev/mcp"


def test_domain_mcp_server_urls_prefer_domain_specific_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=prod",
                "MCP_SERVER_URL_DEV=https://dotenv.dev/mcp",
                "MCP_SERVER_URL_PROD=https://dotenv.prod/mcp",
                "MCP_SERVER_URL_STRATEGY_PROD=https://mcp.prod/strategy/mcp",
                "MCP_SERVER_URL_BACKTEST_PROD=https://mcp.prod/backtest/mcp",
                "MCP_SERVER_URL_MARKET_DATA_PROD=https://mcp.prod/market/mcp",
                "MCP_SERVER_URL_STRESS_PROD=https://mcp.prod/stress/mcp",
                "MCP_SERVER_URL_TRADING_PROD=https://mcp.prod/trading/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.strategy_mcp_server_url == "https://mcp.prod/strategy/mcp"
    assert settings.backtest_mcp_server_url == "https://mcp.prod/backtest/mcp"
    assert settings.market_data_mcp_server_url == "https://mcp.prod/market/mcp"
    assert settings.stress_mcp_server_url == "https://mcp.prod/stress/mcp"
    assert settings.trading_mcp_server_url == "https://mcp.prod/trading/mcp"
    # Legacy property still works and remains separate from domain-specific URLs.
    assert settings.mcp_server_url == "https://dotenv.prod/mcp"


def test_domain_mcp_server_urls_auto_derive_for_minsyai_legacy_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=prod",
                "MCP_SERVER_URL_PROD=https://mcp.minsyai.com/mcp",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.strategy_mcp_server_url == "https://mcp.minsyai.com/strategy/mcp"
    assert settings.backtest_mcp_server_url == "https://mcp.minsyai.com/backtest/mcp"
    assert settings.market_data_mcp_server_url == "https://mcp.minsyai.com/market/mcp"
    assert settings.stress_mcp_server_url == "https://mcp.minsyai.com/stress/mcp"
    assert settings.trading_mcp_server_url == "https://mcp.minsyai.com/trading/mcp"


def test_runtime_env_aliases_and_dev_mode_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=development",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.runtime_env == "dev"
    assert settings.is_dev_mode is True

    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "APP_ENV=production",
            ]
        ),
        encoding="utf-8",
    )
    prod_settings = Settings(_env_file=env_file)
    assert prod_settings.runtime_env == "prod"
    assert prod_settings.is_dev_mode is False


def test_celery_worker_max_memory_per_child_reads_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "CELERY_WORKER_MAX_MEMORY_PER_CHILD=262144",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.celery_worker_max_memory_per_child == 262144


def test_celery_worker_max_memory_per_child_invalid_raises_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=test-secret-key-for-tests",
                "CELERY_WORKER_MAX_MEMORY_PER_CHILD=0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_effective_mcp_context_secret_falls_back_to_secret_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=base-secret",
                "MCP_CONTEXT_SECRET=",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.effective_mcp_context_secret == "base-secret"


def test_effective_mcp_context_secret_prefers_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "SECRET_KEY=base-secret",
                "MCP_CONTEXT_SECRET=mcp-secret",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)
    assert settings.effective_mcp_context_secret == "mcp-secret"
