"""Application configuration loaded from layered env files via Pydantic settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_ALLOWED_ENV_PROFILES: frozenset[str] = frozenset({"dev", "prod"})
_ALLOWED_AUTH_MODES: frozenset[str] = frozenset({"legacy", "hybrid", "clerk"})
_ALLOWED_INCREMENTAL_EXECUTION_MODES: frozenset[str] = frozenset(
    {"disabled", "local_collector", "remote_importer"}
)


def _normalize_env_profile(raw: str | None) -> str:
    profile = (raw or "").strip().lower()
    if not profile:
        return "dev"
    if profile not in _ALLOWED_ENV_PROFILES:
        allowed = ", ".join(sorted(_ALLOWED_ENV_PROFILES))
        raise ValueError(
            f"Unsupported environment profile '{profile}'. Use one of: {allowed}."
        )
    return profile


class Settings(BaseSettings):
    """Runtime settings for API, infrastructure and middleware."""

    app_name: str = Field(default="Minsy", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    api_public_base_url: str = Field(default="", alias="API_PUBLIC_BASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    sentry_env: str | None = Field(default=None, alias="SENTRY_ENV")
    sentry_release: str | None = Field(default=None, alias="SENTRY_RELEASE")
    sentry_traces_sample_rate: float = Field(
        default=0.0, alias="SENTRY_TRACES_SAMPLE_RATE"
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.0,
        alias="SENTRY_PROFILES_SAMPLE_RATE",
    )
    sentry_http_status_capture_enabled: bool = Field(
        default=True,
        alias="SENTRY_HTTP_STATUS_CAPTURE_ENABLED",
    )
    sentry_http_status_min_code: int = Field(
        default=400,
        alias="SENTRY_HTTP_STATUS_MIN_CODE",
    )
    sentry_http_status_max_code: int = Field(
        default=599,
        alias="SENTRY_HTTP_STATUS_MAX_CODE",
    )
    sentry_http_status_exclude_paths: list[str] = Field(
        default_factory=lambda: [
            "/api/v1/health",
            "/api/v1/status",
        ],
        alias="SENTRY_HTTP_STATUS_EXCLUDE_PATHS",
    )
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
    openai_cost_tracking_enabled: bool = Field(
        default=True,
        alias="OPENAI_COST_TRACKING_ENABLED",
    )
    billing_usage_persist_silent_failure_limit: int = Field(
        default=3,
        alias="BILLING_USAGE_PERSIST_SILENT_FAILURE_LIMIT",
    )
    billing_usage_persist_silent_window_seconds: int = Field(
        default=600,
        alias="BILLING_USAGE_PERSIST_SILENT_WINDOW_SECONDS",
    )
    openai_pricing_json: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        alias="OPENAI_PRICING_JSON",
    )
    openai_allowed_models_json: list[str] = Field(
        default_factory=lambda: ["gpt-5.2", "gpt-5.4"],
        alias="OPENAI_ALLOWED_MODELS_JSON",
    )
    stripe_publishable_key: str = Field(default="", alias="STRIPE_PUBLISHABLE_KEY")
    stripe_secret_key: str = Field(default="", alias="STRIPE_SECRET_KEY")
    stripe_webhook_secret: str = Field(default="", alias="STRIPE_WEBHOOK_SECRET")
    stripe_price_go_monthly: str = Field(
        default="price_1T7ek0RruOLLmTefTzIHF7Ge",
        alias="STRIPE_PRICE_GO_MONTHLY",
    )
    stripe_price_plus_monthly: str = Field(
        default="", alias="STRIPE_PRICE_PLUS_MONTHLY"
    )
    stripe_price_pro_monthly: str = Field(default="", alias="STRIPE_PRICE_PRO_MONTHLY")
    stripe_price_go_yearly: str = Field(default="", alias="STRIPE_PRICE_GO_YEARLY")
    stripe_price_plus_yearly: str = Field(default="", alias="STRIPE_PRICE_PLUS_YEARLY")
    stripe_price_pro_yearly: str = Field(default="", alias="STRIPE_PRICE_PRO_YEARLY")
    stripe_product_go: str = Field(
        default="prod_U5qQpo2ztIhimg",
        alias="STRIPE_PRODUCT_GO",
    )
    stripe_promotion_code_plus_pro: str = Field(
        default="",
        alias="STRIPE_PROMOTION_CODE_PLUS_PRO",
    )
    billing_checkout_success_url: str = Field(
        default="",
        alias="BILLING_CHECKOUT_SUCCESS_URL",
    )
    billing_checkout_cancel_url: str = Field(
        default="",
        alias="BILLING_CHECKOUT_CANCEL_URL",
    )
    billing_portal_return_url: str = Field(
        default="",
        alias="BILLING_PORTAL_RETURN_URL",
    )
    billing_frontend_base_url: str = Field(
        default="",
        alias="BILLING_FRONTEND_BASE_URL",
    )
    billing_pricing_json: dict[str, Any] = Field(
        default_factory=dict,
        alias="BILLING_PRICING_JSON",
    )
    # Legacy split config fields (kept as backward-compatible fallback).
    billing_config_json: dict[str, Any] = Field(
        default_factory=dict,
        alias="BILLING_CONFIG_JSON",
    )
    billing_cost_model_json: dict[str, Any] = Field(
        default_factory=dict,
        alias="BILLING_COST_MODEL_JSON",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=1440,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    auth_rate_limit: int = Field(default=30, alias="AUTH_RATE_LIMIT")
    auth_rate_window: int = Field(default=60, alias="AUTH_RATE_WINDOW")
    auth_mode: str = Field(default="legacy", alias="AUTH_MODE")
    clerk_frontend_api_url: str = Field(default="", alias="CLERK_FRONTEND_API_URL")
    clerk_backend_api_url: str = Field(
        default="https://api.clerk.com",
        alias="CLERK_BACKEND_API_URL",
    )
    clerk_jwks_url: str = Field(default="", alias="CLERK_JWKS_URL")
    clerk_jwt_key: str = Field(default="", alias="CLERK_JWT_KEY")
    clerk_publishable_key: str = Field(default="", alias="CLERK_PUBLISHABLE_KEY")
    clerk_secret_key: str = Field(default="", alias="CLERK_SECRET_KEY")
    clerk_api_version: str = Field(default="", alias="CLERK_API_VERSION")
    clerk_authorized_parties: list[str] = Field(
        default_factory=list,
        alias="CLERK_AUTHORIZED_PARTIES",
    )
    openai_response_model: str = Field(default="gpt-5.2", alias="OPENAI_RESPONSE_MODEL")
    openai_chain_reset_turns_per_phase: int = Field(
        default=8,
        alias="OPENAI_CHAIN_RESET_TURNS_PER_PHASE",
    )
    openai_prompt_input_max_chars: int = Field(
        default=32_000,
        alias="OPENAI_PROMPT_INPUT_MAX_CHARS",
    )
    mcp_server_url_strategy_dev: str = Field(
        default="https://dev.minsyai.com/strategy/mcp",
        alias="MCP_SERVER_URL_STRATEGY_DEV",
    )
    mcp_server_url_strategy_prod: str = Field(
        default="https://mcp.minsyai.com/strategy/mcp",
        alias="MCP_SERVER_URL_STRATEGY_PROD",
    )
    mcp_server_url_backtest_dev: str = Field(
        default="https://dev.minsyai.com/backtest/mcp",
        alias="MCP_SERVER_URL_BACKTEST_DEV",
    )
    mcp_server_url_backtest_prod: str = Field(
        default="https://mcp.minsyai.com/backtest/mcp",
        alias="MCP_SERVER_URL_BACKTEST_PROD",
    )
    mcp_server_url_market_data_dev: str = Field(
        default="https://dev.minsyai.com/market/mcp",
        alias="MCP_SERVER_URL_MARKET_DATA_DEV",
    )
    mcp_server_url_market_data_prod: str = Field(
        default="https://mcp.minsyai.com/market/mcp",
        alias="MCP_SERVER_URL_MARKET_DATA_PROD",
    )
    mcp_server_url_stress_dev: str = Field(
        default="https://dev.minsyai.com/stress/mcp",
        alias="MCP_SERVER_URL_STRESS_DEV",
    )
    mcp_server_url_stress_prod: str = Field(
        default="https://mcp.minsyai.com/stress/mcp",
        alias="MCP_SERVER_URL_STRESS_PROD",
    )
    mcp_server_url_trading_dev: str = Field(
        default="https://dev.minsyai.com/trading/mcp",
        alias="MCP_SERVER_URL_TRADING_DEV",
    )
    mcp_server_url_trading_prod: str = Field(
        default="https://mcp.minsyai.com/trading/mcp",
        alias="MCP_SERVER_URL_TRADING_PROD",
    )
    mcp_context_secret: str | None = Field(default=None, alias="MCP_CONTEXT_SECRET")
    mcp_context_ttl_seconds: int = Field(default=300, alias="MCP_CONTEXT_TTL_SECONDS")
    telegram_enabled: bool = Field(default=True, alias="TELEGRAM_ENABLED")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_bot_username: str = Field(default="", alias="TELEGRAM_BOT_USERNAME")
    telegram_webhook_secret_token: str = Field(
        default="", alias="TELEGRAM_WEBHOOK_SECRET_TOKEN"
    )
    telegram_webhook_url: str = Field(default="", alias="TELEGRAM_WEBHOOK_URL")
    telegram_webhook_auto_sync_enabled: bool = Field(
        default=False,
        alias="TELEGRAM_WEBHOOK_AUTO_SYNC_ENABLED",
    )
    telegram_connect_ttl_seconds: int = Field(
        default=600, alias="TELEGRAM_CONNECT_TTL_SECONDS"
    )
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
    telegram_test_target_email: str = Field(
        default="1@test.com",
        alias="TELEGRAM_TEST_TARGET_EMAIL",
    )
    telegram_test_force_target_email_enabled: bool = Field(
        default=True,
        alias="TELEGRAM_TEST_FORCE_TARGET_EMAIL_ENABLED",
    )
    telegram_test_target_require_connected: bool = Field(
        default=True,
        alias="TELEGRAM_TEST_TARGET_REQUIRE_CONNECTED",
    )
    telegram_test_expected_chat_id: str = Field(
        default="",
        alias="TELEGRAM_TEST_EXPECTED_CHAT_ID",
    )

    notifications_enabled: bool = Field(default=True, alias="NOTIFICATIONS_ENABLED")
    notifications_loop_interval_seconds: float = Field(
        default=5.0,
        alias="NOTIFICATIONS_LOOP_INTERVAL_SECONDS",
    )
    notifications_dispatch_batch_size: int = Field(
        default=10,
        alias="NOTIFICATIONS_DISPATCH_BATCH_SIZE",
    )
    notifications_delivery_timeout_seconds: float = Field(
        default=6.0,
        alias="NOTIFICATIONS_DELIVERY_TIMEOUT_SECONDS",
    )
    notifications_dispatch_max_runtime_seconds: float = Field(
        default=8.0,
        alias="NOTIFICATIONS_DISPATCH_MAX_RUNTIME_SECONDS",
    )
    notifications_dispatch_lock_ttl_seconds: float = Field(
        default=30.0,
        alias="NOTIFICATIONS_DISPATCH_LOCK_TTL_SECONDS",
    )
    notifications_retry_max_attempts: int = Field(
        default=3,
        alias="NOTIFICATIONS_RETRY_MAX_ATTEMPTS",
    )
    notifications_retry_backoff_seconds: float = Field(
        default=5.0,
        alias="NOTIFICATIONS_RETRY_BACKOFF_SECONDS",
    )
    trading_approval_enabled: bool = Field(
        default=True, alias="TRADING_APPROVAL_ENABLED"
    )
    trading_approval_expire_scan_interval_seconds: float = Field(
        default=10.0,
        alias="TRADING_APPROVAL_EXPIRE_SCAN_INTERVAL_SECONDS",
    )
    telegram_approval_callback_secret: str = Field(
        default="",
        alias="TELEGRAM_APPROVAL_CALLBACK_SECRET",
    )

    # Trading runtime controls
    paper_trading_enabled: bool = Field(default=True, alias="PAPER_TRADING_ENABLED")
    paper_trading_manual_action_queue: str = Field(
        default="paper_trading_manual",
        alias="PAPER_TRADING_MANUAL_ACTION_QUEUE",
    )
    paper_trading_enqueue_on_start: bool = Field(
        default=True,
        alias="PAPER_TRADING_ENQUEUE_ON_START",
    )
    paper_trading_execute_orders: bool = Field(
        default=True,
        alias="PAPER_TRADING_EXECUTE_ORDERS",
    )
    paper_trading_default_position_pct: float = Field(
        default=0.025,
        alias="PAPER_TRADING_DEFAULT_POSITION_PCT",
    )
    paper_trading_loop_interval_seconds: float = Field(
        default=1.0,
        alias="PAPER_TRADING_LOOP_INTERVAL_SECONDS",
    )
    paper_trading_starting_retry_seconds: float = Field(
        default=10.0,
        alias="PAPER_TRADING_STARTING_RETRY_SECONDS",
    )
    paper_trading_broker_account_sync_interval_seconds: float = Field(
        default=20.0,
        alias="PAPER_TRADING_BROKER_ACCOUNT_SYNC_INTERVAL_SECONDS",
    )
    paper_trading_runtime_task_expires_seconds: float = Field(
        default=120.0,
        alias="PAPER_TRADING_RUNTIME_TASK_EXPIRES_SECONDS",
    )
    paper_trading_queue_backlog_soft_limit: int = Field(
        default=2000,
        alias="PAPER_TRADING_QUEUE_BACKLOG_SOFT_LIMIT",
    )
    paper_trading_scheduler_max_enqueues_per_tick: int = Field(
        default=500,
        alias="PAPER_TRADING_SCHEDULER_MAX_ENQUEUES_PER_TICK",
    )
    paper_trading_max_retries: int = Field(default=3, alias="PAPER_TRADING_MAX_RETRIES")
    paper_trading_kill_switch_global: bool = Field(
        default=False,
        alias="PAPER_TRADING_KILL_SWITCH_GLOBAL",
    )
    paper_trading_kill_switch_users_csv: str = Field(
        default="",
        alias="PAPER_TRADING_KILL_SWITCH_USERS",
    )
    paper_trading_kill_switch_deployments_csv: str = Field(
        default="",
        alias="PAPER_TRADING_KILL_SWITCH_DEPLOYMENTS",
    )
    paper_trading_broker_retry_max_attempts: int = Field(
        default=3,
        alias="PAPER_TRADING_BROKER_RETRY_MAX_ATTEMPTS",
    )
    paper_trading_broker_retry_backoff_seconds: float = Field(
        default=0.2,
        alias="PAPER_TRADING_BROKER_RETRY_BACKOFF_SECONDS",
    )
    paper_trading_circuit_breaker_failure_threshold: int = Field(
        default=5,
        alias="PAPER_TRADING_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    )
    paper_trading_circuit_breaker_recovery_seconds: float = Field(
        default=30.0,
        alias="PAPER_TRADING_CIRCUIT_BREAKER_RECOVERY_SECONDS",
    )
    paper_trading_runtime_health_stale_seconds: int = Field(
        default=120,
        alias="PAPER_TRADING_RUNTIME_HEALTH_STALE_SECONDS",
    )
    paper_trading_deployment_lock_ttl_seconds: float = Field(
        default=20.0,
        alias="PAPER_TRADING_DEPLOYMENT_LOCK_TTL_SECONDS",
    )
    backtest_max_bars: int = Field(
        default=3_000_000,
        alias="BACKTEST_MAX_BARS",
    )
    backtest_stale_job_cleanup_enabled: bool = Field(
        default=True,
        alias="BACKTEST_STALE_JOB_CLEANUP_ENABLED",
    )
    backtest_stale_job_cleanup_interval_minutes: int = Field(
        default=10,
        alias="BACKTEST_STALE_JOB_CLEANUP_INTERVAL_MINUTES",
    )
    backtest_running_stale_minutes: int = Field(
        default=30,
        alias="BACKTEST_RUNNING_STALE_MINUTES",
    )
    backtest_result_max_trades: int = Field(
        default=20000,
        alias="BACKTEST_RESULT_MAX_TRADES",
    )
    backtest_result_max_equity_points: int = Field(
        default=5000,
        alias="BACKTEST_RESULT_MAX_EQUITY_POINTS",
    )
    backtest_result_max_returns: int = Field(
        default=5000,
        alias="BACKTEST_RESULT_MAX_RETURNS",
    )
    backtest_result_max_events: int = Field(
        default=2000,
        alias="BACKTEST_RESULT_MAX_EVENTS",
    )
    stress_default_seed: int = Field(
        default=42,
        alias="STRESS_DEFAULT_SEED",
    )
    stress_monte_carlo_default_trials: int = Field(
        default=2000,
        alias="STRESS_MONTE_CARLO_DEFAULT_TRIALS",
    )
    stress_monte_carlo_default_horizon_bars: int = Field(
        default=252,
        alias="STRESS_MONTE_CARLO_DEFAULT_HORIZON_BARS",
    )
    stress_monte_carlo_default_method: str = Field(
        default="block_bootstrap",
        alias="STRESS_MONTE_CARLO_DEFAULT_METHOD",
    )
    optimization_default_budget: int = Field(
        default=40,
        alias="OPTIMIZATION_DEFAULT_BUDGET",
    )
    trading_credentials_secret: str | None = Field(
        default=None,
        alias="TRADING_CREDENTIALS_SECRET",
    )

    # Alpaca (paper trading first; live kept as future toggle via base URL)
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    alpaca_trading_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_TRADING_BASE_URL",
    )
    alpaca_paper_trading_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_PAPER_TRADING_BASE_URL",
    )
    alpaca_live_trading_base_url: str = Field(
        default="https://api.alpaca.markets",
        alias="ALPACA_LIVE_TRADING_BASE_URL",
    )
    alpaca_market_data_base_url: str = Field(
        default="https://data.alpaca.markets",
        alias="ALPACA_MARKET_DATA_BASE_URL",
    )
    alpaca_stocks_feed: str = Field(default="iex", alias="ALPACA_STOCKS_FEED")
    alpaca_crypto_feed: str = Field(default="us", alias="ALPACA_CRYPTO_FEED")
    alpaca_market_data_stream_url: str = Field(
        default="wss://stream.data.alpaca.markets/v2",
        alias="ALPACA_MARKET_DATA_STREAM_URL",
    )
    alpaca_request_rate_limit_per_minute: int = Field(
        default=200,
        alias="ALPACA_REQUEST_RATE_LIMIT_PER_MINUTE",
    )
    alpaca_stream_reconnect_base_seconds: float = Field(
        default=1.0,
        alias="ALPACA_STREAM_RECONNECT_BASE_SECONDS",
    )
    alpaca_stream_max_retries: int = Field(
        default=10, alias="ALPACA_STREAM_MAX_RETRIES"
    )
    alpaca_account_probe_timeout_seconds: float = Field(
        default=8.0,
        alias="ALPACA_ACCOUNT_PROBE_TIMEOUT_SECONDS",
    )
    market_data_backfill_limit: int = Field(
        default=500, alias="MARKET_DATA_BACKFILL_LIMIT"
    )
    market_data_refresh_active_subscriptions_interval_seconds: float = Field(
        default=15.0,
        alias="MARKET_DATA_REFRESH_ACTIVE_SUBSCRIPTIONS_INTERVAL_SECONDS",
    )
    market_data_aggregate_timeframes_csv: str = Field(
        default="5m,15m,1h,4h,1d",
        alias="MARKET_DATA_AGGREGATE_TIMEFRAMES",
    )
    market_data_aggregate_timezone: str = Field(
        default="UTC",
        alias="MARKET_DATA_AGGREGATE_TIMEZONE",
    )
    market_data_history_target_bars: int = Field(
        default=2500,
        alias="MARKET_DATA_HISTORY_TARGET_BARS",
    )
    market_data_history_warmup_chunk_bars: int = Field(
        default=500,
        alias="MARKET_DATA_HISTORY_WARMUP_CHUNK_BARS",
    )
    market_data_ring_capacity_1m: int = Field(
        default=45000,
        alias="MARKET_DATA_RING_CAPACITY_1M",
    )
    market_data_ring_capacity_aggregated: int = Field(
        default=10000,
        alias="MARKET_DATA_RING_CAPACITY_AGG",
    )
    market_data_factor_cache_max_entries: int = Field(
        default=200000,
        alias="MARKET_DATA_FACTOR_CACHE_MAX_ENTRIES",
    )
    market_data_checkpoint_ttl_seconds: int = Field(
        default=86400,
        alias="MARKET_DATA_CHECKPOINT_TTL_SECONDS",
    )
    market_data_redis_write_enabled: bool | None = Field(
        default=None,
        alias="MARKET_DATA_REDIS_WRITE_ENABLED",
    )
    market_data_redis_read_enabled: bool | None = Field(
        default=None,
        alias="MARKET_DATA_REDIS_READ_ENABLED",
    )
    market_data_redis_subs_enabled: bool | None = Field(
        default=None,
        alias="MARKET_DATA_REDIS_SUBS_ENABLED",
    )
    market_data_memory_cache_enabled: bool = Field(
        default=True,
        alias="MARKET_DATA_MEMORY_CACHE_ENABLED",
    )
    market_data_runtime_fail_fast_on_redis_error: bool | None = Field(
        default=None,
        alias="MARKET_DATA_RUNTIME_FAIL_FAST_ON_REDIS_ERROR",
    )
    trading_ws_pubsub_wait_seconds: float = Field(
        default=1.0,
        alias="TRADING_WS_PUBSUB_WAIT_SECONDS",
    )
    trading_ws_fallback_poll_seconds: float = Field(
        default=0.5,
        alias="TRADING_WS_FALLBACK_POLL_SECONDS",
    )
    trading_ws_reconcile_seconds: float = Field(
        default=2.0,
        alias="TRADING_WS_RECONCILE_SECONDS",
    )
    trading_ws_pubsub_probe_base_seconds: float = Field(
        default=1.0,
        alias="TRADING_WS_PUBSUB_PROBE_BASE_SECONDS",
    )
    trading_ws_pubsub_probe_max_seconds: float = Field(
        default=15.0,
        alias="TRADING_WS_PUBSUB_PROBE_MAX_SECONDS",
    )
    market_data_refresh_dedupe_enabled: bool = Field(
        default=True,
        alias="MARKET_DATA_REFRESH_DEDUPE_ENABLED",
    )
    market_data_refresh_dedupe_window_seconds: int = Field(
        default=20,
        alias="MARKET_DATA_REFRESH_DEDUPE_WINDOW_SECONDS",
    )
    market_data_refresh_symbol_rate_limit: str = Field(
        default="",
        alias="MARKET_DATA_REFRESH_SYMBOL_RATE_LIMIT",
    )
    market_data_sync_batch_limit: int = Field(
        default=5000,
        alias="MARKET_DATA_SYNC_BATCH_LIMIT",
    )
    market_data_sync_max_ranges: int = Field(
        default=512,
        alias="MARKET_DATA_SYNC_MAX_RANGES",
    )
    market_data_sync_default_lookback_days: int = Field(
        default=30,
        alias="MARKET_DATA_SYNC_DEFAULT_LOOKBACK_DAYS",
    )
    market_data_sync_lock_ttl_seconds: int = Field(
        default=1800,
        alias="MARKET_DATA_SYNC_LOCK_TTL_SECONDS",
    )
    market_data_sync_request_start_interval_seconds: float = Field(
        default=0.15,
        alias="MARKET_DATA_SYNC_REQUEST_START_INTERVAL_SECONDS",
    )
    market_data_incremental_execution_mode: str = Field(
        default="disabled",
        alias="MARKET_DATA_INCREMENTAL_EXECUTION_MODE",
    )
    market_data_incremental_sync_enabled: bool = Field(
        default=False,
        alias="MARKET_DATA_INCREMENTAL_SYNC_ENABLED",
    )
    market_data_incremental_sync_cron_hour: int = Field(
        default=4,
        alias="MARKET_DATA_INCREMENTAL_SYNC_CRON_HOUR",
    )
    market_data_incremental_sync_cron_minute: int = Field(
        default=30,
        alias="MARKET_DATA_INCREMENTAL_SYNC_CRON_MINUTE",
    )
    market_data_incremental_receiver_enabled: bool = Field(
        default=True,
        alias="MARKET_DATA_INCREMENTAL_RECEIVER_ENABLED",
    )
    market_data_incremental_remote_base_url: str = Field(
        default="",
        alias="MARKET_DATA_INCREMENTAL_REMOTE_BASE_URL",
    )
    market_data_incremental_remote_timeout_seconds: float = Field(
        default=20.0,
        alias="MARKET_DATA_INCREMENTAL_REMOTE_TIMEOUT_SECONDS",
    )
    market_data_incremental_hmac_key_id: str = Field(
        default="local-sync",
        alias="MARKET_DATA_INCREMENTAL_HMAC_KEY_ID",
    )
    market_data_incremental_hmac_secret: str = Field(
        default="",
        alias="MARKET_DATA_INCREMENTAL_HMAC_SECRET",
    )
    market_data_incremental_hmac_max_skew_seconds: int = Field(
        default=300,
        alias="MARKET_DATA_INCREMENTAL_HMAC_MAX_SKEW_SECONDS",
    )
    market_data_incremental_gcs_bucket: str = Field(
        default="",
        alias="MARKET_DATA_INCREMENTAL_GCS_BUCKET",
    )
    market_data_incremental_gcs_prefix: str = Field(
        default="incremental",
        alias="MARKET_DATA_INCREMENTAL_GCS_PREFIX",
    )
    market_data_incremental_safety_lag_minutes: int = Field(
        default=2,
        alias="MARKET_DATA_INCREMENTAL_SAFETY_LAG_MINUTES",
    )
    google_application_credentials: str = Field(
        default="",
        alias="GOOGLE_APPLICATION_CREDENTIALS",
    )
    google_cloud_project: str = Field(
        default="",
        alias="GOOGLE_CLOUD_PROJECT",
    )
    gcp_project: str = Field(
        default="",
        alias="GCP_PROJECT",
    )
    ibkr_gateway_host: str = Field(
        default="host.docker.internal",
        alias="IBKR_GATEWAY_HOST",
    )
    ibkr_gateway_port: int = Field(
        default=4002,
        alias="IBKR_GATEWAY_PORT",
    )
    ibkr_gateway_client_id: int = Field(
        default=410,
        alias="IBKR_GATEWAY_CLIENT_ID",
    )
    ibkr_gateway_timeout_seconds: float = Field(
        default=10.0,
        alias="IBKR_GATEWAY_TIMEOUT_SECONDS",
    )
    ibkr_forex_what_to_show: str = Field(
        default="MIDPOINT",
        alias="IBKR_FOREX_WHAT_TO_SHOW",
    )
    ibkr_futures_what_to_show: str = Field(
        default="TRADES",
        alias="IBKR_FUTURES_WHAT_TO_SHOW",
    )
    ibkr_futures_exchange_map_json: dict[str, str] = Field(
        default_factory=dict,
        alias="IBKR_FUTURES_EXCHANGE_MAP_JSON",
    )
    pre_strategy_regime_lookback_bars: int = Field(
        default=500,
        alias="PRE_STRATEGY_REGIME_LOOKBACK_BARS",
    )
    pre_strategy_regime_min_bars: int = Field(
        default=150,
        alias="PRE_STRATEGY_REGIME_MIN_BARS",
    )
    pre_strategy_regime_cache_ttl_seconds: int = Field(
        default=600,
        alias="PRE_STRATEGY_REGIME_CACHE_TTL_SECONDS",
    )
    pre_strategy_regime_pivot_window: int = Field(
        default=5,
        alias="PRE_STRATEGY_REGIME_PIVOT_WINDOW",
    )
    pre_strategy_regime_image_max_bars: int = Field(
        default=240,
        alias="PRE_STRATEGY_REGIME_IMAGE_MAX_BARS",
    )
    ccxt_market_data_enabled: bool = Field(
        default=True,
        alias="CCXT_MARKET_DATA_ENABLED",
    )
    ccxt_market_data_exchange_id: str = Field(
        default="binance",
        alias="CCXT_MARKET_DATA_EXCHANGE_ID",
    )
    ccxt_market_data_timeout_seconds: float = Field(
        default=8.0,
        alias="CCXT_MARKET_DATA_TIMEOUT_SECONDS",
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
    celery_result_backend: str | None = Field(
        default=None, alias="CELERY_RESULT_BACKEND"
    )
    celery_task_default_queue: str = Field(
        default="backtest", alias="CELERY_TASK_DEFAULT_QUEUE"
    )
    celery_task_time_limit_seconds: int = Field(
        default=1800,
        alias="CELERY_TASK_TIME_LIMIT_SECONDS",
    )
    celery_task_soft_time_limit_seconds: int = Field(
        default=1740,
        alias="CELERY_TASK_SOFT_TIME_LIMIT_SECONDS",
    )
    celery_worker_max_memory_per_child: int = Field(
        default=524288,
        alias="CELERY_WORKER_MAX_MEMORY_PER_CHILD",
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=1,
        alias="CELERY_WORKER_PREFETCH_MULTIPLIER",
    )
    celery_task_acks_late: bool = Field(default=True, alias="CELERY_TASK_ACKS_LATE")
    celery_task_always_eager: bool = Field(
        default=False, alias="CELERY_TASK_ALWAYS_EAGER"
    )
    celery_timezone: str = Field(default="UTC", alias="CELERY_TIMEZONE")
    flower_enabled: bool = Field(default=False, alias="FLOWER_ENABLED")
    flower_host: str = Field(default="127.0.0.1", alias="FLOWER_HOST")
    flower_port: int = Field(default=5555, alias="FLOWER_PORT")
    flower_user: str = Field(default="", alias="FLOWER_USER")
    flower_password: str = Field(default="", alias="FLOWER_PASSWORD")

    postgres_backup_enabled: bool = Field(default=True, alias="POSTGRES_BACKUP_ENABLED")
    postgres_backup_dir: str = Field(
        default="backups/postgres", alias="POSTGRES_BACKUP_DIR"
    )
    postgres_backup_retention_count: int = Field(
        default=14,
        alias="POSTGRES_BACKUP_RETENTION_COUNT",
    )
    postgres_backup_hour_utc: int = Field(default=3, alias="POSTGRES_BACKUP_HOUR_UTC")
    postgres_backup_minute_utc: int = Field(
        default=0, alias="POSTGRES_BACKUP_MINUTE_UTC"
    )
    postgres_pg_dump_bin: str = Field(default="pg_dump", alias="POSTGRES_PG_DUMP_BIN")

    user_email_csv_export_enabled: bool = Field(
        default=True,
        alias="USER_EMAIL_CSV_EXPORT_ENABLED",
    )
    user_email_csv_path: str = Field(
        default="exports/user_emails.csv", alias="USER_EMAIL_CSV_PATH"
    )
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
        env_file=None,
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
        """Use resolved env files as primary source over process environment variables."""
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    @field_validator("app_env", mode="before")
    @classmethod
    def _validate_app_env(cls, value: object) -> str:
        if isinstance(value, str):
            return _normalize_env_profile(value)
        if value is None:
            return "dev"
        return _normalize_env_profile(str(value))

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

    @field_validator("sentry_http_status_exclude_paths", mode="before")
    @classmethod
    def _parse_sentry_http_status_exclude_paths(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            if normalized.startswith("["):
                return json.loads(normalized)
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return value

    @field_validator("openai_pricing_json", mode="before")
    @classmethod
    def _parse_openai_pricing_json(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return {}
            return json.loads(normalized)
        return {}

    @field_validator("openai_allowed_models_json", mode="before")
    @classmethod
    def _parse_openai_allowed_models_json(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            if normalized.startswith("["):
                return json.loads(normalized)
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return []

    @field_validator("auth_mode", mode="before")
    @classmethod
    def _validate_auth_mode(cls, value: object) -> str:
        if value is None:
            return "legacy"
        normalized = str(value).strip().lower()
        if not normalized:
            return "legacy"
        if normalized not in _ALLOWED_AUTH_MODES:
            allowed = ", ".join(sorted(_ALLOWED_AUTH_MODES))
            raise ValueError(
                f"Unsupported AUTH_MODE '{normalized}'. Use one of: {allowed}."
            )
        return normalized

    @field_validator("clerk_authorized_parties", mode="before")
    @classmethod
    def _parse_clerk_authorized_parties(cls, value: object) -> object:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return []
            if normalized.startswith("["):
                return json.loads(normalized)
            return [item.strip() for item in normalized.split(",") if item.strip()]
        return []

    @field_validator(
        "billing_pricing_json",
        "billing_config_json",
        "billing_cost_model_json",
        mode="before",
    )
    @classmethod
    def _parse_billing_json_config(
        cls,
        value: object,
    ) -> object:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return {}
            return json.loads(normalized)
        return {}

    @field_validator("sentry_traces_sample_rate", "sentry_profiles_sample_rate")
    @classmethod
    def _validate_sentry_sample_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Sentry sample rates must be between 0.0 and 1.0.")
        return value

    @field_validator("sentry_http_status_min_code", "sentry_http_status_max_code")
    @classmethod
    def _validate_sentry_http_status_code(cls, value: int) -> int:
        if not 100 <= value <= 599:
            raise ValueError(
                "Sentry HTTP status capture codes must be between 100 and 599."
            )
        return value

    @model_validator(mode="after")
    def _validate_sentry_http_status_code_range(self) -> Settings:
        if self.sentry_http_status_min_code > self.sentry_http_status_max_code:
            raise ValueError(
                "SENTRY_HTTP_STATUS_MIN_CODE must be <= SENTRY_HTTP_STATUS_MAX_CODE."
            )
        return self

    @model_validator(mode="after")
    def _validate_clerk_auth_configuration(self) -> Settings:
        if self.auth_mode == "legacy":
            return self

        required: dict[str, str] = {
            "CLERK_FRONTEND_API_URL": self.clerk_frontend_api_url,
            "CLERK_PUBLISHABLE_KEY": self.clerk_publishable_key,
            "CLERK_SECRET_KEY": self.clerk_secret_key,
            "CLERK_JWKS_URL": self.clerk_jwks_url,
            "CLERK_JWT_KEY": self.clerk_jwt_key,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(
                f"Missing Clerk configuration for AUTH_MODE={self.auth_mode}: "
                f"{', '.join(missing)}."
            )
        return self

    @field_validator("billing_usage_persist_silent_failure_limit")
    @classmethod
    def _validate_billing_usage_persist_silent_failure_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("BILLING_USAGE_PERSIST_SILENT_FAILURE_LIMIT must be >= 0.")
        return value

    @field_validator("billing_usage_persist_silent_window_seconds")
    @classmethod
    def _validate_billing_usage_persist_silent_window_seconds(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "BILLING_USAGE_PERSIST_SILENT_WINDOW_SECONDS must be >= 1."
            )
        return value

    @field_validator(
        "openai_chain_reset_turns_per_phase",
        "openai_prompt_input_max_chars",
    )
    @classmethod
    def _validate_positive_openai_limits(cls, value: int) -> int:
        if value < 1:
            raise ValueError("OpenAI prompt/chain limits must be >= 1.")
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

    @field_validator("flower_port")
    @classmethod
    def _validate_flower_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("FLOWER_PORT must be in [1, 65535].")
        return value

    @field_validator("celery_worker_max_memory_per_child")
    @classmethod
    def _validate_celery_worker_max_memory_per_child(cls, value: int) -> int:
        if value < 1:
            raise ValueError("CELERY_WORKER_MAX_MEMORY_PER_CHILD must be >= 1 (KB).")
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

    @field_validator("notifications_loop_interval_seconds")
    @classmethod
    def _validate_notifications_loop_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("NOTIFICATIONS_LOOP_INTERVAL_SECONDS must be > 0.")
        return value

    @field_validator("notifications_dispatch_batch_size")
    @classmethod
    def _validate_notifications_dispatch_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("NOTIFICATIONS_DISPATCH_BATCH_SIZE must be >= 1.")
        return value

    @field_validator("notifications_delivery_timeout_seconds")
    @classmethod
    def _validate_notifications_delivery_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("NOTIFICATIONS_DELIVERY_TIMEOUT_SECONDS must be > 0.")
        return value

    @field_validator("notifications_dispatch_max_runtime_seconds")
    @classmethod
    def _validate_notifications_dispatch_max_runtime_seconds(
        cls, value: float
    ) -> float:
        if value <= 0:
            raise ValueError("NOTIFICATIONS_DISPATCH_MAX_RUNTIME_SECONDS must be > 0.")
        return value

    @field_validator("notifications_dispatch_lock_ttl_seconds")
    @classmethod
    def _validate_notifications_dispatch_lock_ttl_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("NOTIFICATIONS_DISPATCH_LOCK_TTL_SECONDS must be > 0.")
        return value

    @field_validator("notifications_retry_max_attempts")
    @classmethod
    def _validate_notifications_retry_max_attempts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("NOTIFICATIONS_RETRY_MAX_ATTEMPTS must be >= 0.")
        return value

    @field_validator("notifications_retry_backoff_seconds")
    @classmethod
    def _validate_notifications_retry_backoff_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("NOTIFICATIONS_RETRY_BACKOFF_SECONDS must be > 0.")
        return value

    @field_validator("trading_approval_expire_scan_interval_seconds")
    @classmethod
    def _validate_trading_approval_expire_scan_interval_seconds(
        cls, value: float
    ) -> float:
        if value <= 0:
            raise ValueError(
                "TRADING_APPROVAL_EXPIRE_SCAN_INTERVAL_SECONDS must be > 0."
            )
        return value

    @field_validator("paper_trading_loop_interval_seconds")
    @classmethod
    def _validate_paper_trading_loop_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("PAPER_TRADING_LOOP_INTERVAL_SECONDS must be > 0.")
        return value

    @field_validator("paper_trading_default_position_pct")
    @classmethod
    def _validate_paper_trading_default_position_pct(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("PAPER_TRADING_DEFAULT_POSITION_PCT must be > 0 and <= 1.")
        return value

    @field_validator("paper_trading_starting_retry_seconds")
    @classmethod
    def _validate_paper_trading_starting_retry_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("PAPER_TRADING_STARTING_RETRY_SECONDS must be > 0.")
        return value

    @field_validator("paper_trading_broker_account_sync_interval_seconds")
    @classmethod
    def _validate_paper_trading_broker_account_sync_interval_seconds(
        cls, value: float
    ) -> float:
        if value <= 0:
            raise ValueError(
                "PAPER_TRADING_BROKER_ACCOUNT_SYNC_INTERVAL_SECONDS must be > 0."
            )
        return value

    @field_validator("paper_trading_runtime_task_expires_seconds")
    @classmethod
    def _validate_paper_trading_runtime_task_expires_seconds(
        cls, value: float
    ) -> float:
        if value <= 0:
            raise ValueError("PAPER_TRADING_RUNTIME_TASK_EXPIRES_SECONDS must be > 0.")
        return value

    @field_validator("paper_trading_queue_backlog_soft_limit")
    @classmethod
    def _validate_paper_trading_queue_backlog_soft_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("PAPER_TRADING_QUEUE_BACKLOG_SOFT_LIMIT must be >= 0.")
        return value

    @field_validator("paper_trading_scheduler_max_enqueues_per_tick")
    @classmethod
    def _validate_paper_trading_scheduler_max_enqueues_per_tick(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "PAPER_TRADING_SCHEDULER_MAX_ENQUEUES_PER_TICK must be >= 0."
            )
        return value

    @field_validator("paper_trading_deployment_lock_ttl_seconds")
    @classmethod
    def _validate_paper_trading_deployment_lock_ttl(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("PAPER_TRADING_DEPLOYMENT_LOCK_TTL_SECONDS must be > 0.")
        return value

    @field_validator("paper_trading_max_retries")
    @classmethod
    def _validate_paper_trading_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("PAPER_TRADING_MAX_RETRIES must be >= 0.")
        return value

    @field_validator("backtest_max_bars")
    @classmethod
    def _validate_backtest_max_bars(cls, value: int) -> int:
        if value < 1:
            raise ValueError("BACKTEST_MAX_BARS must be >= 1.")
        return value

    @field_validator(
        "backtest_stale_job_cleanup_interval_minutes",
        "backtest_running_stale_minutes",
    )
    @classmethod
    def _validate_backtest_cleanup_minutes(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Backtest stale-job cleanup minute settings must be >= 1.")
        return value

    @field_validator(
        "backtest_result_max_trades",
        "backtest_result_max_equity_points",
        "backtest_result_max_returns",
        "backtest_result_max_events",
    )
    @classmethod
    def _validate_backtest_result_caps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Backtest result cap settings must be >= 0.")
        return value

    @field_validator(
        "stress_default_seed",
        "stress_monte_carlo_default_trials",
        "stress_monte_carlo_default_horizon_bars",
        "optimization_default_budget",
    )
    @classmethod
    def _validate_stress_numeric_defaults(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Stress/optimization numeric defaults must be >= 1.")
        return value

    @field_validator("stress_monte_carlo_default_method")
    @classmethod
    def _validate_stress_monte_carlo_default_method(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"iid_bootstrap", "block_bootstrap", "trade_shuffle"}:
            raise ValueError(
                "STRESS_MONTE_CARLO_DEFAULT_METHOD must be one of "
                "{iid_bootstrap, block_bootstrap, trade_shuffle}."
            )
        return normalized

    @field_validator("alpaca_request_rate_limit_per_minute")
    @classmethod
    def _validate_alpaca_request_rate_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ALPACA_REQUEST_RATE_LIMIT_PER_MINUTE must be >= 1.")
        return value

    @field_validator("alpaca_stream_reconnect_base_seconds")
    @classmethod
    def _validate_stream_reconnect_base_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ALPACA_STREAM_RECONNECT_BASE_SECONDS must be > 0.")
        return value

    @field_validator("alpaca_stream_max_retries")
    @classmethod
    def _validate_stream_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ALPACA_STREAM_MAX_RETRIES must be >= 0.")
        return value

    @field_validator("alpaca_account_probe_timeout_seconds")
    @classmethod
    def _validate_alpaca_account_probe_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ALPACA_ACCOUNT_PROBE_TIMEOUT_SECONDS must be > 0.")
        return value

    @field_validator("market_data_backfill_limit")
    @classmethod
    def _validate_market_data_backfill_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MARKET_DATA_BACKFILL_LIMIT must be >= 1.")
        return value

    @field_validator("market_data_refresh_active_subscriptions_interval_seconds")
    @classmethod
    def _validate_market_data_refresh_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(
                "MARKET_DATA_REFRESH_ACTIVE_SUBSCRIPTIONS_INTERVAL_SECONDS must be > 0."
            )
        return value

    @field_validator("market_data_refresh_dedupe_window_seconds")
    @classmethod
    def _validate_market_data_refresh_dedupe_window_seconds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MARKET_DATA_REFRESH_DEDUPE_WINDOW_SECONDS must be >= 1.")
        return value

    @field_validator(
        "market_data_history_target_bars",
        "market_data_history_warmup_chunk_bars",
        "market_data_ring_capacity_1m",
        "market_data_ring_capacity_aggregated",
    )
    @classmethod
    def _validate_market_data_ring_capacity(cls, value: int) -> int:
        if value < 10:
            raise ValueError("MARKET_DATA ring capacity must be >= 10.")
        return value

    @field_validator("market_data_factor_cache_max_entries")
    @classmethod
    def _validate_market_data_factor_cache_max_entries(cls, value: int) -> int:
        if value < 100:
            raise ValueError("MARKET_DATA_FACTOR_CACHE_MAX_ENTRIES must be >= 100.")
        return value

    @field_validator("market_data_checkpoint_ttl_seconds")
    @classmethod
    def _validate_market_data_checkpoint_ttl_seconds(cls, value: int) -> int:
        if value < 60:
            raise ValueError("MARKET_DATA_CHECKPOINT_TTL_SECONDS must be >= 60.")
        return value

    @field_validator(
        "market_data_sync_batch_limit",
        "market_data_sync_max_ranges",
        "market_data_sync_default_lookback_days",
        "market_data_sync_lock_ttl_seconds",
    )
    @classmethod
    def _validate_market_data_sync_limits(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MARKET_DATA sync numeric settings must be >= 1.")
        return value

    @field_validator("market_data_sync_request_start_interval_seconds")
    @classmethod
    def _validate_market_data_sync_request_start_interval_seconds(
        cls, value: float
    ) -> float:
        if value < 0:
            raise ValueError(
                "MARKET_DATA_SYNC_REQUEST_START_INTERVAL_SECONDS must be >= 0."
            )
        return float(value)

    @field_validator("market_data_incremental_execution_mode")
    @classmethod
    def _validate_market_data_incremental_execution_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in _ALLOWED_INCREMENTAL_EXECUTION_MODES:
            allowed = ", ".join(sorted(_ALLOWED_INCREMENTAL_EXECUTION_MODES))
            raise ValueError(
                "MARKET_DATA_INCREMENTAL_EXECUTION_MODE must be one of: "
                f"{allowed}."
            )
        return normalized

    @field_validator(
        "market_data_incremental_sync_cron_hour",
        "market_data_incremental_sync_cron_minute",
    )
    @classmethod
    def _validate_market_data_incremental_cron(cls, value: int, info) -> int:
        raw = int(value)
        if info.field_name == "market_data_incremental_sync_cron_hour":
            if raw < 0 or raw > 23:
                raise ValueError(
                    "MARKET_DATA_INCREMENTAL_SYNC_CRON_HOUR must be in [0, 23]."
                )
        else:
            if raw < 0 or raw > 59:
                raise ValueError(
                    "MARKET_DATA_INCREMENTAL_SYNC_CRON_MINUTE must be in [0, 59]."
                )
        return raw

    @field_validator(
        "market_data_incremental_remote_timeout_seconds",
        "ibkr_gateway_timeout_seconds",
    )
    @classmethod
    def _validate_market_data_incremental_timeouts(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError("Incremental timeout settings must be > 0.")
        return float(value)

    @field_validator("market_data_incremental_hmac_max_skew_seconds")
    @classmethod
    def _validate_market_data_incremental_hmac_max_skew(cls, value: int) -> int:
        raw = int(value)
        if raw < 1:
            raise ValueError(
                "MARKET_DATA_INCREMENTAL_HMAC_MAX_SKEW_SECONDS must be >= 1."
            )
        return raw

    @field_validator("market_data_incremental_safety_lag_minutes")
    @classmethod
    def _validate_market_data_incremental_safety_lag_minutes(cls, value: int) -> int:
        raw = int(value)
        if raw < 0:
            raise ValueError("MARKET_DATA_INCREMENTAL_SAFETY_LAG_MINUTES must be >= 0.")
        return raw

    @field_validator("ibkr_gateway_port")
    @classmethod
    def _validate_ibkr_gateway_port(cls, value: int) -> int:
        raw = int(value)
        if raw < 1 or raw > 65535:
            raise ValueError("IBKR_GATEWAY_PORT must be in [1, 65535].")
        return raw

    @field_validator("ibkr_gateway_client_id")
    @classmethod
    def _validate_ibkr_gateway_client_id(cls, value: int) -> int:
        raw = int(value)
        if raw < 0:
            raise ValueError("IBKR_GATEWAY_CLIENT_ID must be >= 0.")
        return raw

    @field_validator("ibkr_forex_what_to_show", "ibkr_futures_what_to_show")
    @classmethod
    def _validate_ibkr_what_to_show(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if not normalized:
            raise ValueError("IBKR whatToShow values cannot be empty.")
        return normalized

    @field_validator("ibkr_futures_exchange_map_json", mode="before")
    @classmethod
    def _parse_ibkr_futures_exchange_map_json(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("IBKR_FUTURES_EXCHANGE_MAP_JSON must be a JSON object.")
            return parsed
        return value

    @field_validator(
        "pre_strategy_regime_lookback_bars",
        "pre_strategy_regime_min_bars",
        "pre_strategy_regime_cache_ttl_seconds",
        "pre_strategy_regime_pivot_window",
        "pre_strategy_regime_image_max_bars",
    )
    @classmethod
    def _validate_pre_strategy_regime_numeric_settings(cls, value: int) -> int:
        if value < 1:
            raise ValueError("PRE_STRATEGY regime numeric settings must be >= 1.")
        return value

    @field_validator("ccxt_market_data_timeout_seconds")
    @classmethod
    def _validate_ccxt_market_data_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("CCXT_MARKET_DATA_TIMEOUT_SECONDS must be > 0.")
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
            return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def effective_celery_broker_url(self) -> str:
        if isinstance(self.celery_broker_url, str) and self.celery_broker_url.strip():
            return self.celery_broker_url.strip()
        return self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        if (
            isinstance(self.celery_result_backend, str)
            and self.celery_result_backend.strip()
        ):
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
    def effective_telegram_webhook_url(self) -> str:
        """Resolved Telegram webhook URL from explicit URL or API public base URL."""
        explicit = self.telegram_webhook_url.strip()
        if explicit:
            return explicit
        base = self.api_public_base_url.strip().rstrip("/")
        if not base:
            return ""
        return f"{base}{self.api_v1_prefix}/social/webhooks/telegram"

    def _resolve_billing_return_base_url(self) -> str:
        base = self.api_public_base_url.strip().rstrip("/")
        if base:
            return base
        return f"http://127.0.0.1:{self.port}"

    def _default_billing_return_url(self, *, billing: str) -> str:
        base = self._resolve_billing_return_base_url()
        return f"{base}{self.api_v1_prefix}/billing/return?billing={billing}"

    @property
    def effective_billing_checkout_success_url(self) -> str:
        explicit = self.billing_checkout_success_url.strip()
        if explicit:
            return explicit
        return self._default_billing_return_url(billing="success")

    @property
    def effective_billing_checkout_cancel_url(self) -> str:
        explicit = self.billing_checkout_cancel_url.strip()
        if explicit:
            return explicit
        return self._default_billing_return_url(billing="cancel")

    @property
    def effective_billing_portal_return_url(self) -> str:
        explicit = self.billing_portal_return_url.strip()
        if explicit:
            return explicit
        return self._default_billing_return_url(billing="portal")

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
    def effective_market_data_redis_write_enabled(self) -> bool:
        value = self.market_data_redis_write_enabled
        if isinstance(value, bool):
            return value
        return self.runtime_env == "prod"

    @property
    def effective_market_data_redis_read_enabled(self) -> bool:
        value = self.market_data_redis_read_enabled
        if isinstance(value, bool):
            return value
        return self.runtime_env == "prod"

    @property
    def effective_market_data_redis_subs_enabled(self) -> bool:
        value = self.market_data_redis_subs_enabled
        if isinstance(value, bool):
            return value
        return self.runtime_env == "prod"

    @property
    def effective_market_data_runtime_fail_fast_on_redis_error(self) -> bool:
        value = self.market_data_runtime_fail_fast_on_redis_error
        if isinstance(value, bool):
            return value
        return self.runtime_env == "prod"

    @property
    def is_dev_mode(self) -> bool:
        """Whether app runs in local/development mode."""
        return self.runtime_env in {"dev", "test"}

    @property
    def effective_sentry_env(self) -> str:
        if isinstance(self.sentry_env, str) and self.sentry_env.strip():
            return self.sentry_env.strip()
        if self.runtime_env == "prod":
            return "production"
        if self.runtime_env == "dev":
            return "development"
        return self.runtime_env

    @property
    def effective_sentry_release(self) -> str | None:
        if isinstance(self.sentry_release, str):
            normalized = self.sentry_release.strip()
            if normalized:
                return normalized
        return None

    def _resolve_domain_mcp_server_url(self, *, domain: str) -> str:
        suffix = "prod" if self.runtime_env == "prod" else "dev"
        attr_name = f"mcp_server_url_{domain}_{suffix}"
        candidate = getattr(self, attr_name)
        normalized = candidate.strip()
        if not normalized:
            raise ValueError(f"{attr_name} must not be empty.")
        return normalized

    @property
    def strategy_mcp_server_url(self) -> str:
        """Strategy-domain MCP URL."""
        return self._resolve_domain_mcp_server_url(domain="strategy")

    @property
    def backtest_mcp_server_url(self) -> str:
        """Backtest-domain MCP URL."""
        return self._resolve_domain_mcp_server_url(domain="backtest")

    @property
    def market_data_mcp_server_url(self) -> str:
        """Market-data-domain MCP URL."""
        return self._resolve_domain_mcp_server_url(domain="market_data")

    @property
    def stress_mcp_server_url(self) -> str:
        """Stress-domain MCP URL."""
        return self._resolve_domain_mcp_server_url(domain="stress")

    @property
    def trading_mcp_server_url(self) -> str:
        """Trading-domain MCP URL."""
        return self._resolve_domain_mcp_server_url(domain="trading")

    @property
    def effective_mcp_context_secret(self) -> str:
        """Signing secret for MCP context token propagation."""
        if isinstance(self.mcp_context_secret, str) and self.mcp_context_secret.strip():
            return self.mcp_context_secret.strip()
        return self.secret_key

    @property
    def effective_trading_credentials_secret(self) -> str:
        """Signing/encryption secret for persisted broker credentials."""
        if (
            isinstance(self.trading_credentials_secret, str)
            and self.trading_credentials_secret.strip()
        ):
            return self.trading_credentials_secret.strip()
        return self.secret_key

    @property
    def stripe_price_to_tier_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        go = self.stripe_price_go_monthly.strip()
        plus = self.stripe_price_plus_monthly.strip()
        pro = self.stripe_price_pro_monthly.strip()
        go_yearly = self.stripe_price_go_yearly.strip()
        plus_yearly = self.stripe_price_plus_yearly.strip()
        pro_yearly = self.stripe_price_pro_yearly.strip()
        if go:
            mapping[go] = "go"
        if plus:
            mapping[plus] = "plus"
        if pro:
            mapping[pro] = "pro"
        if go_yearly:
            mapping[go_yearly] = "go"
        if plus_yearly:
            mapping[plus_yearly] = "plus"
        if pro_yearly:
            mapping[pro_yearly] = "pro"
        return mapping

    @property
    def stripe_price_to_interval_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        go = self.stripe_price_go_monthly.strip()
        plus = self.stripe_price_plus_monthly.strip()
        pro = self.stripe_price_pro_monthly.strip()
        go_yearly = self.stripe_price_go_yearly.strip()
        plus_yearly = self.stripe_price_plus_yearly.strip()
        pro_yearly = self.stripe_price_pro_yearly.strip()
        if go:
            mapping[go] = "monthly"
        if plus:
            mapping[plus] = "monthly"
        if pro:
            mapping[pro] = "monthly"
        if go_yearly:
            mapping[go_yearly] = "yearly"
        if plus_yearly:
            mapping[plus_yearly] = "yearly"
        if pro_yearly:
            mapping[pro_yearly] = "yearly"
        return mapping

    @property
    def stripe_product_to_tier_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        go = self.stripe_product_go.strip()
        if go:
            mapping[go] = "go"
        return mapping

    @property
    def openai_pricing(self) -> dict[str, dict[str, float]]:
        unified = (
            self.billing_pricing_json.get("openai_pricing")
            if isinstance(self.billing_pricing_json, dict)
            else None
        )
        source = unified if isinstance(unified, dict) else self.openai_pricing_json
        if not isinstance(source, dict):
            return {}

        normalized: dict[str, dict[str, float]] = {}
        for model, values in source.items():
            if not isinstance(model, str) or not isinstance(values, dict):
                continue
            parsed_row: dict[str, float] = {}
            for key, value in values.items():
                if not isinstance(key, str):
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed < 0:
                    continue
                parsed_row[key] = parsed
            normalized[model] = parsed_row
        return normalized

    @property
    def openai_allowed_models(self) -> tuple[str, ...]:
        values: list[str] = []
        for raw in self.openai_allowed_models_json:
            if not isinstance(raw, str):
                continue
            model = raw.strip()
            if not model or model in values:
                continue
            values.append(model)
        if (
            self.openai_response_model.strip()
            and self.openai_response_model.strip() not in values
        ):
            values.append(self.openai_response_model.strip())
        if not values:
            return ("gpt-5.2", "gpt-5.4")
        return tuple(values)

    @property
    def billing_tier_limits(self) -> dict[str, dict[str, int]]:
        default_limits: dict[str, dict[str, int]] = {
            "free": {
                "ai_tokens_monthly_total": 100_000,
                "cpu_tokens_monthly_total": 30,
                "strategies_current_count": 5,
                "deployments_running_count": 1,
            },
            "go": {
                "ai_tokens_monthly_total": 350_000,
                "cpu_tokens_monthly_total": 105,
                "strategies_current_count": 15,
                "deployments_running_count": 3,
            },
            "plus": {
                "ai_tokens_monthly_total": 1_000_000,
                "cpu_tokens_monthly_total": 300,
                "strategies_current_count": 30,
                "deployments_running_count": 5,
            },
            "pro": {
                "ai_tokens_monthly_total": 3_000_000,
                "cpu_tokens_monthly_total": 900,
                "strategies_current_count": 100,
                "deployments_running_count": 20,
            },
        }
        unified_tier_limits = (
            self.billing_pricing_json.get("tier_limits")
            if isinstance(self.billing_pricing_json, dict)
            else None
        )
        raw = (
            unified_tier_limits
            if isinstance(unified_tier_limits, dict)
            else self.billing_config_json.get("tier_limits")
            if isinstance(self.billing_config_json, dict)
            else None
        )
        if not isinstance(raw, dict):
            return default_limits

        merged: dict[str, dict[str, int]] = {
            tier: dict(values) for tier, values in default_limits.items()
        }
        for tier, values in raw.items():
            if (
                not isinstance(tier, str)
                or tier not in merged
                or not isinstance(values, dict)
            ):
                continue
            for key, value in values.items():
                if not isinstance(key, str):
                    continue
                normalized_key = (
                    "cpu_tokens_monthly_total"
                    if key == "cpu_jobs_monthly_total"
                    else key
                )
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed < 0:
                    continue
                merged[tier][normalized_key] = parsed
        return merged

    @property
    def billing_user_quota_overrides(self) -> dict[str, dict[str, Any]]:
        raw = (
            self.billing_pricing_json.get("user_quota_overrides")
            if isinstance(self.billing_pricing_json, dict)
            else None
        )
        if not isinstance(raw, dict):
            return {}

        allowed_metrics = {
            "ai_tokens_monthly_total",
            "cpu_tokens_monthly_total",
            "strategies_current_count",
            "deployments_running_count",
        }
        normalized: dict[str, dict[str, Any]] = {}
        for raw_user_id, raw_config in raw.items():
            if not isinstance(raw_user_id, str) or not isinstance(raw_config, dict):
                continue
            user_id = raw_user_id.strip()
            if not user_id:
                continue

            try:
                percent_bonus = int(raw_config.get("percent_bonus", 0))
            except (TypeError, ValueError):
                continue
            if percent_bonus <= 0:
                continue

            metrics_raw = raw_config.get("metrics")
            metrics: list[str] = []
            if isinstance(metrics_raw, (list, tuple, set)):
                for item in metrics_raw:
                    if not isinstance(item, str):
                        continue
                    metric = item.strip()
                    if not metric or metric not in allowed_metrics or metric in metrics:
                        continue
                    metrics.append(metric)
            if not metrics:
                metrics = [
                    "ai_tokens_monthly_total",
                    "cpu_tokens_monthly_total",
                ]

            normalized[user_id] = {
                "percent_bonus": percent_bonus,
                "metrics": tuple(metrics),
            }
        return normalized

    @property
    def billing_cost_model(self) -> dict[str, float]:
        defaults: dict[str, float] = {
            "token_cost_per_1k_usd": 0.01,
            "ai_usage_unit_usd": 0.00001,
            "cpu_job_cost_usd": 0.20,
            "cpu_bars_per_token": 52_560.0,
            "target_profit_margin": 0.50,
            "go_price_usd": 8.0,
            "plus_price_usd": 20.0,
            "pro_price_usd": 60.0,
            "go_price_usd_yearly": 81.6,
            "plus_price_usd_yearly": 196.8,
            "pro_price_usd_yearly": 576.0,
        }
        unified_cost_model = (
            self.billing_pricing_json.get("cost_model")
            if isinstance(self.billing_pricing_json, dict)
            else None
        )
        raw = (
            unified_cost_model
            if isinstance(unified_cost_model, dict)
            else self.billing_cost_model_json
            if isinstance(self.billing_cost_model_json, dict)
            else {}
        )
        merged = dict(defaults)
        for key in defaults:
            value = raw.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if parsed < 0:
                continue
            merged[key] = parsed
        return merged


@lru_cache
def _normalize_service_name(raw: str | None) -> str:
    if not isinstance(raw, str):
        return ""
    normalized = raw.strip().lower().replace("-", "_")
    return normalized


def _collect_existing_env_files(paths: list[Path]) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    for item in paths:
        candidate = item.expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_file():
            continue
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return tuple(resolved)


def _resolve_default_env_files(*, service: str | None = None) -> tuple[str, ...]:
    profile = _normalize_env_profile(
        os.getenv("MINSY_ENV_PROFILE", "").strip()
        or os.getenv("APP_ENV", "").strip()
        or "dev"
    )
    service_name = _normalize_service_name(service or os.getenv("MINSY_SERVICE"))

    env_dir = Path("env")
    profile_file = env_dir / f".env.{profile}"
    profile_candidate = profile_file.expanduser()
    if not profile_candidate.is_absolute():
        profile_candidate = (Path.cwd() / profile_candidate).resolve()
    if not profile_candidate.is_file():
        raise ValueError(
            f"Missing required profile file: {profile_candidate}. "
            f"Create env/.env.{profile} for profile '{profile}'."
        )

    candidates: list[Path] = [
        env_dir / ".env.secrets",
        env_dir / ".env.common",
        profile_file,
    ]
    if service_name:
        candidates.extend(
            [
                env_dir / f".env.{profile}.{service_name}",
            ]
        )
    return _collect_existing_env_files(candidates)


def _resolve_env_files(
    *,
    env_file: str | None = None,
    service: str | None = None,
) -> str | tuple[str, ...] | None:
    explicit_env_file = (env_file or "").strip()
    if explicit_env_file:
        return explicit_env_file

    explicit_list = os.getenv("MINSY_ENV_FILES", "").strip()
    if explicit_list:
        raw_items = [item.strip() for item in explicit_list.split(",") if item.strip()]
        resolved_list = _collect_existing_env_files([Path(item) for item in raw_items])
        if resolved_list:
            return resolved_list
        return tuple(raw_items)

    legacy_env_file = os.getenv("MINSY_ENV_FILE", "").strip()
    if legacy_env_file:
        return legacy_env_file

    resolved_defaults = _resolve_default_env_files(service=service)
    if resolved_defaults:
        return resolved_defaults
    return None


@lru_cache
def get_settings(
    *,
    env_file: str | None = None,
    service: str | None = None,
) -> Settings:
    resolved_env_files = _resolve_env_files(env_file=env_file, service=service)
    return Settings(_env_file=resolved_env_files)


settings = get_settings()
