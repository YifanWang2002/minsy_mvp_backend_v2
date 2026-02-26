"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure service-scoped env layering resolves before settings import.
os.environ.setdefault("MINSY_SERVICE", "api")

from apps.api.middleware.sentry_http_status import SentryHTTPStatusMiddleware
from apps.api.router import api_router
from apps.api.services.telegram_webhook_sync import sync_telegram_webhook_on_startup
from packages.infra.db.session import close_postgres, init_postgres
from packages.infra.observability.logger import (
    banner,
    configure_logging,
    log_success,
    logger,
)
from packages.infra.observability.sentry import init_backend_sentry
from packages.infra.providers.market_data.data_setup import ensure_market_data
from packages.infra.redis.client import close_redis, init_redis
from packages.shared_settings import get_api_settings

settings = get_api_settings()
configure_logging(settings.log_level, show_sql=settings.sqlalchemy_echo)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize and close infrastructure resources."""
    banner()
    logger.info("Starting application lifecycle.")

    # Ensure market-data parquet files exist (downloads on first run).
    ensure_market_data()

    logger.info(
        (
            "Runtime AI config: model=%s, "
            "mcp_strategy=%s, mcp_backtest=%s, mcp_market=%s, mcp_stress=%s, mcp_trading=%s"
        ),
        settings.openai_response_model,
        settings.strategy_mcp_server_url,
        settings.backtest_mcp_server_url,
        settings.market_data_mcp_server_url,
        settings.stress_mcp_server_url,
        settings.trading_mcp_server_url,
    )
    core_domain_urls = {
        "strategy": settings.strategy_mcp_server_url,
        "backtest": settings.backtest_mcp_server_url,
        "market_data": settings.market_data_mcp_server_url,
    }
    if len(set(core_domain_urls.values())) < len(core_domain_urls):
        logger.warning(
            "MCP core domain URLs overlap unexpectedly: %s",
            core_domain_urls,
        )
    await init_postgres()
    await init_redis()
    await sync_telegram_webhook_on_startup()
    log_success("Infrastructure initialized.")
    try:
        yield
    finally:
        await close_redis()
        await close_postgres()
        logger.info("Application lifecycle closed.")


def create_app() -> FastAPI:
    """Application factory for uvicorn and testing."""
    init_backend_sentry(source="fastapi")
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    if settings.is_dev_mode:
        # Dev/Test mode: disable CORS origin restrictions for dynamic localhost ports.
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=".*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.effective_cors_origins,
            allow_origin_regex=settings.cors_origin_regex,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(SentryHTTPStatusMiddleware)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": settings.app_name, "status": "running"}

    return app


app = create_app()
