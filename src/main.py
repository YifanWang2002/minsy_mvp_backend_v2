"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import api_router
from src.config import settings
from src.models.database import close_postgres, init_postgres
from src.models.redis import close_redis, init_redis
from src.util.data_setup import ensure_market_data
from src.util.logger import banner, configure_logging, log_success, logger

configure_logging(settings.log_level, show_sql=settings.sqlalchemy_echo)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize and close infrastructure resources."""
    banner()
    logger.info("Starting application lifecycle.")

    # Ensure market-data parquet files exist (downloads on first run).
    ensure_market_data()

    logger.info(
        "Runtime AI config: model=%s, mcp_server=%s",
        settings.openai_response_model,
        settings.mcp_server_url,
    )
    await init_postgres()
    await init_redis()
    log_success("Infrastructure initialized.")
    try:
        yield
    finally:
        await close_redis()
        await close_postgres()
        logger.info("Application lifecycle closed.")


def create_app() -> FastAPI:
    """Application factory for uvicorn and testing."""
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
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": settings.app_name, "status": "running"}

    return app


app = create_app()
