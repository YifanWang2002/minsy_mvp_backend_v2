"""Health-check and runtime status endpoints."""

from __future__ import annotations

import asyncio
import platform
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.config import settings
from src.models.database import postgres_healthcheck
from src.models.redis import redis_healthcheck
from src.util.logger import logger
from src.workers.celery_app import celery_app

router = APIRouter(tags=["health"])

ServiceState = Literal["healthy", "degraded", "down"]


def _build_service_status(state: ServiceState, details: str) -> dict[str, str]:
    return {
        "status": state,
        "details": details,
    }


def _overall_status(states: list[ServiceState]) -> ServiceState:
    if any(item == "down" for item in states):
        return "down"
    if any(item == "degraded" for item in states):
        return "degraded"
    return "healthy"


async def _ping_openai_models() -> tuple[bool, int | None, str]:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return False, None, "OPENAI_API_KEY is missing."

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
    except httpx.TimeoutException:
        return False, None, "Timed out while pinging OpenAI models endpoint."
    except httpx.HTTPError:
        return False, None, "Failed to reach OpenAI models endpoint."

    if response.status_code == status.HTTP_200_OK:
        return True, response.status_code, "OpenAI models endpoint is reachable."
    if response.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}:
        return (
            False,
            response.status_code,
            f"OpenAI rejected API key (HTTP {response.status_code}).",
        )

    return (
        False,
        response.status_code,
        f"OpenAI models endpoint returned HTTP {response.status_code}.",
    )


async def _build_chat_service_status() -> dict[str, str]:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return _build_service_status(
            "down",
            "OPENAI_API_KEY is not configured in backend environment.",
        )

    ping_ok, status_code, ping_details = await _ping_openai_models()
    if ping_ok:
        return _build_service_status(
            "healthy",
            "OPENAI_API_KEY detected and OpenAI connectivity verified via /v1/models.",
        )

    if status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}:
        return _build_service_status("down", ping_details)

    return _build_service_status("degraded", ping_details)


def _inspect_celery_workers() -> dict[str, dict[str, str]] | None:
    inspector = celery_app.control.inspect(timeout=1.0)
    if inspector is None:
        return None
    return inspector.ping()


async def _build_backtest_service_status() -> dict[str, str]:
    redis_ok = await redis_healthcheck()
    redis_details = "Redis ping ok." if redis_ok else "Redis ping failed."

    try:
        workers = await asyncio.to_thread(_inspect_celery_workers)
        celery_ok = isinstance(workers, dict) and bool(workers)
    except Exception:  # noqa: BLE001
        logger.exception("Celery worker status check failed.")
        workers = None
        celery_ok = False

    if celery_ok:
        celery_details = f"Celery worker ping ok ({len(workers or {})} worker(s))."
    else:
        celery_details = "No Celery worker responded to ping."

    details = f"{redis_details} {celery_details}"
    if redis_ok and celery_ok:
        return _build_service_status("healthy", details)
    if redis_ok or celery_ok:
        return _build_service_status("degraded", details)
    return _build_service_status("down", details)


@router.get("/health")
async def health_check() -> JSONResponse:
    """Check API, PostgreSQL and Redis runtime health."""
    postgres_ok = await postgres_healthcheck()
    redis_ok = await redis_healthcheck()
    all_ok = postgres_ok and redis_ok

    payload = {
        "status": "ok" if all_ok else "degraded",
        "db": postgres_ok,
        "redis": redis_ok,
        "services": {
            "api": "ok",
            "postgres": "ok" if postgres_ok else "down",
            "redis": "ok" if redis_ok else "down",
        },
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload,
    )


@router.get("/status")
async def status_check() -> JSONResponse:
    """Server status snapshot for frontend settings page."""
    start = perf_counter()

    api_endpoint = _build_service_status(
        "healthy",
        "FastAPI endpoint is responding.",
    )
    chat = await _build_chat_service_status()
    backtest = await _build_backtest_service_status()
    live_trading = _build_service_status(
        "down",
        "Live trading is disabled by default.",
    )

    overall = _overall_status(
        [
            api_endpoint["status"],  # type: ignore[list-item]
            chat["status"],  # type: ignore[list-item]
            backtest["status"],  # type: ignore[list-item]
        ]
    )

    elapsed_ms = (perf_counter() - start) * 1000.0
    payload = {
        "status": overall,
        "services": {
            "api_endpoint": api_endpoint,
            "chat": chat,
            "backtest": backtest,
            "live_trading": live_trading,
        },
        "system": {
            "platform": platform.platform(),
            "cpu_usage_percent": 0.0,
            "memory_usage_percent": 0.0,
            "memory_available_gb": 0.0,
        },
        "response_time_ms": round(elapsed_ms, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=payload,
    )
