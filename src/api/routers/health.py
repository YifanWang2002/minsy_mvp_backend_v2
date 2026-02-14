"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.models.database import postgres_healthcheck
from src.models.redis import redis_healthcheck

router = APIRouter(tags=["health"])


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
