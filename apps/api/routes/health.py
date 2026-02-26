"""Health-check and runtime status endpoints."""

from __future__ import annotations

import asyncio
import platform
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

import httpx
import psutil
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from packages.shared_settings.schema.settings import settings
from packages.domain.trading.runtime.circuit_breaker import get_broker_request_circuit_breaker
from packages.domain.trading.runtime.kill_switch import RuntimeKillSwitch
from packages.infra.redis.stores.runtime_state_store import runtime_state_store
from packages.infra.redis.stores.signal_store import signal_store
from packages.domain.market_data.runtime import market_data_runtime
from packages.infra.db.session import postgres_healthcheck
from packages.infra.redis.client import redis_healthcheck
from packages.infra.observability.logger import logger
from packages.infra.queue.celery_app import celery_app

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


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


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


async def _build_flower_service_status() -> dict[str, str]:
    if not settings.flower_enabled:
        return _build_service_status(
            "down",
            "Flower is disabled by FLOWER_ENABLED=false.",
        )

    flower_url = f"http://{settings.flower_host}:{settings.flower_port}/"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(flower_url)
    except httpx.TimeoutException:
        return _build_service_status("degraded", "Flower probe timed out.")
    except httpx.HTTPError:
        return _build_service_status("down", "Flower endpoint is unreachable.")

    if response.status_code in {
        status.HTTP_200_OK,
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    }:
        return _build_service_status(
            "healthy",
            f"Flower endpoint responded with HTTP {response.status_code}.",
        )
    return _build_service_status(
        "degraded",
        f"Flower endpoint returned HTTP {response.status_code}.",
    )


def _collect_system_metrics() -> dict[str, float | str]:
    memory = psutil.virtual_memory()
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_usage = float(memory.percent)
    memory_available_gb = float(memory.available) / (1024**3)
    return {
        "platform": platform.platform(),
        "cpu_usage_percent": round(float(cpu_usage), 2),
        "memory_usage_percent": round(memory_usage, 2),
        "memory_available_gb": round(memory_available_gb, 2),
    }


async def _build_system_metrics() -> dict[str, float | str]:
    try:
        return await asyncio.to_thread(_collect_system_metrics)
    except Exception:  # noqa: BLE001
        logger.exception("System metrics collection failed.")
        return {
            "platform": platform.platform(),
            "cpu_usage_percent": 0.0,
            "memory_usage_percent": 0.0,
            "memory_available_gb": 0.0,
        }


def _build_state_store_fallback_status() -> dict[str, dict[str, object]]:
    return {
        "runtime_state_store": runtime_state_store.fallback_status(),
        "signal_store": signal_store.fallback_status(),
    }


async def _build_live_trading_service_status() -> dict[str, str]:
    if not settings.paper_trading_enabled:
        return _build_service_status(
            "down",
            "Paper trading is disabled by PAPER_TRADING_ENABLED=false.",
        )

    kill_snapshot = RuntimeKillSwitch().snapshot()
    if bool(kill_snapshot.get("global")):
        return _build_service_status(
            "down",
            "Global paper-trading kill switch is enabled.",
        )

    now_dt = datetime.now(UTC)
    now_ms = int(now_dt.timestamp() * 1000)
    stale_seconds = max(1, settings.paper_trading_runtime_health_stale_seconds)
    stale_ms = stale_seconds * 1000
    if hasattr(market_data_runtime, "redis_data_plane_status"):
        redis_data_plane = market_data_runtime.redis_data_plane_status()
    else:
        redis_data_plane = {
            "enabled": False,
            "available": True,
            "market_data_store_ok": True,
            "subscription_store_ok": True,
            "last_error": None,
        }
    if hasattr(market_data_runtime, "freshest_checkpoint_ms"):
        freshest_checkpoint_ms = market_data_runtime.freshest_checkpoint_ms(timeframe="1m")
    else:
        checkpoints = market_data_runtime.checkpoints()
        freshest_checkpoint_ms = None
        for key, ts_ms in checkpoints.items():
            if key.endswith(":1m"):
                candidate = int(ts_ms)
                if freshest_checkpoint_ms is None or candidate > freshest_checkpoint_ms:
                    freshest_checkpoint_ms = candidate
    recent_market_data = False
    market_data_source = "redis_watermark"
    market_data_lag_seconds: float | None = None
    if freshest_checkpoint_ms is not None:
        market_data_lag_seconds = max(0.0, (now_ms - int(freshest_checkpoint_ms)) / 1000.0)
        if now_ms - int(freshest_checkpoint_ms) <= stale_ms:
            recent_market_data = True

    if not recent_market_data:
        live_health = await runtime_state_store.get_live_trading_health()
        if isinstance(live_health, dict):
            runtime_reason = str(live_health.get("runtime_reason", "")).strip().lower()
            runtime_bar_time = _parse_datetime(live_health.get("runtime_bar_time"))
            runtime_updated_at = _parse_datetime(live_health.get("updated_at"))

            if runtime_bar_time is not None:
                age_seconds = (now_dt - runtime_bar_time).total_seconds()
                if age_seconds <= stale_seconds:
                    recent_market_data = True
                    market_data_source = "runtime_state_store"
            elif runtime_updated_at is not None and runtime_reason not in {
                "",
                "no_market_data",
                "deployment_paused",
                "deployment_stopped",
            }:
                age_seconds = (now_dt - runtime_updated_at).total_seconds()
                if age_seconds <= stale_seconds:
                    recent_market_data = True
                    market_data_source = "runtime_state_store"

    try:
        workers = await asyncio.to_thread(_inspect_celery_workers)
        paper_workers_up = isinstance(workers, dict) and bool(workers)
    except Exception:  # noqa: BLE001
        logger.exception("Live trading worker probe failed.")
        paper_workers_up = False

    breaker_snapshot = await get_broker_request_circuit_breaker().snapshot()
    circuit_open = breaker_snapshot.state == "open"
    order_execution_enabled = bool(settings.paper_trading_execute_orders)
    blocked_users = len(kill_snapshot.get("blocked_users", []))
    blocked_deployments = len(kill_snapshot.get("blocked_deployments", []))
    if hasattr(market_data_runtime, "runtime_metrics"):
        runtime_metrics = market_data_runtime.runtime_metrics()
    else:
        runtime_metrics = {}
    refresh_scheduler_metrics = runtime_metrics.get("refresh_scheduler", {})
    if isinstance(refresh_scheduler_metrics, dict):
        duplicate_rate_pct = float(refresh_scheduler_metrics.get("duplicate_rate_pct", 0.0))
    else:
        duplicate_rate_pct = 0.0
    redis_available = bool(redis_data_plane.get("available", True))
    redis_required = bool(settings.effective_market_data_redis_read_enabled)
    details = (
        f"market_data_fresh={'yes' if recent_market_data else 'no'}; "
        f"market_data_source={market_data_source}; "
        f"market_data_redis_available={'yes' if redis_available else 'no'}; "
        f"market_data_lag_seconds={'n/a' if market_data_lag_seconds is None else round(market_data_lag_seconds, 3)}; "
        f"refresh_duplicate_rate_pct={round(duplicate_rate_pct, 2)}; "
        f"paper_worker_up={'yes' if paper_workers_up else 'no'}; "
        f"order_execution={'broker' if order_execution_enabled else 'simulated'}; "
        f"broker_circuit={breaker_snapshot.state}; "
        f"kill_switch_users={blocked_users}; "
        f"kill_switch_deployments={blocked_deployments}."
    )

    if redis_required and not redis_available:
        if settings.effective_market_data_runtime_fail_fast_on_redis_error:
            return _build_service_status("down", details)
        return _build_service_status("degraded", details)
    if not recent_market_data and not paper_workers_up:
        return _build_service_status("down", details)
    if circuit_open or not recent_market_data or not paper_workers_up or not order_execution_enabled:
        return _build_service_status("degraded", details)
    return _build_service_status("healthy", details)


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
        "state_stores": _build_state_store_fallback_status(),
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
    flower = await _build_flower_service_status()
    system_metrics = await _build_system_metrics()
    live_trading = await _build_live_trading_service_status()
    state_stores = _build_state_store_fallback_status()

    overall = _overall_status(
        [
            api_endpoint["status"],  # type: ignore[list-item]
            chat["status"],  # type: ignore[list-item]
            backtest["status"],  # type: ignore[list-item]
            live_trading["status"],  # type: ignore[list-item]
        ]
    )

    elapsed_ms = (perf_counter() - start) * 1000.0
    payload = {
        "status": overall,
        "services": {
            "api_endpoint": api_endpoint,
            "chat": chat,
            "backtest": backtest,
            "flower": flower,
            "live_trading": live_trading,
        },
        "system": system_metrics,
        "state_stores": state_stores,
        "response_time_ms": round(elapsed_ms, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=payload,
    )
