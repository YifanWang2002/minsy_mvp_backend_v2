"""Celery tasks for market-data refresh and warmup."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import monotonic, time
from typing import Any
from uuid import UUID

import httpx

from src.config import settings
from src.engine.market_data.providers.alpaca_rest import AlpacaRestProvider
from src.engine.market_data.runtime import market_data_runtime
from src.engine.market_data.sync_service import (
    execute_market_data_sync_job_with_fresh_session,
)
from src.models import database as db_module
from src.models.market_data_error_event import MarketDataErrorEvent
from src.models.redis import get_sync_redis_client
from src.util.logger import logger
from src.workers.celery_app import celery_app

_REFRESH_DEDUPE_FALLBACK: dict[str, float] = {}
_REFRESH_DEDUPE_LOCK = Lock()


def _normalize_error_info(exc: Exception) -> tuple[str, int | None, str]:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", None, str(exc)[:500]
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return "http_404", status, str(exc)[:500]
        if status == 429:
            return "http_429", status, str(exc)[:500]
        return "http_error", status, str(exc)[:500]
    return type(exc).__name__, None, str(exc)[:500]


def _refresh_dedupe_key(market: str, symbol: str) -> str:
    return f"md:v1:refresh_dedupe:{market.strip().lower()}:{symbol.strip().upper()}"


def _refresh_dedupe_ttl_seconds() -> int:
    interval = max(1, int(settings.market_data_refresh_active_subscriptions_interval_seconds))
    window = max(1, int(settings.market_data_refresh_dedupe_window_seconds))
    return max(interval, window)


def _reserve_refresh_slot(market: str, symbol: str) -> bool:
    if not settings.market_data_refresh_dedupe_enabled:
        return True
    key = _refresh_dedupe_key(market, symbol)
    ttl = _refresh_dedupe_ttl_seconds()
    try:
        client = get_sync_redis_client()
        reserved = client.set(key, str(int(time() * 1000)), nx=True, ex=ttl)
        return bool(reserved)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[market-data-worker] refresh dedupe redis unavailable, fallback to local lock error=%s",
            type(exc).__name__,
        )
    now = monotonic()
    with _REFRESH_DEDUPE_LOCK:
        expired_keys = [item for item, expire_at in _REFRESH_DEDUPE_FALLBACK.items() if expire_at <= now]
        for item in expired_keys:
            _REFRESH_DEDUPE_FALLBACK.pop(item, None)
        current_expire_at = _REFRESH_DEDUPE_FALLBACK.get(key)
        if current_expire_at is not None and current_expire_at > now:
            return False
        _REFRESH_DEDUPE_FALLBACK[key] = now + float(ttl)
    return True


async def _record_market_data_error_event(
    *,
    market: str,
    symbol: str,
    endpoint: str,
    exc: Exception,
    metadata: dict[str, Any] | None = None,
) -> None:
    error_type, http_status, message = _normalize_error_info(exc)
    try:
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as session:
            session.add(
                MarketDataErrorEvent(
                    market=market,
                    symbol=symbol,
                    error_type=error_type,
                    endpoint=endpoint,
                    http_status=http_status,
                    message=message,
                    occurred_at=datetime.now(UTC),
                    metadata_=metadata or {},
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("failed to persist market_data_error_event")
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


async def _refresh_symbol_once(*, market: str, symbol: str) -> dict[str, str | int]:
    provider = AlpacaRestProvider()
    try:
        errors = 0
        quote = None
        latest_bar = None

        try:
            quote = await provider.fetch_quote(symbol=symbol, market=market)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning(
                "[market-data-worker] quote refresh failed market=%s symbol=%s error=%s",
                market,
                symbol,
                type(exc).__name__,
            )
            await _record_market_data_error_event(
                market=market,
                symbol=symbol,
                endpoint="fetch_quote",
                exc=exc,
            )
        if quote is not None:
            market_data_runtime.upsert_quote(market=market, symbol=symbol, quote=quote)

        try:
            latest_bar = await provider.fetch_latest_1m_bar(symbol=symbol, market=market)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning(
                "[market-data-worker] latest-bar refresh failed market=%s symbol=%s error=%s",
                market,
                symbol,
                type(exc).__name__,
            )
            await _record_market_data_error_event(
                market=market,
                symbol=symbol,
                endpoint="fetch_latest_1m_bar",
                exc=exc,
            )
        if latest_bar is not None:
            market_data_runtime.ingest_1m_bar(market=market, symbol=symbol, bar=latest_bar)

        status = "ok"
        if errors > 0 and (quote is not None or latest_bar is not None):
            status = "partial_error"
        elif errors > 0:
            status = "error"
        return {
            "market": market,
            "symbol": symbol,
            "status": status,
            "bars": 1 if latest_bar else 0,
            "latest_bar_time": latest_bar.timestamp.isoformat() if latest_bar is not None else None,
            "errors": errors,
        }
    finally:
        await provider.aclose()


@celery_app.task(name="market_data.refresh_symbol")
def refresh_symbol_task(market: str, symbol: str) -> dict[str, str | int]:
    """Refresh latest quote and 1m bar for one symbol."""
    logger.info("[market-data-worker] refresh symbol market=%s symbol=%s", market, symbol)
    return asyncio.run(_refresh_symbol_once(market=market, symbol=symbol))


async def _backfill_symbol_once(
    *,
    market: str,
    symbol: str,
    minutes: int,
) -> dict[str, str | int]:
    provider = AlpacaRestProvider()
    try:
        since = datetime.now(UTC) - timedelta(minutes=minutes)
        bars = []
        try:
            bars = await provider.fetch_recent_1m_bars(
                symbol=symbol,
                market=market,
                since=since,
                limit=min(settings.market_data_backfill_limit, max(minutes, 1)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[market-data-worker] backfill failed market=%s symbol=%s error=%s",
                market,
                symbol,
                type(exc).__name__,
            )
            await _record_market_data_error_event(
                market=market,
                symbol=symbol,
                endpoint="fetch_recent_1m_bars",
                exc=exc,
                metadata={"minutes": minutes},
            )
            return {"market": market, "symbol": symbol, "status": "error", "bars": 0, "errors": 1}
        for bar in bars:
            market_data_runtime.ingest_1m_bar(market=market, symbol=symbol, bar=bar)
        return {"market": market, "symbol": symbol, "status": "ok", "bars": len(bars), "errors": 0}
    finally:
        await provider.aclose()


@celery_app.task(name="market_data.backfill_symbol")
def backfill_symbol_task(market: str, symbol: str, minutes: int = 120) -> dict[str, str | int]:
    """Backfill recent 1m bars for warmup."""
    logger.info(
        "[market-data-worker] backfill symbol market=%s symbol=%s minutes=%s",
        market,
        symbol,
        minutes,
    )
    return asyncio.run(_backfill_symbol_once(market=market, symbol=symbol, minutes=minutes))


@celery_app.task(name="market_data.refresh_active_subscriptions")
def refresh_active_subscriptions_task() -> dict[str, int]:
    """Schedule refresh tasks for all currently active symbols."""
    subscriptions = market_data_runtime.active_subscriptions()
    scheduled = 0
    deduped = 0
    for market, symbol in subscriptions:
        if not _reserve_refresh_slot(market, symbol):
            deduped += 1
            continue
        refresh_symbol_task.apply_async(args=(market, symbol), queue="market_data")
        scheduled += 1
    total = len(subscriptions)
    market_data_runtime.record_refresh_scheduler_metrics(
        scheduled=scheduled,
        deduped=deduped,
        total=total,
    )
    return {"scheduled": scheduled, "deduped": deduped, "total": total}


def enqueue_market_data_refresh(*, market: str, symbol: str) -> str:
    result = refresh_symbol_task.apply_async(args=(market, symbol), queue="market_data")
    return str(result.id)


async def _run_market_data_sync_job_once(job_uuid: UUID):
    with suppress(Exception):
        await db_module.close_postgres()

    try:
        return await execute_market_data_sync_job_with_fresh_session(job_uuid)
    finally:
        with suppress(Exception):
            await db_module.close_postgres()


@celery_app.task(name="market_data.sync_missing_ranges")
def sync_missing_ranges_task(job_id: str) -> dict[str, str | int]:
    """Execute one market-data missing-range sync job."""

    job_uuid = UUID(job_id)
    logger.info("[market-data-worker] sync missing ranges job_id=%s", job_uuid)
    view = asyncio.run(_run_market_data_sync_job_once(job_uuid))
    logger.info(
        "[market-data-worker] sync finished job_id=%s status=%s progress=%s rows_written=%s",
        view.job_id,
        view.status,
        view.progress,
        view.rows_written,
    )
    return {
        "job_id": str(view.job_id),
        "status": view.status,
        "progress": view.progress,
        "rows_written": view.rows_written,
    }


def enqueue_market_data_sync_job(job_id: UUID | str) -> str:
    """Enqueue market-data sync job execution and return task id."""

    result = sync_missing_ranges_task.apply_async(args=(str(job_id),), queue="market_data")
    return str(result.id)
