"""Celery tasks for market-data refresh and warmup."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import ceil
from threading import Lock
from time import monotonic, time
from typing import Any
from uuid import UUID

import httpx

from packages.shared_settings.schema.settings import settings
from packages.infra.providers.market_data.alpaca_rest import AlpacaRestProvider
from packages.domain.market_data.runtime import market_data_runtime
from packages.domain.market_data.sync_service import (
    execute_market_data_sync_job_with_fresh_session,
)
from packages.infra.db import session as db_module
from packages.infra.db.models.market_data_error_event import MarketDataErrorEvent
from packages.infra.redis.client import get_sync_redis_client
from packages.infra.observability.logger import logger
from apps.worker.common.celery_base import celery_app

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


def _normalize_timeframe(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return normalized


def _timeframe_step(value: str) -> timedelta | None:
    normalized = _normalize_timeframe(value)
    if normalized is None:
        return None
    if normalized.endswith("m"):
        try:
            minutes = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if minutes <= 0:
            return None
        return timedelta(minutes=minutes)
    if normalized.endswith("h"):
        try:
            hours = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if hours <= 0:
            return None
        return timedelta(hours=hours)
    if normalized.endswith("d"):
        try:
            days = int(normalized[:-1] or "0")
        except ValueError:
            return None
        if days <= 0:
            return None
        return timedelta(days=days)
    return None


def _history_max_lag(market: str, timeframe: str) -> timedelta | None:
    normalized_market = str(market).strip().lower()
    step = _timeframe_step(timeframe)
    if step is None:
        return None
    if normalized_market == "crypto":
        return max(step * 3, timedelta(minutes=20))
    if normalized_market == "forex":
        return max(step * 6, timedelta(hours=6))
    return None


def _bars_are_fresh(
    *,
    market: str,
    timeframe: str,
    bars: list[Any],
) -> bool:
    if not bars:
        return False
    max_lag = _history_max_lag(market, timeframe)
    if max_lag is None:
        return True
    latest = getattr(bars[-1], "timestamp", None)
    if not isinstance(latest, datetime):
        return False
    return latest.astimezone(UTC) >= (datetime.now(UTC) - max_lag)


def _bars_are_ready(
    *,
    market: str,
    timeframe: str,
    bars: list[Any],
    target_bars: int,
) -> bool:
    return len(bars) >= max(1, int(target_bars)) and _bars_are_fresh(
        market=market,
        timeframe=timeframe,
        bars=bars,
    )


def _aggregate_bars_from_source(
    *,
    source_bars: list[Any],
    source_timeframe: str,
    target_timeframe: str,
    target_bars: int,
) -> list[Any]:
    source_step = _timeframe_step(source_timeframe)
    target_step = _timeframe_step(target_timeframe)
    if source_step is None or target_step is None:
        return []
    source_ms = int(source_step.total_seconds() * 1000)
    target_ms = int(target_step.total_seconds() * 1000)
    if source_ms <= 0 or target_ms <= 0 or target_ms < source_ms or target_ms % source_ms != 0:
        return []

    ordered = sorted(
        (
            item
            for item in source_bars
            if isinstance(getattr(item, "timestamp", None), datetime)
        ),
        key=lambda item: item.timestamp,
    )
    if not ordered:
        return []

    buckets: dict[int, Any] = {}
    for bar in ordered:
        ts = bar.timestamp.astimezone(UTC)
        ts_ms = int(ts.timestamp() * 1000)
        bucket_ts_ms = (ts_ms // target_ms) * target_ms
        current = buckets.get(bucket_ts_ms)
        open_ = Decimal(str(getattr(bar, "open", 0.0)))
        high = Decimal(str(getattr(bar, "high", 0.0)))
        low = Decimal(str(getattr(bar, "low", 0.0)))
        close = Decimal(str(getattr(bar, "close", 0.0)))
        volume = Decimal(str(getattr(bar, "volume", 0.0)))
        if current is None:
            buckets[bucket_ts_ms] = {
                "timestamp": datetime.fromtimestamp(bucket_ts_ms / 1000.0, tz=UTC),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
            continue
        current["high"] = max(current["high"], high)
        current["low"] = min(current["low"], low)
        current["close"] = close
        current["volume"] = current["volume"] + volume

    ordered_buckets = [buckets[key] for key in sorted(buckets)]
    if len(ordered_buckets) > target_bars:
        ordered_buckets = ordered_buckets[-target_bars:]
    return [
        type(ordered[0])(
            timestamp=item["timestamp"],
            open=item["open"],
            high=item["high"],
            low=item["low"],
            close=item["close"],
            volume=item["volume"],
        )
        for item in ordered_buckets
    ]


def _fallback_source_request(
    *,
    market: str,
    timeframe: str,
    target_bars: int,
) -> tuple[str, int] | None:
    normalized_market = str(market).strip().lower()
    normalized_tf = _normalize_timeframe(timeframe)
    if normalized_market == "crypto" and normalized_tf == "4h":
        needed = max(1, int(target_bars)) * 4
        return (
            "1h",
            min(int(settings.market_data_ring_capacity_aggregated), needed),
        )
    return None


def _provider_window_bar_cap(market: str, timeframe: str) -> int | None:
    normalized_market = str(market).strip().lower()
    normalized_tf = _normalize_timeframe(timeframe)
    if normalized_market != "crypto" or normalized_tf not in {"1h", "4h"}:
        return None
    step = _timeframe_step(normalized_tf)
    if step is None:
        return None
    seven_days = int(timedelta(days=7).total_seconds())
    step_seconds = max(1, int(step.total_seconds()))
    return max(1, (seven_days // step_seconds) - 1)


def _history_target_bars(timeframe: str, min_bars: int | None = None) -> int:
    normalized_tf = _normalize_timeframe(timeframe) or "1m"
    requested = max(
        int(settings.market_data_history_target_bars),
        int(min_bars or 0),
    )
    capacity = (
        int(settings.market_data_ring_capacity_1m)
        if normalized_tf == "1m"
        else int(settings.market_data_ring_capacity_aggregated)
    )
    return max(10, min(capacity, requested))


def _history_timeframes(requested_timeframe: str | None = None) -> tuple[str, ...]:
    rows: list[str] = ["1m"]
    rows.extend(
        timeframe.strip().lower()
        for timeframe in settings.market_data_aggregate_timeframes_csv.split(",")
        if timeframe.strip()
    )
    normalized_requested = _normalize_timeframe(requested_timeframe)
    if normalized_requested is not None:
        rows.append(normalized_requested)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in rows:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


async def _fetch_timeframe_history_direct(
    *,
    provider: AlpacaRestProvider,
    market: str,
    symbol: str,
    timeframe: str,
    target_bars: int,
) -> list[Any]:
    step = _timeframe_step(timeframe)
    if step is None:
        return []

    chunk_bars = max(
        10,
        min(int(settings.market_data_history_warmup_chunk_bars), int(target_bars)),
    )
    provider_window_cap = _provider_window_bar_cap(market, timeframe)
    effective_window_bars = chunk_bars
    if provider_window_cap is not None:
        effective_window_bars = max(1, min(effective_window_bars, provider_window_cap))
    max_windows = max(1, int(ceil(max(1, target_bars) / float(effective_window_bars))) + 2)
    cursor_end = datetime.now(UTC)
    collected: dict[datetime, Any] = {}

    for _ in range(max_windows):
        if len(collected) >= target_bars:
            break
        remaining = max(1, target_bars - len(collected))
        window_bars = min(effective_window_bars, remaining)
        window_start = cursor_end - (step * window_bars)
        rows = await provider.fetch_recent_bars(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            since=window_start,
            until=cursor_end,
            limit=window_bars,
        )
        if not rows:
            break
        for bar in rows:
            collected[bar.timestamp] = bar
        earliest = min((bar.timestamp for bar in rows), default=None)
        if earliest is None:
            break
        next_cursor_end = earliest - step
        if next_cursor_end >= cursor_end:
            break
        cursor_end = next_cursor_end

    ordered_timestamps = sorted(collected)
    if not ordered_timestamps:
        return []
    if len(ordered_timestamps) > target_bars:
        ordered_timestamps = ordered_timestamps[-target_bars:]
    return [collected[timestamp] for timestamp in ordered_timestamps]


async def _fetch_timeframe_history(
    *,
    provider: AlpacaRestProvider,
    market: str,
    symbol: str,
    timeframe: str,
    target_bars: int,
) -> list[Any]:
    direct_rows = await _fetch_timeframe_history_direct(
        provider=provider,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        target_bars=target_bars,
    )
    if _bars_are_fresh(
        market=market,
        timeframe=timeframe,
        bars=direct_rows,
    ):
        return direct_rows

    fallback = _fallback_source_request(
        market=market,
        timeframe=timeframe,
        target_bars=target_bars,
    )
    if fallback is None:
        return direct_rows

    source_timeframe, source_target_bars = fallback
    source_rows = await _fetch_timeframe_history_direct(
        provider=provider,
        market=market,
        symbol=symbol,
        timeframe=source_timeframe,
        target_bars=source_target_bars,
    )
    if not source_rows:
        return direct_rows

    derived_rows = _aggregate_bars_from_source(
        source_bars=source_rows,
        source_timeframe=source_timeframe,
        target_timeframe=timeframe,
        target_bars=target_bars,
    )
    if not derived_rows:
        return direct_rows

    logger.warning(
        "[market-data-worker] using fallback aggregation market=%s symbol=%s timeframe=%s source_timeframe=%s direct_bars=%s derived_bars=%s",
        market,
        symbol,
        timeframe,
        source_timeframe,
        len(direct_rows),
        len(derived_rows),
    )
    return derived_rows


async def _ensure_symbol_history_once(
    *,
    market: str,
    symbol: str,
    requested_timeframe: str | None = None,
    min_bars: int | None = None,
) -> dict[str, Any]:
    provider = AlpacaRestProvider()
    try:
        summary: dict[str, Any] = {}
        for timeframe in _history_timeframes(requested_timeframe):
            target_bars = _history_target_bars(
                timeframe,
                min_bars=min_bars if timeframe == _normalize_timeframe(requested_timeframe) else None,
            )
            existing = market_data_runtime.get_recent_bars(
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                limit=target_bars,
            )
            if _bars_are_ready(
                market=market,
                timeframe=timeframe,
                bars=existing,
                target_bars=target_bars,
            ):
                restored = market_data_runtime.restore_bars(
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=existing,
                )
                summary[timeframe] = {
                    "status": "ready",
                    "bars": len(restored),
                    "target_bars": target_bars,
                }
                continue

            try:
                fetched = await _fetch_timeframe_history(
                    provider=provider,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    target_bars=target_bars,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[market-data-worker] history warmup failed market=%s symbol=%s timeframe=%s error=%s",
                    market,
                    symbol,
                    timeframe,
                    type(exc).__name__,
                )
                await _record_market_data_error_event(
                    market=market,
                    symbol=symbol,
                    endpoint="fetch_recent_bars",
                    exc=exc,
                    metadata={"timeframe": timeframe, "target_bars": target_bars},
                )
                summary[timeframe] = {
                    "status": "error",
                    "bars": len(existing),
                    "target_bars": target_bars,
                }
                continue

            hydrated = market_data_runtime.hydrate_bars(
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                bars=fetched,
            )
            summary[timeframe] = {
                "status": "hydrated" if hydrated else "empty",
                "bars": len(hydrated),
                "target_bars": target_bars,
            }
        return summary
    finally:
        await provider.aclose()


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


async def _refresh_symbol_once(
    *,
    market: str,
    symbol: str,
    requested_timeframe: str | None = None,
    min_bars: int | None = None,
) -> dict[str, Any]:
    history_summary = await _ensure_symbol_history_once(
        market=market,
        symbol=symbol,
        requested_timeframe=requested_timeframe,
        min_bars=min_bars,
    )

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
            "history": history_summary,
        }
    finally:
        await provider.aclose()


@celery_app.task(name="market_data.refresh_symbol")
def refresh_symbol_task(
    market: str,
    symbol: str,
    requested_timeframe: str | None = None,
    min_bars: int | None = None,
) -> dict[str, Any]:
    """Refresh latest quote and 1m bar for one symbol."""
    logger.info(
        "[market-data-worker] refresh symbol market=%s symbol=%s requested_timeframe=%s min_bars=%s",
        market,
        symbol,
        requested_timeframe,
        min_bars,
    )
    return asyncio.run(
        _refresh_symbol_once(
            market=market,
            symbol=symbol,
            requested_timeframe=requested_timeframe,
            min_bars=min_bars,
        )
    )


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
        hydrated = market_data_runtime.hydrate_bars(
            market=market,
            symbol=symbol,
            timeframe="1m",
            bars=bars,
        )
        return {
            "market": market,
            "symbol": symbol,
            "status": "ok" if hydrated else "empty",
            "bars": len(hydrated),
            "errors": 0,
        }
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


def enqueue_market_data_refresh(
    *,
    market: str,
    symbol: str,
    requested_timeframe: str | None = None,
    min_bars: int | None = None,
) -> str:
    result = refresh_symbol_task.apply_async(
        args=(market, symbol, requested_timeframe, min_bars),
        queue="market_data",
    )
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
