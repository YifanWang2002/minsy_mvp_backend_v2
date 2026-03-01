"""Market-data MCP tools."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any
from uuid import UUID

import httpx
from mcp.server.fastmcp import Context, FastMCP

from apps.mcp.auth.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from apps.mcp.common.utils import log_mcp_tool_result, to_json, utc_now_iso
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.local_coverage import (
    LocalCoverageInputError,
    deserialize_missing_ranges,
    detect_missing_ranges,
    serialize_missing_ranges,
)
from packages.domain.market_data.sync_service import (
    MarketDataNoMissingDataError,
    MarketDataProviderUnavailableError,
    MarketDataSyncInputError,
    MarketDataSyncJobNotFoundError,
    create_market_data_sync_job,
    get_market_data_sync_job_view,
    schedule_market_data_sync_job,
)
from packages.infra.db import session as db_module
from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.trading.adapters.base import OhlcvBar, QuoteSnapshot
from packages.shared_settings.schema.settings import settings

TOOL_NAMES: tuple[str, ...] = (
    "check_symbol_available",
    "get_available_symbols",
    "get_symbol_data_coverage",
    "get_symbol_candles",
    "get_symbol_metadata",
    "market_data_detect_missing_ranges",
    "market_data_fetch_missing_ranges",
    "market_data_get_sync_job",
    # Backward compatibility aliases.
    "market_data_get_candles",
)

_INTERVAL_TO_ALPACA_TIMEFRAME: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "60m": "1Hour",
    "1h": "1Hour",
    "1d": "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
}
VALID_INTERVALS = tuple(_INTERVAL_TO_ALPACA_TIMEFRAME)
VALID_PERIODS = (
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
)
_ALPACA_SUPPORTED_MARKETS: frozenset[str] = frozenset({"us_stocks", "crypto"})

_MARKET_ALIASES: dict[str, str] = {
    "stock": "us_stocks",
    "stocks": "us_stocks",
    "us_stock": "us_stocks",
    "us_stocks": "us_stocks",
    "crypto": "crypto",
    "forex": "forex",
    "futures": "futures",
}

@lru_cache(maxsize=1)
def _get_data_loader() -> DataLoader:
    return DataLoader()


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "category": "market_data",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    resolved_error_code: str | None = None
    resolved_error_message: str | None = None

    if data:
        payload.update(data)
    if context:
        payload["context"] = context
    if not ok:
        resolved_error_code = error_code or "UNKNOWN_ERROR"
        resolved_error_message = error_message or "Unknown error"
        payload["error"] = {
            "code": resolved_error_code,
            "message": resolved_error_message,
        }

    log_mcp_tool_result(
        category="market_data",
        tool=tool,
        ok=ok,
        error_code=resolved_error_code,
        error_message=resolved_error_message,
    )
    return to_json(payload)


def _build_success_payload(tool: str, data: dict[str, Any]) -> str:
    return _payload(tool=tool, ok=True, data=data)


def _build_error_payload(
    *,
    tool: str,
    error_code: str,
    error_message: str,
    context: dict[str, Any] | None = None,
) -> str:
    return _payload(
        tool=tool,
        ok=False,
        error_code=error_code,
        error_message=error_message,
        context=context,
    )


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(str(value).strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def _parse_datetime_field(value: str, field_name: str) -> datetime:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{field_name} cannot be empty")
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


def _resolve_context_claims(ctx: Context | None) -> McpContextClaims | None:
    if ctx is None:
        return None
    try:
        request = ctx.request_context.request
    except Exception:  # noqa: BLE001
        return None
    headers = getattr(request, "headers", None)
    token = extract_mcp_context_token(headers)
    if token is None:
        return None
    return decode_mcp_context_token(token)


def _serialize_sync_view(view: Any) -> dict[str, Any]:
    return {
        "sync_job_id": str(view.job_id),
        "provider": view.provider,
        "market": view.market,
        "symbol": view.symbol,
        "timeframe": view.timeframe,
        "status": view.status,
        "progress": view.progress,
        "current_step": view.current_step,
        "requested_start": view.requested_start.isoformat(),
        "requested_end": view.requested_end.isoformat(),
        "missing_ranges": serialize_missing_ranges(list(view.missing_ranges)),
        "rows_written": view.rows_written,
        "range_filled": view.range_filled,
        "total_ranges": view.total_ranges,
        "errors": list(view.errors),
        "submitted_at": view.submitted_at.isoformat(),
        "completed_at": view.completed_at.isoformat() if view.completed_at else None,
    }


def _timeframe_to_seconds(timeframe: str) -> int | None:
    normalized = str(timeframe).strip().lower()
    if normalized == "1m":
        return 60
    if normalized == "5m":
        return 300
    if normalized == "15m":
        return 900
    if normalized in {"60m", "1h"}:
        return 3600
    if normalized == "1d":
        return 86400
    if normalized == "1wk":
        return 604800
    if normalized == "1mo":
        return 2592000
    return None


def _estimate_sync_wait_seconds(
    *,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
    missing_ranges: list[dict[str, Any]],
) -> int | None:
    timeframe_seconds = _timeframe_to_seconds(timeframe)
    if timeframe_seconds is None:
        return None

    requested_span_seconds = max(
        1,
        int((_ensure_utc(requested_end) - _ensure_utc(requested_start)).total_seconds()),
    )
    expected_bars = max(1, requested_span_seconds // timeframe_seconds)
    range_count = max(1, len(missing_ranges))
    # Heuristic: for typical Alpaca sync jobs this yields a practical
    # waiting estimate without exposing worker internals.
    estimate = int((expected_bars * 0.03) + (range_count * 8))
    return max(15, min(estimate, 900))


def _recommended_poll_interval_seconds(estimated_wait_seconds: int | None) -> int:
    if estimated_wait_seconds is None:
        return 10
    if estimated_wait_seconds <= 60:
        return 5
    if estimated_wait_seconds <= 180:
        return 10
    if estimated_wait_seconds <= 600:
        return 20
    return 30


def _build_market_input_context(
    *,
    market: str,
    symbol: str,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {"market": market, "symbol": symbol}
    if isinstance(extra_context, dict) and extra_context:
        context.update(extra_context)
    return context


def _normalize_market(market: str) -> str:
    market_key = market.strip().lower()
    normalized = _MARKET_ALIASES.get(market_key)
    if not normalized:
        raise ValueError(
            f"Unsupported market '{market}'. "
            "Use one of: stock/us_stocks, crypto, forex, futures."
        )
    return normalized


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol cannot be empty")
    return normalized


def _normalize_sync_provider(provider: str) -> str:
    requested = provider.strip().lower()
    if requested in {"", "default", "local_parquet"}:
        return "alpaca"
    return requested


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _datetime_to_utc_z(value: datetime) -> str:
    return _ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_alpaca_market(market: str) -> None:
    if market in _ALPACA_SUPPORTED_MARKETS:
        return
    raise ValueError(
        f"Market '{market}' is not supported by Alpaca in this tool. "
        "Use one of: stock/us_stocks, crypto."
    )


def _to_alpaca_timeframe(interval: str) -> str:
    normalized = interval.strip().lower()
    timeframe = _INTERVAL_TO_ALPACA_TIMEFRAME.get(normalized)
    if timeframe is None:
        raise ValueError(
            f"Invalid interval '{interval}'. Supported intervals: {sorted(VALID_INTERVALS)}"
        )
    return timeframe


def _period_start(*, period: str, anchor: datetime) -> datetime:
    normalized = period.strip().lower()
    if normalized not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'")
    if normalized == "ytd":
        return datetime(anchor.year, 1, 1, tzinfo=UTC)
    if normalized == "max":
        return anchor - timedelta(days=365 * 10)

    mapping: dict[str, timedelta] = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=365 * 2),
        "5y": timedelta(days=365 * 5),
        "10y": timedelta(days=365 * 10),
    }
    return anchor - mapping[normalized]


def _resolve_candle_window(
    *,
    period: str,
    start: str | None,
    end: str | None,
) -> tuple[datetime, datetime | None]:
    resolved_end = _parse_datetime_field(end, "end") if end else None

    if start:
        resolved_start = _parse_datetime_field(start, "start")
        if resolved_end is not None and resolved_end < resolved_start:
            raise ValueError("end must be greater than or equal to start")
        return resolved_start, resolved_end

    anchor = resolved_end or datetime.now(UTC)
    resolved_start = _period_start(period=period, anchor=anchor)
    return resolved_start, resolved_end


def _bars_to_records(
    *,
    bars: list[OhlcvBar],
    end: datetime | None = None,
) -> list[dict[str, Any]]:
    bound = _ensure_utc(end) if end is not None else None
    records: list[dict[str, Any]] = []
    ordered = sorted(bars, key=lambda item: _ensure_utc(item.timestamp))
    for bar in ordered:
        ts = _ensure_utc(bar.timestamp)
        if bound is not None and ts > bound:
            continue
        records.append(
            {
                "datetime": _datetime_to_utc_z(ts),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    return records


def _metadata_from_quote(quote: QuoteSnapshot) -> dict[str, Any]:
    return {
        "bid": float(quote.bid) if quote.bid is not None else None,
        "ask": float(quote.ask) if quote.ask is not None else None,
        "last": float(quote.last) if quote.last is not None else None,
        "timestamp_utc": _datetime_to_utc_z(quote.timestamp),
    }


def _metadata_from_bar(bar: OhlcvBar) -> dict[str, Any]:
    return {
        "last": float(bar.close),
        "timestamp_utc": _datetime_to_utc_z(bar.timestamp),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response is not None and response.status_code == 429:
            return True
    error_text = str(exc).lower()
    hints = (
        "too many requests",
        "rate limit",
        "status=429",
        "status code=429",
        "response code=429",
        "http 429",
        "429 client error",
    )
    return any(hint in error_text for hint in hints)


def _symbol_available_in_market(
    *,
    loader: DataLoader,
    market_key: str,
    symbol_key: str,
) -> bool:
    return symbol_key in set(loader.get_available_symbols(market_key))


def check_symbol_available(symbol: str, market: str = "") -> str:
    """
    Check whether a symbol is available.

    Provide `symbol` and optionally `market`.
    """
    loader = _get_data_loader()
    symbol_key = symbol.strip().upper()
    if not symbol_key:
        return _build_error_payload(
            tool="check_symbol_available",
            error_code="INVALID_INPUT",
            error_message="symbol cannot be empty",
            context={"symbol": symbol, "market": market},
        )

    try:
        if market.strip():
            market_key = loader.normalize_market(market)
            available = _symbol_available_in_market(
                loader=loader,
                market_key=market_key,
                symbol_key=symbol_key,
            )
            return _build_success_payload(
                tool="check_symbol_available",
                data={
                    "symbol": symbol_key,
                    "market": market_key,
                    "available": available,
                },
            )

        matched_markets: list[str] = []
        for candidate in ("us_stocks", "crypto", "forex", "futures"):
            if _symbol_available_in_market(
                loader=loader,
                market_key=candidate,
                symbol_key=symbol_key,
            ):
                matched_markets.append(candidate)
        return _build_success_payload(
            tool="check_symbol_available",
            data={
                "symbol": symbol_key,
                "available": bool(matched_markets),
                "matched_markets": matched_markets,
            },
        )
    except ValueError as exc:
        return _build_error_payload(
            tool="check_symbol_available",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context={"symbol": symbol, "market": market},
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="check_symbol_available",
            error_code="SYMBOL_LOOKUP_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
            context={"symbol": symbol, "market": market},
        )


def get_available_symbols(market: str) -> str:
    """Get local symbols by market. Provide `market`."""
    loader = _get_data_loader()
    try:
        market_key = loader.normalize_market(market)
        symbols = loader.get_available_symbols(market_key)
        return _build_success_payload(
            tool="get_available_symbols",
            data={
                "market": market_key,
                "count": len(symbols),
                "symbols": symbols,
            },
        )
    except ValueError as exc:
        return _build_error_payload(
            tool="get_available_symbols",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context={"market": market},
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_available_symbols",
            error_code="SYMBOL_LIST_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
            context={"market": market},
        )


def get_symbol_data_coverage(market: str, symbol: str) -> str:
    """
    Get local parquet metadata for a symbol.

    Provide `symbol` and `market`.
    """
    loader = _get_data_loader()
    try:
        metadata = loader.get_symbol_metadata(market, symbol)
        return _build_success_payload(
            tool="get_symbol_data_coverage",
            data={
                "market": metadata["market"],
                "symbol": metadata["symbol"],
                "metadata": metadata,
            },
        )
    except ValueError as exc:
        return _build_error_payload(
            tool="get_symbol_data_coverage",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context={"market": market, "symbol": symbol},
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_symbol_data_coverage",
            error_code="COVERAGE_LOOKUP_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
            context={"market": market, "symbol": symbol},
        )




async def get_symbol_candles(
    symbol: str,
    market: str,
    period: str = "1d",
    interval: str = "1d",
    limit: int = 300,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """
    Get OHLCV candles for a symbol.

    Provide `symbol` and `market`.
    """
    context = _build_market_input_context(
        market=market,
        symbol=symbol,
        extra_context={
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
        },
    )

    try:
        market_key = _normalize_market(market)
        _validate_alpaca_market(market_key)
        symbol_key = _normalize_symbol(symbol)
        timeframe = _to_alpaca_timeframe(interval)
        requested_start, requested_end = _resolve_candle_window(
            period=period,
            start=start,
            end=end,
        )
        safe_limit = max(1, min(int(limit), 2000))
    except ValueError as exc:
        return _build_error_payload(
            tool="get_symbol_candles",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context=context,
        )

    client = AlpacaMarketDataClient()
    try:
        bars = await client.fetch_ohlcv(
            symbol=symbol_key,
            market=market_key,
            timeframe=timeframe,
            since=requested_start,
            limit=safe_limit,
        )
    except Exception as exc:  # noqa: BLE001
        code = "UPSTREAM_RATE_LIMIT" if _is_rate_limit_error(exc) else "CANDLES_FETCH_ERROR"
        return _build_error_payload(
            tool="get_symbol_candles",
            error_code=code,
            error_message=f"{type(exc).__name__}: {exc}",
            context=context,
        )
    finally:
        await client.aclose()

    records = _bars_to_records(
        bars=bars,
        end=requested_end,
    )
    if not records:
        return _build_error_payload(
            tool="get_symbol_candles",
            error_code="CANDLES_NOT_FOUND",
            error_message=f"No candles found for {symbol_key}",
            context=context,
        )

    truncated = len(records) > safe_limit
    if truncated:
        records = records[-safe_limit:]
    return _build_success_payload(
        tool="get_symbol_candles",
        data={
            "market": market_key,
            "symbol": symbol_key,
            "provider": "alpaca",
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
            "rows": len(records),
            "truncated": truncated,
            "candles": records,
        },
    )


async def get_symbol_metadata(symbol: str, market: str) -> str:
    """
    Get latest symbol metadata from Alpaca.

    Provide `symbol` and `market`.
    """
    context = _build_market_input_context(market=market, symbol=symbol)
    try:
        market_key = _normalize_market(market)
        _validate_alpaca_market(market_key)
        symbol_key = _normalize_symbol(symbol)
    except ValueError as exc:
        return _build_error_payload(
            tool="get_symbol_metadata",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context=context,
        )

    client = AlpacaMarketDataClient()
    try:
        quote = await client.fetch_latest_quote(symbol_key, market=market_key)
        if quote is not None:
            return _build_success_payload(
                tool="get_symbol_metadata",
                data={
                    "market": market_key,
                    "symbol": symbol_key,
                    "provider": "alpaca",
                    "metadata_source": "latest_quote",
                    "metadata": _metadata_from_quote(quote),
                },
            )

        latest_bar = await client.fetch_latest_bar(symbol_key, market=market_key)
        if latest_bar is None:
            return _build_error_payload(
                tool="get_symbol_metadata",
                error_code="METADATA_NOT_FOUND",
                error_message=f"No metadata found for {symbol_key}",
                context=context,
            )

        return _build_success_payload(
            tool="get_symbol_metadata",
            data={
                "market": market_key,
                "symbol": symbol_key,
                "provider": "alpaca",
                "metadata_source": "latest_bar",
                "metadata": _metadata_from_bar(latest_bar),
            },
        )
    except Exception as exc:  # noqa: BLE001
        code = "UPSTREAM_RATE_LIMIT" if _is_rate_limit_error(exc) else "METADATA_FETCH_ERROR"
        return _build_error_payload(
            tool="get_symbol_metadata",
            error_code=code,
            error_message=f"{type(exc).__name__}: {exc}",
            context=context,
        )
    finally:
        await client.aclose()


def market_data_detect_missing_ranges(
    market: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> str:
    """Detect local missing timestamp ranges for one symbol/timeframe window."""

    loader = _get_data_loader()
    try:
        report = detect_missing_ranges(
            loader=loader,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start=_parse_datetime_field(start_date, "start_date"),
            end=_parse_datetime_field(end_date, "end_date"),
        )
    except (ValueError, LocalCoverageInputError) as exc:
        return _build_error_payload(
            tool="market_data_detect_missing_ranges",
            error_code="INVALID_RANGE",
            error_message=str(exc),
            context={
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="market_data_detect_missing_ranges",
            error_code="COVERAGE_LOOKUP_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
            context={
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    return _build_success_payload(
        tool="market_data_detect_missing_ranges",
        data={
            "market": report.market,
            "symbol": report.symbol,
            "timeframe": report.timeframe,
            "requested_start": report.start.isoformat(),
            "requested_end": report.end.isoformat(),
            "expected_bars": report.expected_bars,
            "present_bars": report.present_bars,
            "missing_bars": report.missing_bars,
            "missing_ranges": serialize_missing_ranges(list(report.missing_ranges)),
            "local_coverage_pct": report.local_coverage_pct,
        },
    )


async def market_data_fetch_missing_ranges(
    provider: str,
    market: str,
    symbol: str,
    timeframe: str,
    start_date: str = "",
    end_date: str = "",
    missing_ranges: list[dict[str, Any]] | None = None,
    max_lookback_days: int = 0,
    run_async: bool = True,
    ctx: Context | None = None,
) -> str:
    """Create and optionally execute one missing-range sync job."""

    try:
        claims = _resolve_context_claims(ctx)
    except ValueError as exc:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    if not run_async:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="INVALID_INPUT",
            error_message=(
                "run_async=false is no longer supported. "
                "market_data_fetch_missing_ranges always queues a background job."
            ),
        )

    try:
        if start_date.strip() and end_date.strip():
            requested_start = _parse_datetime_field(start_date, "start_date")
            requested_end = _parse_datetime_field(end_date, "end_date")
        else:
            lookback_days = max(int(max_lookback_days), 0) or settings.market_data_sync_default_lookback_days
            requested_end = datetime.now(UTC)
            requested_start = requested_end - timedelta(days=lookback_days)
    except ValueError as exc:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="INVALID_RANGE",
            error_message=str(exc),
            context={
                "start_date": start_date,
                "end_date": end_date,
                "max_lookback_days": max_lookback_days,
            },
        )

    normalized_ranges = serialize_missing_ranges(deserialize_missing_ranges(missing_ranges or []))
    normalized_provider = _normalize_sync_provider(provider)
    estimated_wait_seconds = _estimate_sync_wait_seconds(
        timeframe=timeframe,
        requested_start=requested_start,
        requested_end=requested_end,
        missing_ranges=list(normalized_ranges),
    )
    recommended_poll_interval_seconds = _recommended_poll_interval_seconds(
        estimated_wait_seconds
    )

    try:
        async with await _new_db_session() as db:
            receipt = await create_market_data_sync_job(
                db,
                provider=normalized_provider,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                requested_start=requested_start,
                requested_end=requested_end,
                missing_ranges=normalized_ranges or None,
                user_id=claims.user_id if claims is not None else None,
                auto_commit=True,
            )

            queued_task_id: str | None = None
            if not receipt.deduplicated:
                queued_task_id = await schedule_market_data_sync_job(receipt.job_id)
            view = await get_market_data_sync_job_view(
                db,
                job_id=receipt.job_id,
                user_id=claims.user_id if claims is not None else None,
            )

            data = _serialize_sync_view(view)
            data["queued_task_id"] = queued_task_id
            data["run_async"] = True
            data["provider_requested"] = provider
            data["estimated_wait_seconds"] = estimated_wait_seconds
            data["recommended_poll_interval_seconds"] = recommended_poll_interval_seconds
            # Alias kept for prompt backward compatibility.
            data["recommended_next_poll_seconds"] = recommended_poll_interval_seconds
            return _build_success_payload(
                tool="market_data_fetch_missing_ranges",
                data=data,
            )
    except MarketDataNoMissingDataError as exc:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="NO_MISSING_DATA",
            error_message=str(exc),
            context={
                "provider": normalized_provider,
                "provider_requested": provider,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        )
    except MarketDataProviderUnavailableError as exc:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="PROVIDER_UNAVAILABLE",
            error_message=str(exc),
            context={
                "provider": normalized_provider,
                "provider_requested": provider,
            },
        )
    except (MarketDataSyncInputError, ValueError) as exc:
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="INVALID_RANGE",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="market_data_fetch_missing_ranges",
            error_code="MARKET_DATA_SYNC_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )


async def market_data_get_sync_job(
    sync_job_id: str,
    ctx: Context | None = None,
) -> str:
    """Get one market-data sync job state/result view."""

    try:
        job_uuid = _parse_uuid(sync_job_id, "sync_job_id")
        claims = _resolve_context_claims(ctx)
    except ValueError as exc:
        return _build_error_payload(
            tool="market_data_get_sync_job",
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            view = await get_market_data_sync_job_view(
                db,
                job_id=job_uuid,
                user_id=claims.user_id if claims is not None else None,
            )
    except MarketDataSyncJobNotFoundError as exc:
        return _build_error_payload(
            tool="market_data_get_sync_job",
            error_code="NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="market_data_get_sync_job",
            error_code="MARKET_DATA_SYNC_GET_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _build_success_payload(
        tool="market_data_get_sync_job",
        data=_serialize_sync_view(view),
    )




async def market_data_get_candles(
    symbol: str,
    interval: str = "1d",
    limit: int = 30,
) -> str:
    """Backward compatibility alias for candles tool."""
    return await get_symbol_candles(
        symbol=symbol,
        market="us_stocks",
        period="1mo",
        interval=interval,
        limit=limit,
    )


def register_market_data_tools(mcp: FastMCP) -> None:
    """Register market-data-related tools."""
    for tool in (
        check_symbol_available,
        get_available_symbols,
        get_symbol_data_coverage,
        get_symbol_candles,
        get_symbol_metadata,
        market_data_detect_missing_ranges,
        market_data_fetch_missing_ranges,
        market_data_get_sync_job,
        market_data_get_candles,
    ):
        mcp.tool()(tool)
