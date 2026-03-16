"""Market-data MCP tools."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any
from uuid import UUID

import httpx
import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

from apps.mcp.auth.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from apps.mcp.common.utils import log_mcp_tool_result, to_json, utc_now_iso
from apps.mcp.domains.market_data.regime_render import (
    build_chart_alt_text,
    render_candlestick_image,
)
from apps.mcp.domains.market_data.regime_runtime_cache import (
    get_snapshot as get_regime_snapshot_from_cache,
)
from apps.mcp.domains.market_data.regime_runtime_cache import (
    put_snapshot as put_regime_snapshot_in_cache,
)
from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.local_coverage import (
    LocalCoverageInputError,
    deserialize_missing_ranges,
    detect_missing_ranges,
    serialize_missing_ranges,
)
from packages.domain.market_data.regime import (
    SUPPORTED_REGIME_TIMEFRAMES,
    build_regime_feature_snapshot,
    map_pre_strategy_timeframes,
    score_strategy_families,
)
from packages.domain.market_data.regime.family_scoring import (
    build_family_option_subtitles,
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
    "pre_strategy_get_regime_snapshot",
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
_PRE_STRATEGY_OPPORTUNITY_BUCKETS: frozenset[str] = frozenset(
    {"few_per_month", "few_per_week", "daily", "multiple_per_day"}
)
_PRE_STRATEGY_HOLDING_BUCKETS: frozenset[str] = frozenset(
    {"intraday_scalp", "intraday", "swing_days", "position_weeks_plus"}
)

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


def _bars_to_dataframe(*, bars: list[OhlcvBar], end: datetime | None = None) -> pd.DataFrame:
    records = _bars_to_records(bars=bars, end=end)
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame.from_records(records)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.set_index("datetime").sort_index()
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def _resample_ohlcv_frame(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    normalized = str(timeframe).strip().lower()
    if frame.empty:
        return frame
    if normalized not in {"4h"}:
        return frame
    return (
        frame.resample("4h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def _normalize_regime_timeframe(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"60m"}:
        normalized = "1h"
    if normalized not in set(SUPPORTED_REGIME_TIMEFRAMES):
        raise ValueError(
            "Unsupported timeframe for regime analysis. "
            f"Use one of: {list(SUPPORTED_REGIME_TIMEFRAMES)}"
        )
    return normalized


def _resolve_local_available_start(
    *,
    loader: DataLoader,
    market: str,
    symbol: str,
) -> datetime | None:
    try:
        metadata = loader.get_symbol_metadata(market, symbol)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(metadata, dict):
        return None
    timerange = metadata.get("available_timerange")
    if not isinstance(timerange, dict):
        return None
    start_raw = timerange.get("start")
    if not isinstance(start_raw, str) or not start_raw.strip():
        return None
    try:
        return _parse_datetime_field(start_raw, "available_start")
    except ValueError:
        return None


async def _fetch_alpaca_regime_frame(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    end_utc: datetime,
) -> pd.DataFrame:
    _validate_alpaca_market(market)
    requested = _normalize_regime_timeframe(timeframe)
    if requested == "4h":
        base_timeframe = "1Hour"
        safe_limit = max(200, min(lookback_bars * 4 + 100, 2000))
        seconds = max(_timeframe_to_seconds("1h") or 3600, 60)
    else:
        base_timeframe = _to_alpaca_timeframe(requested)
        safe_limit = max(50, min(lookback_bars + 100, 2000))
        seconds = max(_timeframe_to_seconds(requested) or 60, 60)

    since = end_utc - timedelta(seconds=safe_limit * seconds * 2)
    client = AlpacaMarketDataClient()
    try:
        bars = await client.fetch_ohlcv(
            symbol=symbol,
            market=market,
            timeframe=base_timeframe,
            since=since,
            limit=safe_limit,
        )
    finally:
        await client.aclose()

    frame = _bars_to_dataframe(bars=bars, end=end_utc)
    if requested == "4h":
        frame = _resample_ohlcv_frame(frame, requested)
    if len(frame) > lookback_bars:
        frame = frame.tail(lookback_bars)
    return frame


def _load_local_regime_frame(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    end_utc: datetime,
) -> pd.DataFrame:
    loader = _get_data_loader()
    normalized_timeframe = _normalize_regime_timeframe(timeframe)
    # For sparse sessions (notably RTH equities), fixed windows can under-fetch,
    # so we expand the local lookback progressively until bars are sufficient
    # or we hit the earliest known local timestamp.
    source_timeframe = "1h" if normalized_timeframe == "4h" else normalized_timeframe
    seconds = _timeframe_to_seconds(source_timeframe) or 60
    base_multiplier = 8 if normalized_timeframe == "4h" else 4
    history_span = timedelta(seconds=max(seconds * lookback_bars * base_multiplier, 60 * 60))
    available_start = _resolve_local_available_start(
        loader=loader,
        market=market,
        symbol=symbol,
    )

    best_frame = pd.DataFrame()
    last_error: Exception | None = None

    for _attempt in range(8):
        start_utc = end_utc - history_span
        if available_start is not None and start_utc < available_start:
            start_utc = available_start

        try:
            source_frame = loader.load(
                market=market,
                symbol=symbol,
                timeframe=source_timeframe,
                start_date=start_utc,
                end_date=end_utc,
            )
        except (FileNotFoundError, ValueError) as exc:
            last_error = exc
            if available_start is not None and start_utc <= available_start:
                break
            history_span *= 2
            continue

        if normalized_timeframe == "4h":
            frame = _resample_ohlcv_frame(source_frame, "4h")
        else:
            frame = source_frame

        if len(frame) > len(best_frame):
            best_frame = frame
        if len(frame) >= lookback_bars:
            break
        if available_start is not None and start_utc <= available_start:
            break
        history_span *= 2

    if not best_frame.empty:
        if len(best_frame) > lookback_bars:
            return best_frame.tail(lookback_bars)
        return best_frame

    if last_error is not None:
        raise last_error
    raise ValueError(
        f"No local candles found for {symbol} market={market} timeframe={normalized_timeframe}."
    )


async def _resolve_regime_frame(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    end_utc: datetime,
) -> tuple[pd.DataFrame, str, str]:
    local_frame = _load_local_regime_frame(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        end_utc=end_utc,
    )
    return local_frame, "local_primary", "local_parquet"


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


def _build_regime_summary_text(
    *,
    timeframe: str,
    family_scores: dict[str, Any],
    snapshot: dict[str, Any],
) -> str:
    recommended = str(family_scores.get("recommended_family", "trend_continuation")).strip()
    confidence = float(family_scores.get("confidence", 0.0) or 0.0)
    trend = float(family_scores.get("trend_continuation", 0.0) or 0.0)
    reversion = float(family_scores.get("mean_reversion", 0.0) or 0.0)
    volatility = float(family_scores.get("volatility_regime", 0.0) or 0.0)
    trend_block = snapshot.get("trend_reversion", {})
    adx = float(trend_block.get("adx", 0.0) or 0.0)
    chop = float(trend_block.get("chop", 0.0) or 0.0)
    return (
        f"Timeframe {timeframe} shows recommended family={recommended} "
        f"(confidence {confidence:.2f}). "
        f"Scores trend={trend:.2f}, mean_reversion={reversion:.2f}, volatility={volatility:.2f}. "
        f"ADX={adx:.1f}, CHOP={chop:.1f}."
    )


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _clip(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm_01(value: Any, divisor: float) -> float:
    return _clip(_safe_float(value) / max(divisor, 1e-9), low=0.0, high=1.0)


def _signed_tanh(value: Any, scale: float) -> float:
    return _clip(math.tanh(_safe_float(value) / max(scale, 1e-9)), low=-1.0, high=1.0)


def _safe_log_ratio(value: Any) -> float:
    raw = max(_safe_float(value, default=1.0), 1e-9)
    return _safe_float(math.log(raw), default=0.0)


def _normalized_primary_features(snapshot: dict[str, Any]) -> dict[str, Any]:
    window_stats = snapshot.get("window_stats", {})
    price_path = snapshot.get("price_path_summary", {})
    swing = snapshot.get("swing_structure", {})
    noise = snapshot.get("efficiency_noise", {})
    vol_state = snapshot.get("volatility_state", {})
    trend = snapshot.get("trend_reversion", {})
    volume = snapshot.get("volume_participation", {})
    coupling = snapshot.get("volatility_direction_coupling", {})

    return {
        # Cross-asset comparable transforms (bounded/ratio-based where possible).
        "cumulative_return_n": _signed_tanh(
            window_stats.get("cumulative_return"),
            scale=0.2,
        ),
        "recent_return_n": _signed_tanh(
            window_stats.get("recent_return"),
            scale=0.03,
        ),
        "rolling_return_std_n": _signed_tanh(
            window_stats.get("rolling_return_std"),
            scale=0.05,
        ),
        "return_skew_n": _signed_tanh(window_stats.get("return_skew"), scale=2.0),
        "return_kurtosis_n": _signed_tanh(window_stats.get("return_kurtosis"), scale=5.0),
        "max_drawdown_n": _signed_tanh(price_path.get("max_drawdown"), scale=0.2),
        "up_bar_ratio": _clip(
            _safe_float(price_path.get("up_bar_ratio"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "doji_ratio": _clip(
            _safe_float(price_path.get("doji_ratio"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "gap_frequency": _clip(
            _safe_float(price_path.get("gap_frequency"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "breakout_frequency": _clip(
            _safe_float(swing.get("breakout_frequency"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "false_breakout_frequency": _clip(
            _safe_float(swing.get("false_breakout_frequency"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "distance_to_recent_midrange_n": _clip(
            _safe_float(swing.get("distance_to_recent_midrange"), default=0.0),
            low=-1.0,
            high=1.0,
        ),
        "efficiency_ratio": _clip(
            _safe_float(noise.get("efficiency_ratio"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "sign_autocorrelation": _clip(
            _safe_float(noise.get("sign_autocorrelation"), default=0.0),
            low=-1.0,
            high=1.0,
        ),
        "choppiness_n": _norm_01(noise.get("choppiness_index"), 100.0),
        "vol_percentile": _clip(
            _safe_float(vol_state.get("vol_percentile"), default=0.5),
            low=0.0,
            high=1.0,
        ),
        "short_long_vol_log": _safe_log_ratio(vol_state.get("short_long_vol_ratio")),
        "atr_short_long_log": _safe_log_ratio(vol_state.get("atr_short_long_ratio")),
        "squeeze_score": _clip(
            _safe_float(vol_state.get("squeeze_score"), default=0.5),
            low=0.0,
            high=1.0,
        ),
        "volatility_change_n": _signed_tanh(
            vol_state.get("volatility_change_rate"),
            scale=0.5,
        ),
        "volatility_expansion_flag": 1.0
        if _safe_float(vol_state.get("volatility_expansion_flag"), default=0.0) >= 0.5
        else 0.0,
        "volatility_contraction_flag": 1.0
        if _safe_float(vol_state.get("volatility_contraction_flag"), default=0.0) >= 0.5
        else 0.0,
        "adx_n": _norm_01(trend.get("adx"), 100.0),
        "price_zscore_n": _signed_tanh(trend.get("price_zscore"), scale=3.0),
        "bollinger_position_n": _clip(
            _safe_float(trend.get("bollinger_position"), default=0.5),
            low=0.0,
            high=1.0,
        ),
        "rsi_n": _norm_01(trend.get("rsi"), 100.0),
        "distance_from_vwap_n": _clip(
            _safe_float(trend.get("distance_from_vwap"), default=0.0),
            low=-1.0,
            high=1.0,
        ),
        "volume_reliable_flag": 1.0
        if _safe_float(volume.get("volume_reliable_flag"), default=0.0) >= 0.5
        else 0.0,
        "relative_volume_log": _safe_log_ratio(volume.get("relative_volume")),
        "mfi_n": _norm_01(volume.get("mfi"), 100.0),
        "dry_up_reversal_hint": _clip(
            _safe_float(volume.get("dry_up_reversal_hint"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "trend_with_low_vol": bool(coupling.get("trend_with_low_vol", False)),
        "trend_with_expanding_vol": bool(
            coupling.get("trend_with_expanding_vol", False)
        ),
        "range_with_low_vol": bool(coupling.get("range_with_low_vol", False)),
        "panic_reversal_with_high_vol": bool(
            coupling.get("panic_reversal_with_high_vol", False)
        ),
    }


def _normalized_secondary_features(primary_features: dict[str, Any]) -> dict[str, Any]:
    brief_keys: tuple[str, ...] = (
        "adx_n",
        "choppiness_n",
        "efficiency_ratio",
        "price_zscore_n",
        "vol_percentile",
        "short_long_vol_log",
        "squeeze_score",
        "volatility_change_n",
        "breakout_frequency",
        "distance_from_vwap_n",
        "relative_volume_log",
        "max_drawdown_n",
        "distance_to_recent_midrange_n",
    )
    return {
        key: primary_features[key]
        for key in brief_keys
        if key in primary_features
    }


def _compact_family_scores(
    scores_payload: dict[str, Any],
    *,
    include_evidence: bool,
) -> dict[str, Any]:
    compact = {
        "trend_continuation": _clip(
            _safe_float(scores_payload.get("trend_continuation"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "mean_reversion": _clip(
            _safe_float(scores_payload.get("mean_reversion"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "volatility_regime": _clip(
            _safe_float(scores_payload.get("volatility_regime"), default=0.0),
            low=0.0,
            high=1.0,
        ),
        "recommended_family": str(
            scores_payload.get("recommended_family", "trend_continuation")
        ).strip(),
        "confidence": _clip(
            _safe_float(scores_payload.get("confidence"), default=0.0),
            low=0.0,
            high=1.0,
        ),
    }
    if include_evidence:
        evidence_for = scores_payload.get("evidence_for")
        evidence_against = scores_payload.get("evidence_against")
        if isinstance(evidence_for, dict):
            compact["evidence_for"] = evidence_for
        if isinstance(evidence_against, dict):
            compact["evidence_against"] = evidence_against
    return compact


async def pre_strategy_get_regime_snapshot(
    market: str,
    symbol: str,
    opportunity_frequency_bucket: str,
    holding_period_bucket: str,
    lookback_bars: int = 500,
    end_utc: str | None = None,
) -> str:
    """Build pre-strategy regime snapshot for one symbol with two mapped timeframes."""

    context = _build_market_input_context(
        market=market,
        symbol=symbol,
        extra_context={
            "opportunity_frequency_bucket": opportunity_frequency_bucket,
            "holding_period_bucket": holding_period_bucket,
            "lookback_bars": lookback_bars,
            "end_utc": end_utc,
        },
    )
    try:
        market_key = _normalize_market(market)
        symbol_key = _normalize_symbol(symbol)
        frequency_key = str(opportunity_frequency_bucket).strip().lower()
        holding_key = str(holding_period_bucket).strip().lower()
        if frequency_key not in _PRE_STRATEGY_OPPORTUNITY_BUCKETS:
            raise ValueError(
                "Unsupported opportunity_frequency_bucket. "
                f"Use one of: {sorted(_PRE_STRATEGY_OPPORTUNITY_BUCKETS)}"
            )
        if holding_key not in _PRE_STRATEGY_HOLDING_BUCKETS:
            raise ValueError(
                "Unsupported holding_period_bucket. "
                f"Use one of: {sorted(_PRE_STRATEGY_HOLDING_BUCKETS)}"
            )
        requested_lookback = int(lookback_bars)
        if requested_lookback <= 0:
            requested_lookback = int(settings.pre_strategy_regime_lookback_bars)
        safe_lookback = max(
            int(settings.pre_strategy_regime_min_bars),
            min(requested_lookback, 1200),
        )
        resolved_end = (
            _parse_datetime_field(end_utc, "end_utc")
            if isinstance(end_utc, str) and end_utc.strip()
            else datetime.now(UTC)
        )
        timeframe_plan = map_pre_strategy_timeframes(
            opportunity_frequency_bucket=frequency_key,
            holding_period_bucket=holding_key,
        )
    except ValueError as exc:
        return _build_error_payload(
            tool="pre_strategy_get_regime_snapshot",
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context=context,
        )

    by_timeframe: dict[str, dict[str, Any]] = {}
    candles_by_timeframe: dict[str, pd.DataFrame] = {}

    for timeframe in timeframe_plan.candidates:
        try:
            frame, source_mode, source_label = await _resolve_regime_frame(
                market=market_key,
                symbol=symbol_key,
                timeframe=timeframe,
                lookback_bars=safe_lookback,
                end_utc=resolved_end,
            )
        except Exception as exc:  # noqa: BLE001
            return _build_error_payload(
                tool="pre_strategy_get_regime_snapshot",
                error_code="REGIME_DATA_FETCH_ERROR",
                error_message=f"{type(exc).__name__}: {exc}",
                context={**context, "timeframe": timeframe},
            )

        if frame.empty:
            return _build_error_payload(
                tool="pre_strategy_get_regime_snapshot",
                error_code="CANDLES_NOT_FOUND",
                error_message=f"No candles available for {symbol_key} on timeframe={timeframe}.",
                context={**context, "timeframe": timeframe},
            )
        if len(frame) < int(settings.pre_strategy_regime_min_bars):
            return _build_error_payload(
                tool="pre_strategy_get_regime_snapshot",
                error_code="INSUFFICIENT_BARS",
                error_message=(
                    f"Insufficient candles for stable regime analysis. bars={len(frame)} timeframe={timeframe}"
                ),
                context={**context, "timeframe": timeframe},
            )

        snapshot = build_regime_feature_snapshot(
            frame,
            timeframe=timeframe,
            lookback_bars=safe_lookback,
            pivot_window=max(2, int(settings.pre_strategy_regime_pivot_window)),
        )
        scores = score_strategy_families(snapshot)
        scores_payload = scores.to_dict()
        summary = _build_regime_summary_text(
            timeframe=timeframe,
            family_scores=scores_payload,
            snapshot=snapshot,
        )
        subtitles = build_family_option_subtitles(snapshot, scores)
        normalized_primary = _normalized_primary_features(snapshot)
        normalized_secondary = _normalized_secondary_features(normalized_primary)
        payload = {
            "timeframe": timeframe,
            "bars": len(frame),
            "source_mode": source_mode,
            "source_label": source_label,
            "family_scores": scores_payload,
            "summary": summary,
            "choice_option_subtitles": subtitles,
            "normalized_primary_features": normalized_primary,
            "normalized_secondary_features": normalized_secondary,
            "snapshot": snapshot,
        }
        by_timeframe[timeframe] = payload
        candles_by_timeframe[timeframe] = frame

    primary_payload = by_timeframe.get(timeframe_plan.primary)
    secondary_payload = by_timeframe.get(timeframe_plan.secondary)
    if not isinstance(primary_payload, dict) or not isinstance(secondary_payload, dict):
        return _build_error_payload(
            tool="pre_strategy_get_regime_snapshot",
            error_code="REGIME_INTERNAL_ERROR",
            error_message="Mapped timeframe payload is missing.",
            context=context,
        )

    cache_payload = {
        "market": market_key,
        "symbol": symbol_key,
        "timeframe_plan": timeframe_plan.to_dict(),
        "selected_timeframe": timeframe_plan.primary,
        "snapshots_by_timeframe": by_timeframe,
        "candles_by_timeframe": candles_by_timeframe,
    }
    snapshot_id = put_regime_snapshot_in_cache(
        cache_payload,
        ttl_seconds=settings.pre_strategy_regime_cache_ttl_seconds,
    )

    response_data = {
        "market": market_key,
        "symbol": symbol_key,
        "lookback_bars": safe_lookback,
        "end_utc": _datetime_to_utc_z(resolved_end),
        "timeframe_plan": timeframe_plan.to_dict(),
        "primary": {
            "timeframe": str(primary_payload.get("timeframe", "")).strip(),
            "bars": int(primary_payload.get("bars", 0) or 0),
            "source_mode": str(primary_payload.get("source_mode", "")).strip(),
            "source_label": str(primary_payload.get("source_label", "")).strip(),
            "summary": str(primary_payload.get("summary", "")).strip(),
            "family_scores": _compact_family_scores(
                dict(primary_payload.get("family_scores", {})),
                include_evidence=True,
            ),
            "choice_option_subtitles": dict(
                primary_payload.get("choice_option_subtitles", {})
            ),
            "features": dict(primary_payload.get("normalized_primary_features", {})),
        },
        "secondary": {
            "timeframe": str(secondary_payload.get("timeframe", "")).strip(),
            "bars": int(secondary_payload.get("bars", 0) or 0),
            "summary": str(secondary_payload.get("summary", "")).strip(),
            "family_scores": _compact_family_scores(
                dict(secondary_payload.get("family_scores", {})),
                include_evidence=False,
            ),
            "features": dict(secondary_payload.get("normalized_secondary_features", {})),
        },
        "snapshot_id": snapshot_id,
        "snapshot_cache_ttl_seconds": int(settings.pre_strategy_regime_cache_ttl_seconds),
    }
    return _build_success_payload(
        tool="pre_strategy_get_regime_snapshot",
        data=response_data,
    )


def pre_strategy_render_candlestick(
    snapshot_id: str,
    timeframe: str = "primary",
    bars: int = 240,
) -> Any:
    """Render candlestick image from cached pre-strategy snapshot."""

    normalized_snapshot_id = str(snapshot_id).strip()
    if not normalized_snapshot_id:
        return _build_error_payload(
            tool="pre_strategy_render_candlestick",
            error_code="INVALID_INPUT",
            error_message="snapshot_id cannot be empty",
        )

    cached = get_regime_snapshot_from_cache(normalized_snapshot_id)
    if not isinstance(cached, dict):
        return _build_error_payload(
            tool="pre_strategy_render_candlestick",
            error_code="NOT_FOUND",
            error_message="snapshot_id is missing or expired. Please call pre_strategy_get_regime_snapshot again.",
            context={"snapshot_id": normalized_snapshot_id},
        )

    timeframe_plan = cached.get("timeframe_plan")
    if isinstance(timeframe_plan, dict):
        primary = str(timeframe_plan.get("primary", "")).strip().lower()
        secondary = str(timeframe_plan.get("secondary", "")).strip().lower()
    else:
        primary = ""
        secondary = ""

    requested = str(timeframe).strip().lower() or "primary"
    if requested == "primary":
        resolved_timeframe = primary or "1h"
    elif requested == "secondary":
        resolved_timeframe = secondary or primary or "1h"
    else:
        try:
            resolved_timeframe = _normalize_regime_timeframe(requested)
        except ValueError as exc:
            return _build_error_payload(
                tool="pre_strategy_render_candlestick",
                error_code="INVALID_INPUT",
                error_message=str(exc),
                context={"timeframe": timeframe},
            )

    snapshots_by_tf = cached.get("snapshots_by_timeframe", {})
    selected_snapshot = (
        snapshots_by_tf.get(resolved_timeframe)
        if isinstance(snapshots_by_tf, dict)
        else None
    )
    candles_by_tf = cached.get("candles_by_timeframe", {})
    frame = (
        candles_by_tf.get(resolved_timeframe)
        if isinstance(candles_by_tf, dict)
        else None
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _build_error_payload(
            tool="pre_strategy_render_candlestick",
            error_code="CANDLES_NOT_FOUND",
            error_message=f"No cached candles for timeframe={resolved_timeframe}.",
            context={"snapshot_id": normalized_snapshot_id, "timeframe": resolved_timeframe},
        )

    selected_scores = (
        selected_snapshot.get("family_scores", {})
        if isinstance(selected_snapshot, dict)
        else {}
    )
    selected_summary = (
        str(selected_snapshot.get("summary", "")).strip()
        if isinstance(selected_snapshot, dict)
        else ""
    )
    alt_text = build_chart_alt_text(
        timeframe=resolved_timeframe,
        summary=selected_summary,
        family_scores=selected_scores if isinstance(selected_scores, dict) else {},
    )
    safe_bars = max(80, min(int(bars), int(settings.pre_strategy_regime_image_max_bars)))

    try:
        image = render_candlestick_image(
            candles=frame,
            title=f"{cached.get('symbol', 'symbol')} {resolved_timeframe}",
            max_bars=safe_bars,
        )
        log_mcp_tool_result(
            category="market_data",
            tool="pre_strategy_render_candlestick",
            ok=True,
        )
        return [image, alt_text]
    except Exception as exc:  # noqa: BLE001
        fallback = (
            f"{alt_text} "
            f"Image rendering degraded to text fallback: {type(exc).__name__}."
        )
        log_mcp_tool_result(
            category="market_data",
            tool="pre_strategy_render_candlestick",
            ok=True,
        )
        return [fallback]


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
        pre_strategy_get_regime_snapshot,
        market_data_detect_missing_ranges,
        market_data_fetch_missing_ranges,
        market_data_get_sync_job,
        market_data_get_candles,
    ):
        mcp.tool()(tool)
