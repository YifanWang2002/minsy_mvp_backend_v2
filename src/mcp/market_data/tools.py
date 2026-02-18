"""Market-data MCP tools."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import yfinance as yf
from mcp.server.fastmcp import FastMCP
from platformdirs import user_cache_dir

from src.engine import DataLoader
from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso

TOOL_NAMES: tuple[str, ...] = (
    "check_symbol_available",
    "get_available_symbols",
    "get_symbol_data_coverage",
    "get_symbol_quote",
    "get_symbol_candles",
    "get_symbol_metadata",
    # Backward compatibility aliases.
    "market_data_get_quote",
    "market_data_get_candles",
)

VALID_INTERVALS = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
)
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

_MARKET_ALIASES: dict[str, str] = {
    "stock": "us_stocks",
    "stocks": "us_stocks",
    "us_stock": "us_stocks",
    "us_stocks": "us_stocks",
    "crypto": "crypto",
    "forex": "forex",
    "futures": "futures",
}

_VENUE_TO_MARKET: dict[str, str] = {
    "US": "us_stocks",
    "STOCK": "us_stocks",
    "STOCKS": "us_stocks",
    "CRYPTO": "crypto",
    "FOREX": "forex",
    "FX": "forex",
    "FUTURES": "futures",
}

_KNOWN_FUTURES_YF_MAP: dict[str, str] = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    "CL": "CL=F",
    "GC": "GC=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "NG": "NG=F",
    "ZN": "ZN=F",
    "ZB": "ZB=F",
}

_YF_GUARD_LOCK = Lock()


@lru_cache(maxsize=1)
def _get_data_loader() -> DataLoader:
    return DataLoader()


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def _yf_guard_state_file() -> Path:
    configured = os.getenv("YF_CACHE_GUARD_STATE_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("/tmp/minsy_yfinance_guard_state.json")


def _yf_guard_limit() -> int:
    return max(1, _parse_int_env("YF_CACHE_GUARD_LIMIT", 800))


def _yf_guard_buffer() -> int:
    return max(0, _parse_int_env("YF_CACHE_GUARD_BUFFER", 80))


def _yf_guard_trigger(limit: int, buffer: int) -> int:
    # Trigger a little before hard limit; at least 1.
    return max(1, limit - min(buffer, limit - 1))


def _load_yf_guard_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_yf_guard_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=True)
    path.write_text(payload, encoding="utf-8")


def _resolve_yfinance_cache_dir() -> Path:
    configured = os.getenv("YF_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(user_cache_dir()) / "py-yfinance"


def _clear_yfinance_cache() -> None:
    """Clear yfinance persistent + in-memory cache best-effort."""
    try:
        from yfinance import cache as yf_cache

        for manager_name in ("_CookieDBManager", "_TzDBManager", "_ISINDBManager"):
            manager = getattr(yf_cache, manager_name, None)
            close_db = getattr(manager, "close_db", None)
            if callable(close_db):
                close_db()
    except Exception:  # noqa: BLE001
        pass

    cache_dir = _resolve_yfinance_cache_dir()
    for pattern in (
        "cookies.db*",
        "tkr-tz.db*",
        "isin-tkr.db*",
    ):
        for cache_file in cache_dir.glob(pattern):
            try:
                if cache_file.is_file():
                    cache_file.unlink(missing_ok=True)
                elif cache_file.is_dir():
                    shutil.rmtree(cache_file, ignore_errors=True)
            except Exception:  # noqa: BLE001
                continue

    try:
        from yfinance import data as yf_data

        yf_data_singleton = yf_data.YfData()
        cache_get = getattr(yf_data_singleton, "cache_get", None)
        if callable(getattr(cache_get, "cache_clear", None)):
            cache_get.cache_clear()
        session = getattr(yf_data_singleton, "_session", None)
        if session is not None and hasattr(session, "cookies"):
            session.cookies.clear()
        yf_data_singleton._cookie = None
        yf_data_singleton._crumb = None
        yf_data_singleton._cookie_strategy = "basic"
    except Exception:  # noqa: BLE001
        pass


def _record_yf_request_and_maybe_clear_cache() -> None:
    """Track yfinance usage locally and clear cache near configured limit."""
    if not _parse_bool_env("YF_CACHE_GUARD_ENABLED", True):
        return

    state_file = _yf_guard_state_file()
    limit = _yf_guard_limit()
    buffer = _yf_guard_buffer()
    trigger = _yf_guard_trigger(limit, buffer)
    today = datetime.now(UTC).date().isoformat()

    with _YF_GUARD_LOCK:
        state = _load_yf_guard_state(state_file)
        if str(state.get("date")) != today:
            state = {
                "date": today,
                "count": 0,
                "limit": limit,
                "buffer": buffer,
                "trigger": trigger,
                "clear_count": 0,
                "last_clear_utc": "",
            }

        count = int(state.get("count", 0))
        count += 1
        state["count"] = count
        state["limit"] = limit
        state["buffer"] = buffer
        state["trigger"] = trigger

        if count >= trigger:
            _clear_yfinance_cache()
            state["count"] = 0
            state["clear_count"] = int(state.get("clear_count", 0)) + 1
            state["last_clear_utc"] = utc_now_iso()

        _save_yf_guard_state(state_file, state)


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


def _resolve_ticker_context(
    *,
    market: str,
    symbol: str,
) -> tuple[str, str, str, yf.Ticker]:
    market_key = _normalize_market(market)
    symbol_key = _normalize_symbol(symbol)
    yfinance_symbol = _to_yfinance_symbol(market=market_key, symbol=symbol_key)
    return market_key, symbol_key, yfinance_symbol, yf.Ticker(yfinance_symbol)


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


def _run_market_tool(
    *,
    tool: str,
    market: str,
    symbol: str,
    fetch_error_code: str,
    runner: Callable[[str, str, str, yf.Ticker], str],
    extra_context: dict[str, Any] | None = None,
) -> str:
    try:
        market_key, symbol_key, yfinance_symbol, ticker = _resolve_ticker_context(
            market=market,
            symbol=symbol,
        )
        return runner(market_key, symbol_key, yfinance_symbol, ticker)
    except ValueError as exc:
        return _build_error_payload(
            tool=tool,
            error_code="INVALID_INPUT",
            error_message=str(exc),
            context=_build_market_input_context(
                market=market,
                symbol=symbol,
                extra_context=extra_context,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        code = "UPSTREAM_RATE_LIMIT" if _is_rate_limit_error(exc) else fetch_error_code
        return _build_error_payload(
            tool=tool,
            error_code=code,
            error_message=f"{type(exc).__name__}: {exc}",
            context=_build_market_input_context(
                market=market,
                symbol=symbol,
                extra_context=extra_context,
            ),
        )


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


def _to_yfinance_symbol(*, market: str, symbol: str) -> str:
    market_key = _normalize_market(market)
    symbol_key = _normalize_symbol(symbol)

    if market_key == "us_stocks":
        # BRK.B -> BRK-B
        return symbol_key.replace(".", "-")

    if market_key == "crypto":
        if symbol_key.endswith("-USD") or symbol_key.endswith("-USDT"):
            return symbol_key.replace("-USDT", "-USD")
        compact = symbol_key.replace("-", "").replace("/", "")
        if compact.endswith("USDT"):
            return f"{compact[:-4]}-USD"
        if compact.endswith("USD"):
            return f"{compact[:-3]}-USD"
        return f"{compact}-USD"

    if market_key == "forex":
        if symbol_key.endswith("=X"):
            return symbol_key
        compact = symbol_key.replace("-", "").replace("/", "")
        if len(compact) != 6:
            raise ValueError(
                f"Unsupported forex symbol '{symbol}'. Expected 6-letter pair like EURUSD."
            )
        return f"{compact}=X"

    # futures
    if symbol_key.endswith("=F"):
        return symbol_key
    return _KNOWN_FUTURES_YF_MAP.get(symbol_key, f"{symbol_key}=F")


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _ticker_info_summary(ticker: yf.Ticker) -> dict[str, Any]:
    info = _coerce_mapping(ticker.info)
    if not info:
        return {}
    keys = [
        "shortName",
        "longName",
        "symbol",
        "exchange",
        "market",
        "quoteType",
        "currency",
        "marketCap",
        "enterpriseValue",
        "trailingPE",
        "forwardPE",
        "dividendYield",
        "beta",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "regularMarketPrice",
        "regularMarketOpen",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketVolume",
        "averageVolume",
        "sector",
        "industry",
        "country",
    ]
    return {key: info[key] for key in keys if key in info}


def _ticker_fast_info_summary(
    ticker: yf.Ticker,
    *,
    yfinance_symbol: str,
) -> dict[str, Any]:
    try:
        fast_info = ticker.fast_info
    except Exception:  # noqa: BLE001
        return {}
    if fast_info is None:
        return {}
    if isinstance(fast_info, Mapping):
        getter = fast_info.get
    elif callable(getattr(fast_info, "get", None)):
        getter = fast_info.get
    else:
        return {}

    summary = {
        "symbol": yfinance_symbol,
        "currency": getter("currency"),
        "exchange": getter("exchange"),
        "marketCap": getter("marketCap"),
        "regularMarketPrice": getter("lastPrice"),
        "regularMarketOpen": getter("open"),
        "regularMarketDayHigh": getter("dayHigh"),
        "regularMarketDayLow": getter("dayLow"),
        "regularMarketVolume": getter("lastVolume"),
        "fiftyDayAverage": getter("fiftyDayAverage"),
        "twoHundredDayAverage": getter("twoHundredDayAverage"),
        "yearHigh": getter("yearHigh"),
        "yearLow": getter("yearLow"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _is_rate_limit_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    hints = (
        "too many requests",
        "rate limit",
        "status=429",
        "status code=429",
        "response code=429",
        "http 429",
        "429 client error",
        "yf ratelimiterror",
    )
    return any(hint in error_text for hint in hints)


def _retry_yfinance_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    delays = (0.6, 1.2)
    for attempt in range(len(delays) + 1):
        try:
            _record_yf_request_and_maybe_clear_cache()
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt >= len(delays) or not _is_rate_limit_error(exc):
                raise
            _clear_yfinance_cache()
            time.sleep(delays[attempt])


def _retry_optional_yfinance_call(
    func: Any,
    *args: Any,
    default: Any,
    **kwargs: Any,
) -> Any:
    """Best-effort fetch for optional yfinance endpoints.

    Some endpoints (notably ``Ticker.info``) are far more prone to Yahoo 429s
    than others. For these optional fields we degrade gracefully instead of
    failing the entire tool call.
    """
    try:
        return _retry_yfinance_call(func, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit_error(exc):
            return default
        raise


def _history_to_records(history_df: Any) -> list[dict[str, Any]]:
    if history_df.empty:
        return []

    normalized = history_df.copy()
    if getattr(normalized.index, "tz", None) is not None:
        datetimes = normalized.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        datetimes = normalized.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized = normalized.assign(datetime=datetimes)

    has_volume = "Volume" in normalized.columns
    records = []
    for _, row in normalized.iterrows():
        records.append(
            {
                "datetime": row["datetime"],
                "open": float(row.get("Open", 0.0)),
                "high": float(row.get("High", 0.0)),
                "low": float(row.get("Low", 0.0)),
                "close": float(row.get("Close", 0.0)),
                "volume": float(row.get("Volume", 0.0)) if has_volume else 0.0,
            }
        )
    return records


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


def get_symbol_quote(symbol: str, market: str) -> str:
    """
    Get latest quote for a symbol.

    Provide `symbol` and `market`.
    """
    def _runner(
        market_key: str,
        symbol_key: str,
        yfinance_symbol: str,
        ticker: yf.Ticker,
    ) -> str:
        summary = _retry_optional_yfinance_call(
            _ticker_info_summary,
            ticker,
            default={},
        )
        if not summary:
            summary = _ticker_fast_info_summary(
                ticker,
                yfinance_symbol=yfinance_symbol,
            )
        if not summary:
            history_df = _retry_yfinance_call(
                ticker.history,
                period="5d",
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            records = _history_to_records(history_df)
            if records:
                last = records[-1]
                summary = {
                    "symbol": yfinance_symbol,
                    "regularMarketPrice": last["close"],
                    "regularMarketOpen": last["open"],
                    "regularMarketDayHigh": last["high"],
                    "regularMarketDayLow": last["low"],
                    "regularMarketVolume": last["volume"],
                }
        if not summary:
            return _build_error_payload(
                tool="get_symbol_quote",
                error_code="QUOTE_NOT_FOUND",
                error_message=f"No quote data found for {symbol_key}",
                context={
                    "market": market_key,
                    "symbol": symbol_key,
                    "yfinance_symbol": yfinance_symbol,
                },
            )
        return _build_success_payload(
            tool="get_symbol_quote",
            data={
                "market": market_key,
                "symbol": symbol_key,
                "yfinance_symbol": yfinance_symbol,
                "quote": summary,
            },
        )

    return _run_market_tool(
        tool="get_symbol_quote",
        market=market,
        symbol=symbol,
        fetch_error_code="QUOTE_FETCH_ERROR",
        runner=_runner,
    )


def get_symbol_candles(
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
    def _runner(
        market_key: str,
        symbol_key: str,
        yfinance_symbol: str,
        ticker: yf.Ticker,
    ) -> str:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'")
        if not start and period not in VALID_PERIODS:
            raise ValueError(f"Invalid period '{period}'")

        kwargs: dict[str, Any] = {
            "interval": interval,
            "auto_adjust": False,
            "actions": False,
        }
        if start:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        try:
            history_df = _retry_yfinance_call(ticker.history, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc) and interval in {
                "1m",
                "2m",
                "5m",
                "15m",
                "30m",
                "60m",
                "90m",
                "1h",
            }:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["interval"] = "1d"
                if "period" in fallback_kwargs:
                    fallback_kwargs["period"] = "1mo"
                history_df = _retry_yfinance_call(ticker.history, **fallback_kwargs)
            else:
                raise

        records = _history_to_records(history_df)
        if not records:
            return _build_error_payload(
                tool="get_symbol_candles",
                error_code="CANDLES_NOT_FOUND",
                error_message=f"No candles found for {symbol_key}",
                context={
                    "market": market_key,
                    "symbol": symbol_key,
                    "yfinance_symbol": yfinance_symbol,
                    "period": period,
                    "interval": interval,
                },
            )

        safe_limit = max(1, min(limit, 2000))
        truncated = len(records) > safe_limit
        if truncated:
            records = records[-safe_limit:]
        return _build_success_payload(
            tool="get_symbol_candles",
            data={
                "market": market_key,
                "symbol": symbol_key,
                "yfinance_symbol": yfinance_symbol,
                "period": period,
                "interval": interval,
                "start": start,
                "end": end,
                "rows": len(records),
                "truncated": truncated,
                "candles": records,
            },
        )

    return _run_market_tool(
        tool="get_symbol_candles",
        market=market,
        symbol=symbol,
        fetch_error_code="CANDLES_FETCH_ERROR",
        runner=_runner,
        extra_context={
            "period": period,
            "interval": interval,
            "start": start,
            "end": end,
        },
    )


def get_symbol_metadata(symbol: str, market: str) -> str:
    """
    Get symbol/company metadata from yfinance.

    Provide `symbol` and `market`.
    """
    def _runner(
        market_key: str,
        symbol_key: str,
        yfinance_symbol: str,
        ticker: yf.Ticker,
    ) -> str:
        info = _coerce_mapping(
            _retry_optional_yfinance_call(
                lambda: ticker.info,
                default={},
            )
        )
        metadata_source = "info"
        if not info:
            info = _ticker_fast_info_summary(
                ticker,
                yfinance_symbol=yfinance_symbol,
            )
            metadata_source = "fast_info"
        if not info:
            return _build_error_payload(
                tool="get_symbol_metadata",
                error_code="METADATA_NOT_FOUND",
                error_message=f"No metadata found for {symbol_key}",
                context={
                    "market": market_key,
                    "symbol": symbol_key,
                    "yfinance_symbol": yfinance_symbol,
                },
            )
        return _build_success_payload(
            tool="get_symbol_metadata",
            data={
                "market": market_key,
                "symbol": symbol_key,
                "yfinance_symbol": yfinance_symbol,
                "metadata_source": metadata_source,
                "metadata": info,
            },
        )

    return _run_market_tool(
        tool="get_symbol_metadata",
        market=market,
        symbol=symbol,
        fetch_error_code="METADATA_FETCH_ERROR",
        runner=_runner,
    )


def market_data_get_quote(symbol: str, venue: str = "US") -> str:
    """Backward compatibility alias for quote tool."""
    market = _VENUE_TO_MARKET.get(venue.strip().upper(), "us_stocks")
    return get_symbol_quote(symbol=symbol, market=market)


def market_data_get_candles(
    symbol: str,
    interval: str = "1d",
    limit: int = 30,
) -> str:
    """Backward compatibility alias for candles tool."""
    return get_symbol_candles(
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
        get_symbol_quote,
        get_symbol_candles,
        get_symbol_metadata,
        market_data_get_quote,
        market_data_get_candles,
    ):
        mcp.tool()(tool)
