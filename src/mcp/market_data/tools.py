"""Market-data MCP tools."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import yfinance as yf
from mcp.server.fastmcp import FastMCP

from src.engine import DataLoader
from src.mcp._utils import to_json, utc_now_iso

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


@lru_cache(maxsize=1)
def _get_data_loader() -> DataLoader:
    return DataLoader()


def _build_success_payload(tool: str, data: dict[str, Any]) -> str:
    payload = {
        "category": "market_data",
        "tool": tool,
        "ok": True,
        **data,
        "timestamp_utc": utc_now_iso(),
    }
    return to_json(payload)


def _build_error_payload(
    *,
    tool: str,
    error: str,
    context: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "category": "market_data",
        "tool": tool,
        "ok": False,
        "error": error,
        "timestamp_utc": utc_now_iso(),
    }
    if context:
        payload["context"] = context
    return to_json(payload)


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


def _ticker_info_summary(ticker: yf.Ticker) -> dict[str, Any]:
    info = ticker.info or {}
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

    summary = {
        "symbol": yfinance_symbol,
        "currency": fast_info.get("currency"),
        "exchange": fast_info.get("exchange"),
        "marketCap": fast_info.get("marketCap"),
        "regularMarketPrice": fast_info.get("lastPrice"),
        "regularMarketOpen": fast_info.get("open"),
        "regularMarketDayHigh": fast_info.get("dayHigh"),
        "regularMarketDayLow": fast_info.get("dayLow"),
        "regularMarketVolume": fast_info.get("lastVolume"),
        "fiftyDayAverage": fast_info.get("fiftyDayAverage"),
        "twoHundredDayAverage": fast_info.get("twoHundredDayAverage"),
        "yearHigh": fast_info.get("yearHigh"),
        "yearLow": fast_info.get("yearLow"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _is_rate_limit_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return "too many requests" in error_text or "rate limit" in error_text


def _retry_yfinance_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    delays = (0.6, 1.2)
    for attempt in range(len(delays) + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt >= len(delays) or not _is_rate_limit_error(exc):
                raise
            time.sleep(delays[attempt])


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
            error="symbol cannot be empty",
            context={"symbol": symbol, "market": market},
        )

    try:
        if market.strip():
            market_key = loader.normalize_market(market)
            available = symbol_key in set(loader.get_available_symbols(market_key))
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
            if symbol_key in set(loader.get_available_symbols(candidate)):
                matched_markets.append(candidate)
        return _build_success_payload(
            tool="check_symbol_available",
            data={
                "symbol": symbol_key,
                "available": bool(matched_markets),
                "matched_markets": matched_markets,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="check_symbol_available",
            error=str(exc),
            context={"symbol": symbol, "market": market},
        )


def get_available_symbols(market: str) -> str:
    """Get local symbols by market. Provide `market`."""
    loader = _get_data_loader()
    try:
        market_key = loader.normalize_market(market)
        symbols = loader.get_available_symbols(market)
        return _build_success_payload(
            tool="get_available_symbols",
            data={
                "market": market_key,
                "count": len(symbols),
                "symbols": symbols,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_available_symbols",
            error=str(exc),
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
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_symbol_data_coverage",
            error=str(exc),
            context={"market": market, "symbol": symbol},
        )


def get_symbol_quote(symbol: str, market: str) -> str:
    """
    Get latest quote for a symbol.

    Provide `symbol` and `market`.
    """
    try:
        market_key = _normalize_market(market)
        symbol_key = _normalize_symbol(symbol)
        yfinance_symbol = _to_yfinance_symbol(market=market_key, symbol=symbol_key)
        ticker = yf.Ticker(yfinance_symbol)
        summary = _retry_yfinance_call(_ticker_info_summary, ticker)
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
                error=f"No quote data found for {symbol_key}",
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
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_symbol_quote",
            error=str(exc),
            context={"market": market, "symbol": symbol},
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
    try:
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'")
        if not start and period not in VALID_PERIODS:
            raise ValueError(f"Invalid period '{period}'")

        market_key = _normalize_market(market)
        symbol_key = _normalize_symbol(symbol)
        yfinance_symbol = _to_yfinance_symbol(market=market_key, symbol=symbol_key)

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

        ticker = yf.Ticker(yfinance_symbol)
        try:
            history_df = _retry_yfinance_call(ticker.history, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc) and interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
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
                error=f"No candles found for {symbol_key}",
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
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_symbol_candles",
            error=str(exc),
            context={
                "market": market,
                "symbol": symbol,
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
    try:
        market_key = _normalize_market(market)
        symbol_key = _normalize_symbol(symbol)
        yfinance_symbol = _to_yfinance_symbol(market=market_key, symbol=symbol_key)
        ticker = yf.Ticker(yfinance_symbol)
        info = _retry_yfinance_call(lambda: ticker.info) or {}
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
                error=f"No metadata found for {symbol_key}",
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
    except Exception as exc:  # noqa: BLE001
        return _build_error_payload(
            tool="get_symbol_metadata",
            error=str(exc),
            context={"market": market, "symbol": symbol},
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
