from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from src.engine.data import DataLoader
from src.mcp.market_data import tools as market_tools


def _make_ohlcv(
    *,
    start: str,
    periods: int,
    freq: str,
    base_price: float,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    opens = pd.Series([base_price + i for i in range(periods)], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.5,
            "volume": pd.Series([10.0 * (i + 1) for i in range(periods)], dtype=float),
        }
    )


@pytest.fixture
def loader_with_mock_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DataLoader:
    files: dict[str, pd.DataFrame] = {
        "crypto/BTCUSD_1min_eth_2024.parquet": _make_ohlcv(
            start="2024-01-01 00:00:00",
            periods=5,
            freq="1min",
            base_price=100.0,
        ),
        "crypto/ETHUSD_1min_eth_2024.parquet": _make_ohlcv(
            start="2024-01-01 00:00:00",
            periods=5,
            freq="1min",
            base_price=200.0,
        ),
        "us_stocks/SPY_5min_rth_2024.parquet": _make_ohlcv(
            start="2024-01-02 14:30:00",
            periods=4,
            freq="5min",
            base_price=400.0,
        ),
    }

    parquet_map: dict[Path, pd.DataFrame] = {}
    for relative_path, dataframe in files.items():
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        parquet_map[file_path.resolve()] = dataframe

    loader = DataLoader(data_dir=tmp_path)

    def fake_read_parquet(
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        dataframe = parquet_map[Path(file_path).resolve()]
        if columns is None:
            return dataframe.copy()
        return dataframe.loc[:, columns].copy()

    monkeypatch.setattr(loader, "_read_parquet", fake_read_parquet)
    return loader


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


def _make_fake_yfinance_ticker_class() -> type:
    class FakeTicker:
        instances: list[str] = []

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            type(self).instances.append(symbol)

        @property
        def info(self) -> dict[str, Any]:
            return {
                "symbol": self.symbol,
                "shortName": f"Mock {self.symbol}",
                "regularMarketPrice": 123.45,
                "currency": "USD",
                "market": "mock",
            }

        def history(self, **_: Any) -> pd.DataFrame:
            index = pd.date_range("2025-01-01 00:00:00", periods=3, freq="5min", tz="UTC")
            return pd.DataFrame(
                {
                    "Open": [1.0, 2.0, 3.0],
                    "High": [2.0, 3.0, 4.0],
                    "Low": [0.5, 1.5, 2.5],
                    "Close": [1.5, 2.5, 3.5],
                    "Volume": [10.0, 20.0, 30.0],
                },
                index=index,
            )

    return FakeTicker


def _make_empty_yfinance_ticker_class() -> type:
    class EmptyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict[str, Any]:
            return {}

        @property
        def fast_info(self) -> dict[str, Any]:
            return {}

        def history(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    return EmptyTicker


def _make_non_mapping_info_yfinance_ticker_class() -> type:
    class ResponseLikeInfo:
        def __contains__(self, _key: object) -> bool:
            return True

    class NonMappingInfoTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> Any:
            return ResponseLikeInfo()

        @property
        def fast_info(self) -> dict[str, Any]:
            return {
                "lastPrice": 456.78,
                "open": 450.0,
                "dayHigh": 460.0,
                "dayLow": 445.0,
                "lastVolume": 987654.0,
                "currency": "USD",
                "exchange": "NMS",
            }

        def history(self, **_: Any) -> pd.DataFrame:
            return pd.DataFrame()

    return NonMappingInfoTicker


def _make_rate_limited_info_yfinance_ticker_class() -> type:
    class RateLimitedInfoTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict[str, Any]:
            raise RuntimeError("Too Many Requests. Rate limited")

        @property
        def fast_info(self) -> dict[str, Any]:
            return {
                "lastPrice": 321.0,
                "open": 319.0,
                "dayHigh": 325.0,
                "dayLow": 315.0,
                "lastVolume": 123456.0,
                "currency": "USD",
                "exchange": "NMS",
            }

        def history(self, **_: Any) -> pd.DataFrame:
            index = pd.date_range("2025-01-01 00:00:00", periods=2, freq="1d", tz="UTC")
            return pd.DataFrame(
                {
                    "Open": [318.0, 320.0],
                    "High": [323.0, 326.0],
                    "Low": [316.0, 319.0],
                    "Close": [322.0, 324.0],
                    "Volume": [111111.0, 222222.0],
                },
                index=index,
            )

    return RateLimitedInfoTicker


def test_yf_cache_guard_clears_cache_when_near_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_file = tmp_path / "yf_guard_state.json"
    monkeypatch.setenv("YF_CACHE_GUARD_ENABLED", "1")
    monkeypatch.setenv("YF_CACHE_GUARD_LIMIT", "3")
    monkeypatch.setenv("YF_CACHE_GUARD_BUFFER", "1")
    monkeypatch.setenv("YF_CACHE_GUARD_STATE_FILE", str(state_file))

    clear_calls: list[str] = []
    monkeypatch.setattr(market_tools, "_clear_yfinance_cache", lambda: clear_calls.append("cleared"))

    market_tools._record_yf_request_and_maybe_clear_cache()
    assert clear_calls == []
    first = market_tools._load_yf_guard_state(state_file)
    assert int(first["count"]) == 1
    assert int(first["trigger"]) == 2

    market_tools._record_yf_request_and_maybe_clear_cache()
    second = market_tools._load_yf_guard_state(state_file)
    assert clear_calls == ["cleared"]
    assert int(second["count"]) == 0
    assert int(second["clear_count"]) == 1
    assert isinstance(second.get("last_clear_utc"), str) and second["last_clear_utc"]


def test_yf_cache_guard_resets_counter_on_new_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_file = tmp_path / "yf_guard_state.json"
    monkeypatch.setenv("YF_CACHE_GUARD_ENABLED", "1")
    monkeypatch.setenv("YF_CACHE_GUARD_LIMIT", "100")
    monkeypatch.setenv("YF_CACHE_GUARD_BUFFER", "20")
    monkeypatch.setenv("YF_CACHE_GUARD_STATE_FILE", str(state_file))
    monkeypatch.setattr(market_tools, "_clear_yfinance_cache", lambda: None)

    yesterday = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()
    stale = {
        "date": yesterday,
        "count": 99,
        "limit": 100,
        "buffer": 20,
        "trigger": 80,
        "clear_count": 7,
        "last_clear_utc": "2026-01-01T00:00:00Z",
    }
    market_tools._save_yf_guard_state(state_file, stale)

    market_tools._record_yf_request_and_maybe_clear_cache()
    refreshed = market_tools._load_yf_guard_state(state_file)
    assert refreshed["date"] != yesterday
    assert int(refreshed["count"]) == 1
    assert int(refreshed["clear_count"]) == 0


@pytest.mark.asyncio
async def test_local_data_tools(loader_with_mock_parquet: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    fake_ticker = _make_fake_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", fake_ticker)

    mcp = FastMCP("test-market-data")
    market_tools.register_market_data_tools(mcp)

    symbols_call = await mcp.call_tool("get_available_symbols", {"market": "crypto"})
    symbols_payload = _extract_payload(symbols_call)
    assert symbols_payload["ok"] is True
    assert symbols_payload["symbols"] == ["BTCUSD", "ETHUSD"]

    coverage_call = await mcp.call_tool(
        "get_symbol_data_coverage",
        {"market": "stock", "symbol": "SPY"},
    )
    coverage_payload = _extract_payload(coverage_call)
    assert coverage_payload["ok"] is True
    assert coverage_payload["metadata"]["available_timerange"] == {
        "start": "2024-01-02T14:30:00+00:00",
        "end": "2024-01-02T14:45:00+00:00",
    }
    assert coverage_payload["metadata"]["available_timeframes"] == [
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "1d",
    ]


@pytest.mark.asyncio
async def test_yfinance_wrappers_cover_four_markets(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    fake_ticker = _make_fake_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", fake_ticker)

    mcp = FastMCP("test-market-data")
    market_tools.register_market_data_tools(mcp)

    cases = [
        ("stock", "AAPL", "AAPL"),
        ("crypto", "BTCUSD", "BTC-USD"),
        ("forex", "EURUSD", "EURUSD=X"),
        ("futures", "ES", "ES=F"),
    ]

    for market, symbol, expected_yf_symbol in cases:
        quote_call = await mcp.call_tool(
            "get_symbol_quote",
            {"market": market, "symbol": symbol},
        )
        quote_payload = _extract_payload(quote_call)
        assert quote_payload["ok"] is True
        assert quote_payload["yfinance_symbol"] == expected_yf_symbol
        assert quote_payload["quote"]["symbol"] == expected_yf_symbol

        candles_call = await mcp.call_tool(
            "get_symbol_candles",
            {"market": market, "symbol": symbol},
        )
        candles_payload = _extract_payload(candles_call)
        assert candles_payload["ok"] is True
        assert candles_payload["yfinance_symbol"] == expected_yf_symbol
        assert candles_payload["period"] == "1d"
        assert candles_payload["interval"] == "1d"
        assert candles_payload["rows"] == 3

        metadata_call = await mcp.call_tool(
            "get_symbol_metadata",
            {"market": market, "symbol": symbol},
        )
        metadata_payload = _extract_payload(metadata_call)
        assert metadata_payload["ok"] is True
        assert metadata_payload["yfinance_symbol"] == expected_yf_symbol
        assert metadata_payload["metadata"]["symbol"] == expected_yf_symbol


@pytest.mark.asyncio
async def test_quote_and_metadata_handle_non_mapping_info(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    non_mapping_ticker = _make_non_mapping_info_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", non_mapping_ticker)

    mcp = FastMCP("test-market-data-non-mapping-info")
    market_tools.register_market_data_tools(mcp)

    quote_call = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "stock", "symbol": "MSFT"},
    )
    quote_payload = _extract_payload(quote_call)
    assert quote_payload["ok"] is True
    assert quote_payload["quote"]["regularMarketPrice"] == 456.78
    assert quote_payload["quote"]["regularMarketVolume"] == 987654.0

    metadata_call = await mcp.call_tool(
        "get_symbol_metadata",
        {"market": "stock", "symbol": "MSFT"},
    )
    metadata_payload = _extract_payload(metadata_call)
    assert metadata_payload["ok"] is True
    assert metadata_payload["metadata_source"] == "fast_info"
    assert metadata_payload["metadata"]["regularMarketPrice"] == 456.78


@pytest.mark.asyncio
async def test_quote_and_metadata_fallback_when_info_rate_limited(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    rate_limited_ticker = _make_rate_limited_info_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", rate_limited_ticker)

    mcp = FastMCP("test-market-data-rate-limited-info-fallback")
    market_tools.register_market_data_tools(mcp)

    quote_call = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "stock", "symbol": "AAPL"},
    )
    quote_payload = _extract_payload(quote_call)
    assert quote_payload["ok"] is True
    assert quote_payload["quote"]["regularMarketPrice"] == 321.0

    metadata_call = await mcp.call_tool(
        "get_symbol_metadata",
        {"market": "stock", "symbol": "AAPL"},
    )
    metadata_payload = _extract_payload(metadata_call)
    assert metadata_payload["ok"] is True
    assert metadata_payload["metadata_source"] == "fast_info"
    assert metadata_payload["metadata"]["regularMarketPrice"] == 321.0


@pytest.mark.asyncio
async def test_market_data_tools_error_payload_shape(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    monkeypatch.setattr(market_tools, "_ticker_fast_info_summary", lambda *_args, **_kwargs: {})
    empty_ticker = _make_empty_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", empty_ticker)

    mcp = FastMCP("test-market-data-errors")
    market_tools.register_market_data_tools(mcp)

    invalid_quote = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "invalid_market", "symbol": "AAPL"},
    )
    invalid_quote_payload = _extract_payload(invalid_quote)
    assert invalid_quote_payload["ok"] is False
    assert isinstance(invalid_quote_payload["error"], dict)
    assert invalid_quote_payload["error"]["code"] == "INVALID_INPUT"
    assert "message" in invalid_quote_payload["error"]
    assert isinstance(invalid_quote_payload.get("context"), dict)

    no_quote = await mcp.call_tool(
        "get_symbol_quote",
        {"market": "stock", "symbol": "AAPL"},
    )
    no_quote_payload = _extract_payload(no_quote)
    assert no_quote_payload["ok"] is False
    assert no_quote_payload["error"]["code"] == "QUOTE_NOT_FOUND"
    assert isinstance(no_quote_payload.get("context"), dict)

    invalid_candles = await mcp.call_tool(
        "get_symbol_candles",
        {"market": "stock", "symbol": "AAPL", "interval": "bad"},
    )
    invalid_candles_payload = _extract_payload(invalid_candles)
    assert invalid_candles_payload["ok"] is False
    assert invalid_candles_payload["error"]["code"] == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_check_symbol_available_covers_global_and_market_scoped_lookup(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    fake_ticker = _make_fake_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", fake_ticker)

    mcp = FastMCP("test-market-data-check-symbol")
    market_tools.register_market_data_tools(mcp)

    global_hit_call = await mcp.call_tool(
        "check_symbol_available",
        {"symbol": "BTCUSD"},
    )
    global_hit_payload = _extract_payload(global_hit_call)
    assert global_hit_payload["ok"] is True
    assert global_hit_payload["available"] is True
    assert global_hit_payload["matched_markets"] == ["crypto"]

    market_hit_call = await mcp.call_tool(
        "check_symbol_available",
        {"market": "stock", "symbol": "SPY"},
    )
    market_hit_payload = _extract_payload(market_hit_call)
    assert market_hit_payload["ok"] is True
    assert market_hit_payload["market"] == "us_stocks"
    assert market_hit_payload["available"] is True

    market_miss_call = await mcp.call_tool(
        "check_symbol_available",
        {"market": "futures", "symbol": "SPY"},
    )
    market_miss_payload = _extract_payload(market_miss_call)
    assert market_miss_payload["ok"] is True
    assert market_miss_payload["market"] == "futures"
    assert market_miss_payload["available"] is False


@pytest.mark.asyncio
async def test_legacy_market_data_alias_tools_map_to_yfinance_symbols(
    loader_with_mock_parquet: DataLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: loader_with_mock_parquet)
    fake_ticker = _make_fake_yfinance_ticker_class()
    monkeypatch.setattr(market_tools.yf, "Ticker", fake_ticker)

    mcp = FastMCP("test-market-data-legacy-aliases")
    market_tools.register_market_data_tools(mcp)

    quote_call = await mcp.call_tool(
        "market_data_get_quote",
        {"symbol": "EURUSD", "venue": "FOREX"},
    )
    quote_payload = _extract_payload(quote_call)
    assert quote_payload["ok"] is True
    assert quote_payload["market"] == "forex"
    assert quote_payload["yfinance_symbol"] == "EURUSD=X"

    candles_call = await mcp.call_tool(
        "market_data_get_candles",
        {"symbol": "AAPL", "interval": "1d", "limit": 2},
    )
    candles_payload = _extract_payload(candles_call)
    assert candles_payload["ok"] is True
    assert candles_payload["market"] == "us_stocks"
    assert candles_payload["yfinance_symbol"] == "AAPL"
    assert candles_payload["period"] == "1mo"
    assert candles_payload["interval"] == "1d"
    assert candles_payload["rows"] == 2
    assert candles_payload["truncated"] is True
