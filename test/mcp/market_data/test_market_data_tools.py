from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from apps.mcp.domains.market_data import tools as market_tools
from packages.infra.providers.trading.adapters.base import OhlcvBar, QuoteSnapshot


async def test_000_get_symbol_candles_uses_alpaca_and_truncates(
    monkeypatch: Any,
) -> None:
    calls: dict[str, Any] = {}

    class _FakeAlpacaClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            *,
            since: datetime | None = None,
            timeframe: str = "1Min",
            limit: int = 500,
            market: str | None = None,
        ) -> list[OhlcvBar]:
            calls["fetch_ohlcv"] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "market": market,
                "since": since,
            }
            return [
                OhlcvBar(
                    timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100.5"),
                    volume=Decimal("10"),
                ),
                OhlcvBar(
                    timestamp=datetime(2026, 1, 1, 10, 1, tzinfo=UTC),
                    open=Decimal("100.5"),
                    high=Decimal("102"),
                    low=Decimal("100"),
                    close=Decimal("101"),
                    volume=Decimal("12"),
                ),
                OhlcvBar(
                    timestamp=datetime(2026, 1, 1, 10, 2, tzinfo=UTC),
                    open=Decimal("101"),
                    high=Decimal("103"),
                    low=Decimal("100.8"),
                    close=Decimal("102"),
                    volume=Decimal("13"),
                ),
            ]

        async def aclose(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FakeAlpacaClient)

    result = await market_tools.get_symbol_candles(
        symbol="spy",
        market="stock",
        period="1d",
        interval="1m",
        limit=2,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["market"] == "us_stocks"
    assert payload["symbol"] == "SPY"
    assert payload["provider"] == "alpaca"
    assert payload["rows"] == 2
    assert payload["truncated"] is True
    assert len(payload["candles"]) == 2
    assert payload["candles"][0]["datetime"] == "2026-01-01T10:01:00Z"
    assert calls["fetch_ohlcv"]["symbol"] == "SPY"
    assert calls["fetch_ohlcv"]["market"] == "us_stocks"
    assert calls["fetch_ohlcv"]["timeframe"] == "1Min"
    assert calls["fetch_ohlcv"]["limit"] == 2
    assert calls["closed"] is True


async def test_010_get_symbol_candles_rejects_invalid_interval(
    monkeypatch: Any,
) -> None:
    class _FailIfCalledClient:
        def __init__(self) -> None:
            raise AssertionError("client should not be created for invalid interval")

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FailIfCalledClient)

    result = await market_tools.get_symbol_candles(
        symbol="SPY",
        market="stock",
        interval="2m",
    )
    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"


async def test_020_get_symbol_metadata_prefers_latest_quote(
    monkeypatch: Any,
) -> None:
    calls: dict[str, Any] = {"quote_called": False, "bar_called": False, "closed": False}

    class _FakeAlpacaClient:
        async def fetch_latest_quote(
            self,
            symbol: str,
            *,
            market: str | None = None,
        ) -> QuoteSnapshot | None:
            calls["quote_called"] = True
            return QuoteSnapshot(
                symbol=symbol,
                bid=Decimal("190"),
                ask=Decimal("191"),
                last=Decimal("190.5"),
                timestamp=datetime(2026, 1, 2, 9, 30, tzinfo=UTC),
                raw={},
            )

        async def fetch_latest_bar(
            self,
            symbol: str,
            *,
            market: str | None = None,
        ) -> OhlcvBar | None:
            calls["bar_called"] = True
            return None

        async def aclose(self) -> None:
            calls["closed"] = True

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FakeAlpacaClient)

    result = await market_tools.get_symbol_metadata(symbol="AAPL", market="stock")
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["metadata_source"] == "latest_quote"
    assert payload["metadata"]["last"] == 190.5
    assert calls["quote_called"] is True
    assert calls["bar_called"] is False
    assert calls["closed"] is True


async def test_030_get_symbol_metadata_falls_back_to_latest_bar(
    monkeypatch: Any,
) -> None:
    class _FakeAlpacaClient:
        async def fetch_latest_quote(
            self,
            symbol: str,
            *,
            market: str | None = None,
        ) -> QuoteSnapshot | None:
            return None

        async def fetch_latest_bar(
            self,
            symbol: str,
            *,
            market: str | None = None,
        ) -> OhlcvBar | None:
            return OhlcvBar(
                timestamp=datetime(2026, 1, 2, 9, 35, tzinfo=UTC),
                open=Decimal("120"),
                high=Decimal("121"),
                low=Decimal("119"),
                close=Decimal("120.7"),
                volume=Decimal("20"),
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FakeAlpacaClient)

    result = await market_tools.get_symbol_metadata(symbol="BTCUSD", market="crypto")
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["metadata_source"] == "latest_bar"
    assert payload["metadata"]["last"] == 120.7


async def test_040_get_symbol_metadata_rejects_unsupported_market(
    monkeypatch: Any,
) -> None:
    class _FailIfCalledClient:
        def __init__(self) -> None:
            raise AssertionError("client should not be created for unsupported market")

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FailIfCalledClient)

    result = await market_tools.get_symbol_metadata(symbol="EURUSD", market="forex")
    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"


async def test_050_market_data_get_candles_alias_calls_new_async_path(
    monkeypatch: Any,
) -> None:
    calls: dict[str, Any] = {}

    class _FakeAlpacaClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            *,
            since: datetime | None = None,
            timeframe: str = "1Day",
            limit: int = 30,
            market: str | None = None,
        ) -> list[OhlcvBar]:
            calls["fetch"] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "market": market,
                "since": since,
            }
            return [
                OhlcvBar(
                    timestamp=datetime(2026, 1, 3, 0, 0, tzinfo=UTC),
                    open=Decimal("300"),
                    high=Decimal("301"),
                    low=Decimal("299"),
                    close=Decimal("300.8"),
                    volume=Decimal("1000"),
                )
            ]

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(market_tools, "AlpacaMarketDataClient", _FakeAlpacaClient)

    result = await market_tools.market_data_get_candles(symbol="MSFT", interval="1d", limit=30)
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["market"] == "us_stocks"
    assert payload["symbol"] == "MSFT"
    assert calls["fetch"]["timeframe"] == "1Day"
    assert calls["fetch"]["market"] == "us_stocks"


async def test_060_market_data_fetch_missing_ranges_deduplicated_job_not_requeued(
    monkeypatch: Any,
) -> None:
    job_id = uuid4()
    calls = {"schedule": 0}
    requested_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    requested_end = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)

    class _SessionCtx:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    async def _fake_new_db_session() -> _SessionCtx:
        return _SessionCtx()

    async def _fake_create_job(db: Any, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["provider"] == "alpaca"
        return SimpleNamespace(
            job_id=job_id,
            status="running",
            progress=37,
            deduplicated=True,
        )

    async def _fake_schedule(_job_id: Any) -> str:
        calls["schedule"] += 1
        return "unexpected-task-id"

    async def _fake_view(db: Any, **kwargs: Any) -> SimpleNamespace:
        assert kwargs["job_id"] == job_id
        return SimpleNamespace(
            job_id=job_id,
            provider="alpaca",
            market="us_stocks",
            symbol="SPY",
            timeframe="1m",
            status="running",
            progress=37,
            current_step="fetching",
            requested_start=requested_start,
            requested_end=requested_end,
            missing_ranges=(),
            rows_written=0,
            range_filled=0,
            total_ranges=0,
            errors=(),
            submitted_at=requested_start,
            completed_at=None,
        )

    monkeypatch.setattr(market_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(market_tools, "create_market_data_sync_job", _fake_create_job)
    monkeypatch.setattr(market_tools, "schedule_market_data_sync_job", _fake_schedule)
    monkeypatch.setattr(market_tools, "get_market_data_sync_job_view", _fake_view)

    result = await market_tools.market_data_fetch_missing_ranges(
        provider="alpaca",
        market="stock",
        symbol="SPY",
        timeframe="1m",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T00:30:00Z",
        run_async=True,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["sync_job_id"] == str(job_id)
    assert payload["queued_task_id"] is None
    assert payload["estimated_wait_seconds"] == 15
    assert payload["recommended_poll_interval_seconds"] == 5
    assert payload["recommended_next_poll_seconds"] == 5
    assert calls["schedule"] == 0


async def test_070_market_data_fetch_missing_ranges_new_job_enqueues(
    monkeypatch: Any,
) -> None:
    job_id = uuid4()
    calls = {"schedule": 0}
    requested_start = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    requested_end = datetime(2026, 1, 2, 0, 30, tzinfo=UTC)

    class _SessionCtx:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    async def _fake_new_db_session() -> _SessionCtx:
        return _SessionCtx()

    async def _fake_create_job(db: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            status="pending",
            progress=0,
            deduplicated=False,
        )

    async def _fake_schedule(job_uuid: Any) -> str:
        assert str(job_uuid) == str(job_id)
        calls["schedule"] += 1
        return "task-123"

    async def _fake_view(db: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            provider="alpaca",
            market="us_stocks",
            symbol="SPY",
            timeframe="1m",
            status="pending",
            progress=0,
            current_step="queued",
            requested_start=requested_start,
            requested_end=requested_end,
            missing_ranges=(),
            rows_written=0,
            range_filled=0,
            total_ranges=0,
            errors=(),
            submitted_at=requested_start,
            completed_at=None,
        )

    monkeypatch.setattr(market_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(market_tools, "create_market_data_sync_job", _fake_create_job)
    monkeypatch.setattr(market_tools, "schedule_market_data_sync_job", _fake_schedule)
    monkeypatch.setattr(market_tools, "get_market_data_sync_job_view", _fake_view)

    result = await market_tools.market_data_fetch_missing_ranges(
        provider="alpaca",
        market="stock",
        symbol="SPY",
        timeframe="1m",
        start_date="2026-01-02T00:00:00Z",
        end_date="2026-01-02T00:30:00Z",
        run_async=True,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["sync_job_id"] == str(job_id)
    assert payload["queued_task_id"] == "task-123"
    assert payload["estimated_wait_seconds"] == 15
    assert payload["recommended_poll_interval_seconds"] == 5
    assert payload["recommended_next_poll_seconds"] == 5
    assert calls["schedule"] == 1


async def test_080_market_data_fetch_missing_ranges_normalizes_provider_aliases(
    monkeypatch: Any,
) -> None:
    job_id = uuid4()
    requested_start = datetime(2026, 1, 3, 0, 0, tzinfo=UTC)
    requested_end = datetime(2026, 1, 3, 0, 30, tzinfo=UTC)
    captured_provider: dict[str, str] = {}

    class _SessionCtx:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    async def _fake_new_db_session() -> _SessionCtx:
        return _SessionCtx()

    async def _fake_create_job(db: Any, **kwargs: Any) -> SimpleNamespace:
        captured_provider["value"] = str(kwargs["provider"])
        return SimpleNamespace(
            job_id=job_id,
            status="pending",
            progress=0,
            deduplicated=False,
        )

    async def _fake_schedule(_job_uuid: Any) -> str:
        return "task-456"

    async def _fake_view(db: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            provider="alpaca",
            market="us_stocks",
            symbol="SPY",
            timeframe="1m",
            status="pending",
            progress=0,
            current_step="queued",
            requested_start=requested_start,
            requested_end=requested_end,
            missing_ranges=(),
            rows_written=0,
            range_filled=0,
            total_ranges=0,
            errors=(),
            submitted_at=requested_start,
            completed_at=None,
        )

    monkeypatch.setattr(market_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(market_tools, "create_market_data_sync_job", _fake_create_job)
    monkeypatch.setattr(market_tools, "schedule_market_data_sync_job", _fake_schedule)
    monkeypatch.setattr(market_tools, "get_market_data_sync_job_view", _fake_view)

    result = await market_tools.market_data_fetch_missing_ranges(
        provider="local_parquet",
        market="stock",
        symbol="SPY",
        timeframe="1m",
        start_date="2026-01-03T00:00:00Z",
        end_date="2026-01-03T00:30:00Z",
        run_async=True,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert captured_provider["value"] == "alpaca"
    assert payload["provider"] == "alpaca"
    assert payload["provider_requested"] == "local_parquet"
    assert payload["estimated_wait_seconds"] == 15
    assert payload["recommended_poll_interval_seconds"] == 5
    assert payload["recommended_next_poll_seconds"] == 5


def _build_sample_regime_frame(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    x = np.linspace(0, 24, rows)
    close = 100 + np.sin(x) * 2 + np.linspace(0, 6, rows)
    open_ = close + np.cos(x) * 0.2
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1_000 + np.sin(x * 2.2) * 120 + np.linspace(0, 40, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _build_hourly_regime_frame(rows: int, *, start: str = "2025-01-01T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="1h", tz="UTC")
    x = np.linspace(0, 12, rows)
    close = 150 + np.sin(x) * 1.2 + np.linspace(0, 3, rows)
    open_ = close + np.cos(x) * 0.1
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    volume = 800 + np.sin(x * 1.5) * 80 + np.linspace(0, 20, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


async def test_090_pre_strategy_get_regime_snapshot_builds_snapshot_and_cache(
    monkeypatch: Any,
) -> None:
    frame = _build_sample_regime_frame()

    async def _fake_resolve_regime_frame(**kwargs: Any) -> tuple[pd.DataFrame, str, str]:
        del kwargs
        return frame.copy(), "fallback_local", "local_parquet"

    monkeypatch.setattr(market_tools, "_resolve_regime_frame", _fake_resolve_regime_frame)

    result = await market_tools.pre_strategy_get_regime_snapshot(
        market="crypto",
        symbol="BTCUSD",
        opportunity_frequency_bucket="daily",
        holding_period_bucket="swing_days",
        lookback_bars=300,
    )
    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["snapshot_id"]
    assert payload["timeframe_plan"]["primary"]
    assert "selected_timeframe" not in payload
    assert isinstance(payload.get("primary"), dict)
    assert isinstance(payload.get("secondary"), dict)
    assert payload["primary"]["timeframe"] == payload["timeframe_plan"]["primary"]
    assert payload["secondary"]["timeframe"] == payload["timeframe_plan"]["secondary"]
    assert payload["primary"]["family_scores"]["recommended_family"] in {
        "trend_continuation",
        "mean_reversion",
        "volatility_regime",
    }
    assert 0.0 <= payload["primary"]["features"]["adx_n"] <= 1.0
    assert -1.0 <= payload["primary"]["features"]["cumulative_return_n"] <= 1.0


def test_095_load_local_regime_frame_expands_lookback_for_4h(
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {"load": 0}

    class _FakeLoader:
        def get_symbol_metadata(self, market: str, symbol: str) -> dict[str, Any]:
            del market, symbol
            return {
                "available_timerange": {
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2026-03-13T20:59:00+00:00",
                }
            }

        def load(
            self,
            market: str,
            symbol: str,
            timeframe: str,
            start_date: datetime,
            end_date: datetime,
        ) -> pd.DataFrame:
            del market, symbol, start_date, end_date
            calls["load"] += 1
            assert timeframe == "1h"
            if calls["load"] == 1:
                # First attempt intentionally under-fetches (insufficient after 4h resample).
                return _build_hourly_regime_frame(320, start="2026-01-01T00:00:00Z")
            # Second attempt provides enough history for 500x 4h bars.
            return _build_hourly_regime_frame(2600, start="2025-01-01T00:00:00Z")

    monkeypatch.setattr(market_tools, "_get_data_loader", lambda: _FakeLoader())

    frame = market_tools._load_local_regime_frame(
        market="us_stocks",
        symbol="AAPL",
        timeframe="4h",
        lookback_bars=500,
        end_utc=datetime(2026, 3, 15, 0, 0, tzinfo=UTC),
    )

    assert calls["load"] >= 2
    assert len(frame) == 500


async def test_096_resolve_regime_frame_uses_local_storage_only(
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {"alpaca": 0, "local": 0}
    sample = _build_sample_regime_frame(rows=700)

    async def _fake_fetch_alpaca_regime_frame(**kwargs: Any) -> pd.DataFrame:
        del kwargs
        calls["alpaca"] += 1
        return sample.copy()

    def _fake_load_local_regime_frame(**kwargs: Any) -> pd.DataFrame:
        del kwargs
        calls["local"] += 1
        return sample.copy()

    monkeypatch.setattr(
        market_tools,
        "_fetch_alpaca_regime_frame",
        _fake_fetch_alpaca_regime_frame,
    )
    monkeypatch.setattr(
        market_tools,
        "_load_local_regime_frame",
        _fake_load_local_regime_frame,
    )

    frame, source_mode, source_label = await market_tools._resolve_regime_frame(
        market="crypto",
        symbol="BTCUSD",
        timeframe="1h",
        lookback_bars=500,
        end_utc=datetime(2026, 3, 15, 0, 0, tzinfo=UTC),
    )

    assert calls["alpaca"] == 0
    assert calls["local"] == 1
    assert source_mode == "local_primary"
    assert source_label == "local_parquet"
    assert len(frame) == 700


async def test_100_pre_strategy_render_candlestick_uses_cached_snapshot(
    monkeypatch: Any,
) -> None:
    frame = _build_sample_regime_frame()

    async def _fake_resolve_regime_frame(**kwargs: Any) -> tuple[pd.DataFrame, str, str]:
        del kwargs
        return frame.copy(), "fallback_local", "local_parquet"

    monkeypatch.setattr(market_tools, "_resolve_regime_frame", _fake_resolve_regime_frame)
    monkeypatch.setattr(
        market_tools,
        "render_candlestick_image",
        lambda **kwargs: "fake-image-object",
    )

    snapshot_payload = json.loads(
        await market_tools.pre_strategy_get_regime_snapshot(
            market="crypto",
            symbol="ETHUSD",
            opportunity_frequency_bucket="few_per_week",
            holding_period_bucket="intraday",
            lookback_bars=280,
        )
    )
    snapshot_id = snapshot_payload["snapshot_id"]

    rendered = market_tools.pre_strategy_render_candlestick(
        snapshot_id=snapshot_id,
        timeframe="primary",
        bars=120,
    )

    assert isinstance(rendered, list)
    assert rendered[0] == "fake-image-object"
    assert isinstance(rendered[1], str)
    assert "Chart timeframe=" in rendered[1]


def test_110_register_market_data_tools_includes_pre_strategy_regime_tools() -> None:
    class _FakeMCP:
        def __init__(self) -> None:
            self.registered_names: list[str] = []

        def tool(self):  # noqa: ANN001
            def _decorator(fn):  # noqa: ANN001
                self.registered_names.append(fn.__name__)
                return fn

            return _decorator

    fake_mcp = _FakeMCP()
    market_tools.register_market_data_tools(fake_mcp)  # type: ignore[arg-type]

    assert "pre_strategy_get_regime_snapshot" in fake_mcp.registered_names
    assert "pre_strategy_render_candlestick" not in fake_mcp.registered_names
    assert "pre_strategy_get_regime_snapshot" in market_tools.TOOL_NAMES
    assert "pre_strategy_render_candlestick" not in market_tools.TOOL_NAMES
