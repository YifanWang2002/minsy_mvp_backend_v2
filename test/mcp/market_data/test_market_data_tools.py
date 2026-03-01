from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

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
