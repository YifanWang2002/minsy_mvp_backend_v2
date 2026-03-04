from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.domain.market_data.runtime import RuntimeBar, market_data_runtime
from packages.domain.trading.runtime import runtime_service
from packages.infra.providers.trading.adapters.base import OhlcvBar


class _BackfillAwareAdapter:
    def __init__(self) -> None:
        self.history_calls: list[datetime | None] = []
        self.base_start = datetime(2026, 1, 1, tzinfo=UTC)

    async def fetch_ohlcv_1m(
        self,
        _symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        self.history_calls.append(since)
        if since is None:
            return []
        start = self.base_start
        rows: list[OhlcvBar] = []
        for index in range(limit):
            ts = start + timedelta(minutes=index)
            price = Decimal("100") + Decimal(str(index))
            rows.append(
                OhlcvBar(
                    timestamp=ts,
                    open=price,
                    high=price + Decimal("1"),
                    low=price - Decimal("1"),
                    close=price + Decimal("0.5"),
                    volume=Decimal("10"),
                )
            )
        return rows

    async def fetch_latest_1m_bar(self, _symbol: str):
        return None

    async def fetch_latest_quote(self, _symbol: str):
        return None

    async def aclose(self) -> None:
        return None


def test_seed_runtime_market_data_backfills_stocks_when_latest_window_is_empty(monkeypatch) -> None:
    adapter = _BackfillAwareAdapter()

    async def _fake_builder(**_kwargs):
        return adapter, "sandbox"

    monkeypatch.setattr(
        runtime_service,
        "_build_market_data_adapter_for_run",
        _fake_builder,
    )
    monkeypatch.setattr(market_data_runtime, "_redis_read_enabled", lambda: False)
    monkeypatch.setattr(market_data_runtime, "_redis_write_enabled", lambda: False)
    monkeypatch.setattr(market_data_runtime, "_memory_cache_enabled", lambda: True)
    market_data_runtime.reset()

    try:
        market_data_runtime.hydrate_bars(
            market="stocks",
            symbol="JNJ",
            timeframe="1m",
            bars=[
                RuntimeBar(
                    timestamp=adapter.base_start + timedelta(minutes=15),
                    open=115.0,
                    high=116.0,
                    low=114.0,
                    close=115.5,
                    volume=10.0,
                )
            ],
        )
        bars, metadata, _ = asyncio.run(
            runtime_service._seed_runtime_market_data_from_provider(
                db=None,
                run=object(),
                market="stocks",
                symbol="JNJ",
                timeframe="1m",
                limit=16,
            )
        )
        assert len(bars) == 16
        assert metadata["market_data_fallback"] == "hydrated"
        assert metadata["market_data_fallback_backfill_bars"] == 16
        assert adapter.history_calls[0] is None
        assert isinstance(adapter.history_calls[1], datetime)
    finally:
        market_data_runtime.reset()
