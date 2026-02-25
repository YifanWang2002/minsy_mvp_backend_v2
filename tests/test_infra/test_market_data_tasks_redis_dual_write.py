from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.config import settings
from src.engine.execution.adapters.base import OhlcvBar, QuoteSnapshot
from src.engine.market_data.redis_store import RedisMarketDataStore
from src.engine.market_data.redis_subscription_store import RedisSubscriptionStore
from src.engine.market_data.runtime import MarketDataRuntime
from src.workers import market_data_tasks


class _StubRefreshProvider:
    async def fetch_quote(self, *, symbol: str, market: str) -> QuoteSnapshot:
        _ = (symbol, market)
        return QuoteSnapshot(
            symbol="AAPL",
            bid=Decimal("100"),
            ask=Decimal("101"),
            last=Decimal("100.5"),
            timestamp=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
        )

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar:
        _ = (symbol, market)
        return OhlcvBar(
            timestamp=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=Decimal("10"),
        )

    async def aclose(self) -> None:
        return None


class _StubBackfillProvider:
    async def fetch_recent_1m_bars(
        self,
        *,
        symbol: str,
        market: str,
        since: datetime,
        limit: int,
    ) -> list[OhlcvBar]:
        _ = (symbol, market, since, limit)
        return [
            OhlcvBar(
                timestamp=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.5"),
                volume=Decimal("10"),
            ),
            OhlcvBar(
                timestamp=datetime(2026, 2, 24, 10, 1, tzinfo=UTC),
                open=Decimal("101"),
                high=Decimal("102"),
                low=Decimal("100"),
                close=Decimal("101.5"),
                volume=Decimal("11"),
            ),
        ]

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_refresh_symbol_once_writes_to_redis_when_dual_write_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = f"md:test:{uuid4().hex}"
    redis_store = RedisMarketDataStore(key_prefix=prefix)
    redis_sub_store = RedisSubscriptionStore(key_prefix=prefix)
    runtime = MarketDataRuntime(redis_store=redis_store, subscription_store=redis_sub_store)
    runtime.reset()
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", True)
    monkeypatch.setattr(market_data_tasks, "market_data_runtime", runtime)
    monkeypatch.setattr(market_data_tasks, "AlpacaRestProvider", _StubRefreshProvider)

    try:
        result = await market_data_tasks._refresh_symbol_once(market="stocks", symbol="AAPL")
        assert result["status"] == "ok"
        assert result["bars"] == 1

        quote = redis_store.get_latest_quote(market="stocks", symbol="AAPL")
        bars = redis_store.get_recent_bars(market="stocks", symbol="AAPL", timeframe="1m", limit=10)
        assert quote is not None
        assert quote.last == Decimal("100.5")
        assert len(bars) == 1
        assert bars[-1].close == pytest.approx(100.5)
    finally:
        redis_store.clear_prefix()
        redis_sub_store.clear_prefix()


@pytest.mark.asyncio
async def test_backfill_symbol_once_writes_to_redis_when_dual_write_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = f"md:test:{uuid4().hex}"
    redis_store = RedisMarketDataStore(key_prefix=prefix)
    redis_sub_store = RedisSubscriptionStore(key_prefix=prefix)
    runtime = MarketDataRuntime(redis_store=redis_store, subscription_store=redis_sub_store)
    runtime.reset()
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", True)
    monkeypatch.setattr(market_data_tasks, "market_data_runtime", runtime)
    monkeypatch.setattr(market_data_tasks, "AlpacaRestProvider", _StubBackfillProvider)

    try:
        result = await market_data_tasks._backfill_symbol_once(market="stocks", symbol="AAPL", minutes=5)
        assert result["status"] == "ok"
        assert result["bars"] == 2

        bars = redis_store.get_recent_bars(market="stocks", symbol="AAPL", timeframe="1m", limit=10)
        checkpoint = redis_store.get_checkpoint(market="stocks", symbol="AAPL", timeframe="1m")
        assert len(bars) == 2
        assert bars[-1].close == pytest.approx(101.5)
        assert checkpoint == int(datetime(2026, 2, 24, 10, 1, tzinfo=UTC).timestamp() * 1000)
    finally:
        redis_store.clear_prefix()
        redis_sub_store.clear_prefix()
