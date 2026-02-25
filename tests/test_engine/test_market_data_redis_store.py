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


def test_redis_market_data_store_roundtrip_quote_bars_and_checkpoint() -> None:
    prefix = f"md:test:{uuid4().hex}"
    store = RedisMarketDataStore(key_prefix=prefix)
    try:
        quote = QuoteSnapshot(
            symbol="AAPL",
            bid=Decimal("100.1"),
            ask=Decimal("100.3"),
            last=Decimal("100.2"),
            timestamp=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
        )
        assert store.upsert_quote(market="stocks", symbol="aapl", quote=quote) is True

        loaded_quote = store.get_latest_quote(market="stocks", symbol="AAPL")
        assert loaded_quote is not None
        assert loaded_quote.symbol == "AAPL"
        assert loaded_quote.last == Decimal("100.2")

        for minute in range(3):
            bar_time = datetime(2026, 2, 24, 10, minute, tzinfo=UTC)
            assert (
                store.append_bar(
                    market="stocks",
                    symbol="AAPL",
                    timeframe="1m",
                    timestamp=bar_time,
                    open_=100 + minute,
                    high=101 + minute,
                    low=99 + minute,
                    close=100.5 + minute,
                    volume=10 + minute,
                )
                is True
            )

        bars = store.get_recent_bars(market="stocks", symbol="AAPL", timeframe="1m", limit=10)
        assert len(bars) == 3
        assert bars[-1].close == pytest.approx(102.5)

        latest_ts_ms = int(datetime(2026, 2, 24, 10, 2, tzinfo=UTC).timestamp() * 1000)
        checkpoint = store.get_checkpoint(market="stocks", symbol="AAPL", timeframe="1m")
        assert checkpoint == latest_ts_ms

        checkpoints = store.list_checkpoints()
        assert checkpoints["stocks:AAPL:1m"] == latest_ts_ms
    finally:
        store.clear_prefix()


def test_redis_subscription_store_bidirectional_index() -> None:
    prefix = f"md:test:{uuid4().hex}"
    store = RedisSubscriptionStore(key_prefix=prefix)
    try:
        assert (
            store.set_subscriber_instruments(
                "subscriber:1",
                (("stocks", "aapl"), ("crypto", "btcusd")),
            )
            is True
        )
        assert (
            store.subscriber_instruments("subscriber:1")
            == (("crypto", "BTCUSD"), ("stocks", "AAPL"))
        )
        assert store.subscribers_for_instrument(market="stocks", symbol="AAPL") == (
            "subscriber:1",
        )

        assert (
            store.set_subscriber_instruments(
                "subscriber:2",
                (("stocks", "AAPL"),),
            )
            is True
        )
        assert store.subscribers_for_instrument(market="stocks", symbol="AAPL") == (
            "subscriber:1",
            "subscriber:2",
        )
        assert ("stocks", "AAPL") in store.active_instruments()

        assert store.clear_subscriber("subscriber:1") is True
        assert store.subscriber_instruments("subscriber:1") == ()
        assert store.subscribers_for_instrument(market="stocks", symbol="AAPL") == (
            "subscriber:2",
        )
    finally:
        store.clear_prefix()


def test_market_data_runtime_dual_write_keeps_memory_and_redis_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = f"md:test:{uuid4().hex}"
    redis_store = RedisMarketDataStore(key_prefix=prefix)
    redis_sub_store = RedisSubscriptionStore(key_prefix=prefix)
    runtime = MarketDataRuntime(
        redis_store=redis_store,
        subscription_store=redis_sub_store,
    )
    runtime.reset()
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", True)
    monkeypatch.setattr(settings, "market_data_redis_subs_enabled", True)
    try:
        runtime.subscribe("deployment:123", ["aapl"], market="stocks")
        for minute in range(6):
            runtime.ingest_1m_bar(
                market="stocks",
                symbol="AAPL",
                bar=OhlcvBar(
                    timestamp=datetime(2026, 2, 24, 10, minute, tzinfo=UTC),
                    open=Decimal(100 + minute),
                    high=Decimal(101 + minute),
                    low=Decimal(99 + minute),
                    close=Decimal("100.5") + Decimal(minute),
                    volume=Decimal("10"),
                ),
            )
        runtime.upsert_quote(
            market="stocks",
            symbol="AAPL",
            quote=QuoteSnapshot(
                symbol="AAPL",
                bid=Decimal("105"),
                ask=Decimal("106"),
                last=Decimal("105.5"),
                timestamp=datetime(2026, 2, 24, 10, 5, tzinfo=UTC),
            ),
        )

        memory_bars = runtime.get_recent_bars(market="stocks", symbol="AAPL", timeframe="1m", limit=10)
        redis_bars = redis_store.get_recent_bars(market="stocks", symbol="AAPL", timeframe="1m", limit=10)
        assert len(memory_bars) == len(redis_bars) == 6
        assert redis_bars[-1].timestamp == memory_bars[-1].timestamp
        assert redis_bars[-1].close == pytest.approx(memory_bars[-1].close)

        memory_agg = runtime.get_recent_bars(market="stocks", symbol="AAPL", timeframe="5m", limit=10)
        redis_agg = redis_store.get_recent_bars(market="stocks", symbol="AAPL", timeframe="5m", limit=10)
        assert len(memory_agg) == len(redis_agg) == 1
        assert redis_agg[-1].close == pytest.approx(memory_agg[-1].close)

        redis_quote = redis_store.get_latest_quote(market="stocks", symbol="AAPL")
        assert redis_quote is not None
        assert redis_quote.last == Decimal("105.5")

        assert redis_sub_store.subscribers_for_instrument(market="stocks", symbol="AAPL") == (
            "deployment:123",
        )
        runtime.unsubscribe("deployment:123")
        assert redis_sub_store.subscribers_for_instrument(market="stocks", symbol="AAPL") == ()
    finally:
        redis_store.clear_prefix()
        redis_sub_store.clear_prefix()


def test_market_data_runtime_reads_redis_first_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = f"md:test:{uuid4().hex}"
    redis_store = RedisMarketDataStore(key_prefix=prefix)
    redis_sub_store = RedisSubscriptionStore(key_prefix=prefix)
    runtime = MarketDataRuntime(redis_store=redis_store, subscription_store=redis_sub_store)
    runtime.reset()
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", True)
    monkeypatch.setattr(settings, "market_data_redis_read_enabled", True)
    try:
        runtime.ingest_1m_bar(
            market="stocks",
            symbol="AAPL",
            bar=OhlcvBar(
                timestamp=datetime(2026, 2, 24, 10, 0, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.5"),
                volume=Decimal("10"),
            ),
        )
        assert (
            redis_store.append_bar(
                market="stocks",
                symbol="AAPL",
                timeframe="1m",
                timestamp=datetime(2026, 2, 24, 10, 1, tzinfo=UTC),
                open_=200,
                high=201,
                low=199,
                close=200.5,
                volume=99,
            )
            is True
        )
        bars = runtime.get_recent_bars(
            market="stocks",
            symbol="AAPL",
            timeframe="1m",
            limit=1,
        )
        assert len(bars) == 1
        assert bars[-1].close == pytest.approx(200.5)
    finally:
        redis_store.clear_prefix()
        redis_sub_store.clear_prefix()
