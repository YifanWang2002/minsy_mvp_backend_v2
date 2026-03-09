from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from packages.infra.providers.trading.adapters.base import OhlcvBar, OrderIntent, QuoteSnapshot
from packages.infra.providers.trading.adapters import sandbox_trading
from packages.infra.providers.trading.adapters.sandbox_trading import SandboxTradingAdapter


class _StubMarketDataProvider:
    def __init__(
        self,
        *,
        quote: QuoteSnapshot | None = None,
        latest_bar: OhlcvBar | None = None,
    ) -> None:
        self._quote = quote
        self._latest_bar = latest_bar

    async def fetch_quote(self, *, symbol: str, market: str) -> QuoteSnapshot | None:
        _ = (symbol, market)
        return self._quote

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar | None:
        _ = (symbol, market)
        return self._latest_bar

    async def fetch_recent_1m_bars(
        self,
        *,
        symbol: str,
        market: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        _ = (symbol, market, since, limit)
        if self._latest_bar is None:
            return []
        return [self._latest_bar]

    async def aclose(self) -> None:
        return None


class _FakeRedis:
    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._lists: dict[str, list[str]] = {}

    def get(self, key: str) -> str | None:
        return self._kv.get(key)

    def set(self, key: str, value: str) -> bool:
        self._kv[key] = value
        return True

    def hset(self, key: str, field: str, value: str) -> int:
        self._hashes.setdefault(key, {})[field] = value
        return 1

    def hget(self, key: str, field: str) -> str | None:
        row = self._hashes.get(key)
        if row is None:
            return None
        return row.get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def rpush(self, key: str, value: str) -> int:
        bucket = self._lists.setdefault(key, [])
        bucket.append(value)
        return len(bucket)

    def ltrim(self, key: str, start: int, end: int) -> bool:
        values = self._lists.get(key, [])
        self._lists[key] = values[start : end + 1 if end != -1 else None]
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self._lists.get(key, [])
        return values[start : end + 1 if end != -1 else None]


@pytest.fixture()
def _fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    client = _FakeRedis()
    monkeypatch.setattr(sandbox_trading, "get_sync_redis_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_submit_order_prefers_intent_submitted_mark_price(_fake_redis: _FakeRedis) -> None:
    _ = _fake_redis
    quote = QuoteSnapshot(
        symbol="AAPL",
        bid=Decimal("245.44"),
        ask=Decimal("271.56"),
        last=Decimal("258.50"),
        timestamp=datetime.now(UTC) - timedelta(minutes=5),
        raw={},
    )
    adapter = SandboxTradingAdapter(
        account_uid=f"sandbox-test-{uuid4().hex}",
        starting_cash=Decimal("100000"),
        slippage_bps=Decimal("0"),
        fee_bps=Decimal("0"),
        market_data_provider=_StubMarketDataProvider(quote=quote),
    )

    before = datetime.now(UTC) - timedelta(seconds=1)
    order = await adapter.submit_order(
        OrderIntent(
            client_order_id=f"coid-{uuid4().hex}",
            symbol="AAPL",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
            metadata={
                "market": "stocks",
                "submitted_mark_price": "257.42",
            },
        )
    )
    after = datetime.now(UTC) + timedelta(seconds=1)

    assert order.avg_fill_price == Decimal("257.42")
    assert order.submitted_at is not None
    assert before <= order.submitted_at <= after


@pytest.mark.asyncio
async def test_submit_order_uses_quote_mid_when_last_is_outside_spread(_fake_redis: _FakeRedis) -> None:
    _ = _fake_redis
    quote = QuoteSnapshot(
        symbol="AAPL",
        bid=Decimal("245.44"),
        ask=Decimal("271.56"),
        last=Decimal("300.00"),
        timestamp=datetime.now(UTC),
        raw={},
    )
    adapter = SandboxTradingAdapter(
        account_uid=f"sandbox-test-{uuid4().hex}",
        starting_cash=Decimal("100000"),
        slippage_bps=Decimal("0"),
        fee_bps=Decimal("0"),
        market_data_provider=_StubMarketDataProvider(quote=quote),
    )

    order = await adapter.submit_order(
        OrderIntent(
            client_order_id=f"coid-{uuid4().hex}",
            symbol="AAPL",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
            metadata={"market": "stocks"},
        )
    )

    expected_mid = (Decimal("245.44") + Decimal("271.56")) / Decimal("2")
    assert order.avg_fill_price == expected_mid
