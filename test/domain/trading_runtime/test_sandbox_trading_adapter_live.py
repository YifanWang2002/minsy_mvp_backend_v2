from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import packages.infra.providers.trading.adapters.sandbox_trading as sandbox_module
from packages.infra.providers.trading.adapters.base import (
    OhlcvBar,
    OrderIntent,
    QuoteSnapshot,
)
from packages.infra.providers.trading.adapters.sandbox_trading import (
    SandboxTradingAdapter,
)


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
        bucket = self._hashes.setdefault(key, {})
        bucket[field] = value
        return 1

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def rpush(self, key: str, value: str) -> int:
        bucket = self._lists.setdefault(key, [])
        bucket.append(value)
        return len(bucket)

    def ltrim(self, key: str, start: int, end: int) -> bool:
        bucket = self._lists.setdefault(key, [])
        n = len(bucket)
        if n == 0:
            return True
        start_idx = start if start >= 0 else max(0, n + start)
        end_idx = end if end >= 0 else n + end
        end_idx = min(n - 1, end_idx)
        if start_idx > end_idx:
            self._lists[key] = []
            return True
        self._lists[key] = bucket[start_idx : end_idx + 1]
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        bucket = self._lists.get(key, [])
        n = len(bucket)
        if n == 0:
            return []
        start_idx = start if start >= 0 else max(0, n + start)
        end_idx = end if end >= 0 else n + end
        end_idx = min(n - 1, end_idx)
        if start_idx > end_idx:
            return []
        return bucket[start_idx : end_idx + 1]


class _FakeMarketDataProvider:
    def __init__(self) -> None:
        self.price = Decimal("100")

    async def fetch_quote(self, *, symbol: str, market: str) -> QuoteSnapshot | None:
        _ = market
        return QuoteSnapshot(
            symbol=symbol,
            bid=self.price - Decimal("0.1"),
            ask=self.price + Decimal("0.1"),
            last=self.price,
            timestamp=datetime.now(UTC),
            raw={"source": "fake_quote"},
        )

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str) -> OhlcvBar | None:
        _ = market
        return OhlcvBar(
            timestamp=datetime.now(UTC),
            open=self.price,
            high=self.price,
            low=self.price,
            close=self.price,
            volume=Decimal("1"),
        )

    async def fetch_recent_1m_bars(
        self,
        *,
        symbol: str,
        market: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        _ = market
        _ = since
        safe_limit = max(1, int(limit))
        return [
            OhlcvBar(
                timestamp=datetime.now(UTC),
                open=self.price,
                high=self.price,
                low=self.price,
                close=self.price,
                volume=Decimal("1"),
            )
            for _ in range(safe_limit)
        ]

    async def aclose(self) -> None:
        return None


async def test_000_accessibility_sandbox_submit_and_account_snapshot(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    fake_market_data = _FakeMarketDataProvider()
    monkeypatch.setattr(sandbox_module, "get_sync_redis_client", lambda: fake_redis)

    adapter = SandboxTradingAdapter(
        account_uid="sandbox-test-000",
        starting_cash=Decimal("10000"),
        market_data_provider=fake_market_data,
    )
    buy_state = await adapter.submit_order(
        OrderIntent(
            client_order_id="sandbox-buy-1",
            symbol="BTC/USD",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
        )
    )
    assert buy_state.status == "filled"
    assert buy_state.avg_fill_price is not None

    positions = await adapter.fetch_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTC/USD"
    assert positions[0].side == "long"
    assert positions[0].qty == Decimal("1")

    account = await adapter.fetch_account_state()
    assert account.cash < Decimal("10000")
    assert account.equity > Decimal("0")

    fake_market_data.price = Decimal("101")
    sell_state = await adapter.submit_order(
        OrderIntent(
            client_order_id="sandbox-sell-1",
            symbol="BTC/USD",
            side="sell",
            qty=Decimal("1"),
            order_type="market",
        )
    )
    assert sell_state.status == "filled"

    fills = await adapter.fetch_recent_fills()
    assert len(fills) >= 2
    await adapter.aclose()


async def test_010_accessibility_sandbox_market_data_passthrough(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    fake_market_data = _FakeMarketDataProvider()
    monkeypatch.setattr(sandbox_module, "get_sync_redis_client", lambda: fake_redis)

    adapter = SandboxTradingAdapter(
        account_uid="sandbox-test-010",
        market_data_provider=fake_market_data,
    )
    bars = await adapter.fetch_ohlcv_1m("BTC/USD", limit=3)
    assert len(bars) == 3
    latest = await adapter.fetch_latest_1m_bar("BTC/USD")
    assert latest is not None
    quote = await adapter.fetch_latest_quote("BTC/USD")
    assert quote is not None
    await adapter.aclose()


async def test_020_accessibility_sandbox_fee_and_slippage_profiles(monkeypatch) -> None:
    fake_redis = _FakeRedis()
    fake_market_data = _FakeMarketDataProvider()
    monkeypatch.setattr(sandbox_module, "get_sync_redis_client", lambda: fake_redis)

    ledger_events: list[dict[str, str]] = []

    async def _capture_ledger(_self, payload: dict[str, str]) -> None:
        ledger_events.append(payload)

    monkeypatch.setattr(SandboxTradingAdapter, "_append_ledger_entry", _capture_ledger)

    adapter = SandboxTradingAdapter(
        account_uid="sandbox-test-020",
        starting_cash=Decimal("10000"),
        slippage_bps=Decimal("0"),
        fee_bps=Decimal("0"),
        slippage_bps_by_asset_class={"crypto": "10", "us_equity": "5"},
        fee_bps_by_asset_class={"crypto": "20", "us_equity": "1"},
        market_data_provider=fake_market_data,
    )

    crypto_order = await adapter.submit_order(
        OrderIntent(
            client_order_id="sandbox-crypto-buy-1",
            symbol="BTC/USD",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
            metadata={"market": "crypto"},
        )
    )
    assert crypto_order.status == "filled"
    assert crypto_order.avg_fill_price is not None
    assert crypto_order.raw.get("asset_class") == "crypto"
    assert crypto_order.raw.get("slippage_bps") == "10"
    assert crypto_order.raw.get("fee_bps") == "20"

    equity_order = await adapter.submit_order(
        OrderIntent(
            client_order_id="sandbox-equity-buy-1",
            symbol="AAPL",
            side="buy",
            qty=Decimal("1"),
            order_type="market",
            metadata={"market": "stocks"},
        )
    )
    assert equity_order.status == "filled"
    assert equity_order.raw.get("asset_class") == "us_equity"
    assert equity_order.raw.get("slippage_bps") == "5"
    assert equity_order.raw.get("fee_bps") == "1"

    fills = await adapter.fetch_recent_fills()
    assert len(fills) >= 2
    latest_fill = fills[-1]
    assert latest_fill.fee > Decimal("0")

    positions = await adapter.fetch_positions()
    assert positions
    # Opening trades should still reflect fee drag in realized_pnl.
    assert any(row.realized_pnl < Decimal("0") for row in positions)

    assert len(ledger_events) >= 2
    assert ledger_events[0]["asset_class"] == "crypto"
    assert ledger_events[1]["asset_class"] == "us_equity"
    await adapter.aclose()


def test_030_accessibility_sandbox_asset_class_inference() -> None:
    assert sandbox_module._infer_asset_class("BTC/USD") == "crypto"
    assert sandbox_module._infer_asset_class("AAPL") == "us_equity"
    assert sandbox_module._infer_asset_class("EUR/USD") == "forex"
    assert sandbox_module._infer_asset_class("ES=F") == "futures"
