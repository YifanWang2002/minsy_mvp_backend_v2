from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

import packages.infra.providers.trading.adapters.sandbox_trading as sandbox_module
from packages.infra.db import session as db_session_module
from packages.infra.db.models.sandbox_ledger_entry import SandboxLedgerEntry
from packages.infra.providers.trading.adapters.base import (
    OhlcvBar,
    OrderIntent,
    QuoteSnapshot,
)
from packages.infra.providers.trading.adapters.sandbox_trading import SandboxTradingAdapter


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


def test_000_accessibility_sandbox_writes_postgres_ledger(
    api_test_client: TestClient,
    monkeypatch,
) -> None:
    _ = api_test_client
    fake_redis = _FakeRedis()
    fake_market_data = _FakeMarketDataProvider()
    monkeypatch.setattr(sandbox_module, "get_sync_redis_client", lambda: fake_redis)
    account_uid = f"sandbox-ledger-{uuid4().hex[:12]}"

    async def _exercise() -> None:
        adapter = SandboxTradingAdapter(
            account_uid=account_uid,
            starting_cash=Decimal("5000"),
            slippage_bps=Decimal("0"),
            fee_bps=Decimal("0"),
            slippage_bps_by_asset_class={"crypto": "8"},
            fee_bps_by_asset_class={"crypto": "15"},
            market_data_provider=fake_market_data,
        )
        await adapter.submit_order(
            OrderIntent(
                client_order_id=f"ledger-test-{uuid4().hex[:8]}",
                symbol="BTC/USD",
                side="buy",
                qty=Decimal("0.5"),
                order_type="market",
                metadata={"market": "crypto"},
            )
        )
        await adapter.aclose()

    assert api_test_client.portal is not None
    api_test_client.portal.call(_exercise)

    async def _load_latest() -> SandboxLedgerEntry | None:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            stmt = (
                select(SandboxLedgerEntry)
                .where(SandboxLedgerEntry.account_uid == account_uid)
                .order_by(SandboxLedgerEntry.happened_at.desc())
            )
            return await db.scalar(stmt)

    entry = api_test_client.portal.call(_load_latest)
    assert entry is not None
    assert entry.account_uid == account_uid
    assert entry.event_type == "order_fill"
    assert entry.asset_class == "crypto"
    assert Decimal(str(entry.fee)) > Decimal("0")
    assert Decimal(str(entry.cash_after)) < Decimal(str(entry.cash_before))
