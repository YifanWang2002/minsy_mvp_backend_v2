from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.engine.execution.adapters.base import (
    AccountState,
    BrokerAdapter,
    FillRecord,
    MarketDataEvent,
    OhlcvBar,
    OrderIntent,
    OrderState,
    PositionRecord,
    QuoteSnapshot,
)


class _MockAdapter(BrokerAdapter):
    provider = "mock"

    async def fetch_account_state(self) -> AccountState:
        return AccountState(
            cash=Decimal("1000"),
            equity=Decimal("1200"),
            buying_power=Decimal("2000"),
            margin_used=Decimal("100"),
        )

    async def fetch_positions(self) -> list[PositionRecord]:
        return [
            PositionRecord(
                symbol="AAPL",
                side="long",
                qty=Decimal("1"),
                avg_entry_price=Decimal("100"),
                mark_price=Decimal("101"),
                unrealized_pnl=Decimal("1"),
            )
        ]

    async def submit_order(self, intent: OrderIntent) -> OrderState:
        return OrderState(
            provider_order_id="provider-1",
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            qty=intent.qty,
            filled_qty=Decimal("0"),
            status="new",
            submitted_at=datetime.now(UTC),
        )

    async def cancel_order(self, order_id: str) -> bool:
        return bool(order_id)

    async def fetch_order(self, order_id: str) -> OrderState | None:
        return OrderState(
            provider_order_id=order_id,
            client_order_id="client-1",
            symbol="AAPL",
            side="buy",
            order_type="market",
            qty=Decimal("1"),
            filled_qty=Decimal("1"),
            status="filled",
            submitted_at=datetime.now(UTC),
            avg_fill_price=Decimal("101"),
        )

    async def fetch_recent_fills(self, since: datetime | None = None) -> list[FillRecord]:
        _ = since
        return [
            FillRecord(
                provider_fill_id="fill-1",
                provider_order_id="provider-1",
                symbol="AAPL",
                side="buy",
                qty=Decimal("1"),
                price=Decimal("101"),
                fee=Decimal("0.1"),
                filled_at=datetime.now(UTC),
            )
        ]

    async def fetch_ohlcv_1m(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        _ = (symbol, since, limit)
        return [
            OhlcvBar(
                timestamp=datetime.now(UTC),
                open=Decimal("1"),
                high=Decimal("2"),
                low=Decimal("0.5"),
                close=Decimal("1.5"),
                volume=Decimal("10"),
            )
        ]

    async def fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:
        _ = symbol
        return OhlcvBar(
            timestamp=datetime.now(UTC),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("0.5"),
            close=Decimal("1.5"),
            volume=Decimal("10"),
        )

    async def fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        return QuoteSnapshot(
            symbol=symbol,
            bid=Decimal("100"),
            ask=Decimal("101"),
            last=Decimal("100.5"),
            timestamp=datetime.now(UTC),
        )

    async def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        for symbol in symbols:
            yield MarketDataEvent(
                channel="quote",
                symbol=symbol,
                timestamp=datetime.now(UTC),
                payload={"last": "100.5"},
            )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_broker_adapter_contract() -> None:
    adapter = _MockAdapter()

    account = await adapter.fetch_account_state()
    assert account.cash == Decimal("1000")

    positions = await adapter.fetch_positions()
    assert positions and positions[0].symbol == "AAPL"

    order = await adapter.submit_order(
        OrderIntent(
            client_order_id="client-1",
            symbol="AAPL",
            side="buy",
            qty=Decimal("1"),
        )
    )
    assert order.client_order_id == "client-1"

    assert await adapter.cancel_order(order.provider_order_id) is True

    fetched_order = await adapter.fetch_order(order.provider_order_id)
    assert fetched_order is not None and fetched_order.status == "filled"

    fills = await adapter.fetch_recent_fills()
    assert fills and fills[0].provider_fill_id == "fill-1"

    bars = await adapter.fetch_ohlcv_1m("AAPL")
    assert len(bars) == 1

    latest_bar = await adapter.fetch_latest_1m_bar("AAPL")
    assert latest_bar is not None

    quote = await adapter.fetch_latest_quote("AAPL")
    assert quote is not None and quote.symbol == "AAPL"

    events = [event async for event in adapter.stream_market_data(["AAPL", "TSLA"])]
    assert [event.symbol for event in events] == ["AAPL", "TSLA"]

    await adapter.aclose()
