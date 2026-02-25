"""Broker adapter abstractions for paper/live execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


class AdapterError(RuntimeError):
    """Raised when adapter operations fail at transport/provider level."""


@dataclass(frozen=True, slots=True)
class AccountState:
    """Normalized account-level balances and equity."""

    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    margin_used: Decimal = Decimal("0")
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """Normalized broker position snapshot."""

    symbol: str
    side: str
    qty: Decimal
    avg_entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Normalized fill event."""

    provider_fill_id: str
    provider_order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    fee: Decimal
    filled_at: datetime
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """Normalized order submission intent."""

    client_order_id: str
    symbol: str
    side: str
    qty: Decimal
    order_type: str = "market"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "gtc"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderState:
    """Normalized order status from provider."""

    provider_order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: Decimal
    filled_qty: Decimal
    status: str
    submitted_at: datetime | None
    avg_fill_price: Decimal | None = None
    reject_reason: str | None = None
    provider_updated_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OhlcvBar:
    """Standardized OHLCV bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """Standardized best bid/ask quote snapshot."""

    symbol: str
    bid: Decimal | None
    ask: Decimal | None
    last: Decimal | None
    timestamp: datetime
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketDataEvent:
    """Streaming market-data event."""

    channel: str
    symbol: str
    timestamp: datetime
    payload: dict[str, Any]


class BrokerAdapter(ABC):
    """Unified broker interface for provider implementations."""

    provider: str

    @abstractmethod
    async def fetch_account_state(self) -> AccountState:
        """Return normalized account balances and equity."""

    @abstractmethod
    async def fetch_positions(self) -> list[PositionRecord]:
        """Return normalized open position records."""

    @abstractmethod
    async def submit_order(self, intent: OrderIntent) -> OrderState:
        """Submit one order intent to provider."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel one provider order id."""

    @abstractmethod
    async def fetch_order(self, order_id: str) -> OrderState | None:
        """Fetch one order by provider order id."""

    @abstractmethod
    async def fetch_recent_fills(self, since: datetime | None = None) -> list[FillRecord]:
        """Fetch recent fills, optionally after a timestamp."""

    @abstractmethod
    async def fetch_ohlcv_1m(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        """Fetch historical 1m bars."""

    @abstractmethod
    async def fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:
        """Fetch latest single 1m bar."""

    @abstractmethod
    async def fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:
        """Fetch latest quote snapshot."""

    @abstractmethod
    def stream_market_data(self, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        """Stream provider market data events."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close any network/session resources."""
