"""Execution adapters for broker providers."""

from typing import TYPE_CHECKING, Any

from packages.infra.providers.trading.adapters.base import (
    AccountState,
    AdapterError,
    BrokerAdapter,
    FillRecord,
    MarketDataEvent,
    OhlcvBar,
    OrderIntent,
    OrderState,
    PositionRecord,
    QuoteSnapshot,
)

if TYPE_CHECKING:
    from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
    from packages.infra.providers.trading.adapters.registry import AdapterRegistry


def __getattr__(name: str) -> Any:
    if name == "AlpacaTradingAdapter":
        from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter

        return AlpacaTradingAdapter
    if name == "AdapterRegistry":
        from packages.infra.providers.trading.adapters.registry import AdapterRegistry

        return AdapterRegistry
    if name == "adapter_registry":
        from packages.infra.providers.trading.adapters.registry import adapter_registry

        return adapter_registry
    raise AttributeError(name)


__all__ = [
    "AccountState",
    "AdapterError",
    "AlpacaTradingAdapter",
    "BrokerAdapter",
    "FillRecord",
    "MarketDataEvent",
    "OhlcvBar",
    "OrderIntent",
    "OrderState",
    "PositionRecord",
    "QuoteSnapshot",
    "AdapterRegistry",
    "adapter_registry",
]
