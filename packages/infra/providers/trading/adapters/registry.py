"""Adapter provider registry."""

from __future__ import annotations

from collections.abc import Callable

from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter
from packages.infra.providers.trading.adapters.base import BrokerAdapter

AdapterFactory = Callable[[], BrokerAdapter]


class AdapterRegistry:
    """Runtime registry mapping provider name to adapter factory."""

    def __init__(self) -> None:
        self._providers: dict[str, AdapterFactory] = {}

    def register(self, provider: str, factory: AdapterFactory) -> None:
        self._providers[provider.strip().lower()] = factory

    def get(self, provider: str) -> AdapterFactory | None:
        return self._providers.get(provider.strip().lower())

    def create(self, provider: str) -> BrokerAdapter:
        factory = self.get(provider)
        if factory is None:
            raise ValueError(f"Unsupported broker provider: {provider}")
        return factory()

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


adapter_registry = AdapterRegistry()
adapter_registry.register("alpaca", AlpacaTradingAdapter)
