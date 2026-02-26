"""In-memory subscription dedup for market-data streaming."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubscriptionDelta:
    """Tracks underlying stream changes after one subscribe/unsubscribe call."""

    added_symbols: tuple[str, ...]
    removed_symbols: tuple[str, ...]
    active_symbols: tuple[str, ...]


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    if normalized not in {"stocks", "crypto", "forex", "futures", "commodities"}:
        return "stocks"
    return normalized


def _instrument_key(market: str, symbol: str) -> str:
    return f"{_normalize_market(market)}|{_normalize_symbol(symbol)}"


def _split_instrument_key(key: str) -> tuple[str, str]:
    market, symbol = key.split("|", 1)
    return market, symbol


class SubscriptionRegistry:
    """Deduplicate subscriptions across users/deployments by symbol."""

    def __init__(self) -> None:
        self._by_subscriber: dict[str, set[str]] = defaultdict(set)
        self._ref_count: dict[str, int] = defaultdict(int)

    def subscribe(
        self,
        subscriber_id: str,
        symbols: list[str],
        *,
        market: str = "stocks",
    ) -> SubscriptionDelta:
        normalized = {_instrument_key(market, symbol) for symbol in symbols if symbol.strip()}
        previous = set(self._by_subscriber.get(subscriber_id, set()))
        to_add = normalized - previous

        added_symbols: list[str] = []
        for instrument in sorted(to_add):
            self._ref_count[instrument] += 1
            if self._ref_count[instrument] == 1:
                _, symbol = _split_instrument_key(instrument)
                added_symbols.append(symbol)

        self._by_subscriber[subscriber_id] = normalized
        return SubscriptionDelta(
            added_symbols=tuple(added_symbols),
            removed_symbols=(),
            active_symbols=tuple(self.active_symbols()),
        )

    def unsubscribe(self, subscriber_id: str) -> SubscriptionDelta:
        previous = self._by_subscriber.pop(subscriber_id, set())
        removed_symbols: list[str] = []
        for instrument in sorted(previous):
            next_count = self._ref_count.get(instrument, 0) - 1
            if next_count <= 0:
                self._ref_count.pop(instrument, None)
                _, symbol = _split_instrument_key(instrument)
                removed_symbols.append(symbol)
            else:
                self._ref_count[instrument] = next_count
        return SubscriptionDelta(
            added_symbols=(),
            removed_symbols=tuple(removed_symbols),
            active_symbols=tuple(self.active_symbols()),
        )

    def active_symbols(self) -> tuple[str, ...]:
        symbols = {
            _split_instrument_key(instrument)[1]
            for instrument, count in self._ref_count.items()
            if count > 0
        }
        return tuple(sorted(symbols))

    def active_instruments(self) -> tuple[tuple[str, str], ...]:
        rows = [
            _split_instrument_key(instrument)
            for instrument, count in self._ref_count.items()
            if count > 0
        ]
        return tuple(sorted(rows))

    def subscriber_symbols(self, subscriber_id: str) -> tuple[str, ...]:
        symbols = {_split_instrument_key(key)[1] for key in self._by_subscriber.get(subscriber_id, set())}
        return tuple(sorted(symbols))

    def subscriber_instruments(self, subscriber_id: str) -> tuple[tuple[str, str], ...]:
        rows = [_split_instrument_key(key) for key in self._by_subscriber.get(subscriber_id, set())]
        return tuple(sorted(rows))

    def subscriber_count(self, symbol: str, *, market: str | None = None) -> int:
        if market is not None:
            return int(self._ref_count.get(_instrument_key(market, symbol), 0))
        normalized_symbol = _normalize_symbol(symbol)
        return sum(
            int(count)
            for instrument, count in self._ref_count.items()
            if _split_instrument_key(instrument)[1] == normalized_symbol
        )
