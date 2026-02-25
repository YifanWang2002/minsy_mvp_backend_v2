"""Shared factor cache for multi-strategy dedup."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


def factor_signature(
    *,
    factor_type: str,
    params: dict[str, Any],
    source: str,
) -> str:
    payload = {
        "factor_type": factor_type.strip().lower(),
        "params": params,
        "source": source.strip().lower(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha1(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorCacheStats:
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class FactorCache:
    """Simple LRU cache keyed by market/symbol/timeframe/signature/bar timestamp."""

    def __init__(self, *, max_entries: int = 200_000) -> None:
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, str, str, str, int], Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        signature: str,
        bar_ts: int,
        compute: Callable[[], T],
    ) -> T:
        key = (market, symbol, timeframe, signature, bar_ts)
        if key in self._cache:
            self._hits += 1
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        self._misses += 1
        value = compute()
        self._cache[key] = value
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return value

    def evict_before(self, *, bar_ts: int) -> int:
        removed = 0
        keys = [key for key in self._cache if key[-1] < bar_ts]
        for key in keys:
            self._cache.pop(key, None)
            removed += 1
        return removed

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> FactorCacheStats:
        return FactorCacheStats(hits=self._hits, misses=self._misses)
