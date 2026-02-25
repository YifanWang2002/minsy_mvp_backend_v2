"""In-process market-data runtime cache and query facade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from typing import Any

from src.config import settings
from src.engine.execution.adapters.base import OhlcvBar, QuoteSnapshot
from src.engine.market_data.aggregator import AggregatedBar, BarAggregator
from src.engine.market_data.factor_cache import FactorCache
from src.engine.market_data.redis_store import (
    RedisBar,
    RedisMarketDataStore,
    redis_market_data_store,
)
from src.engine.market_data.redis_subscription_store import (
    RedisSubscriptionStore,
    redis_subscription_store,
)
from src.engine.market_data.ring_buffer import OhlcvRing
from src.engine.market_data.subscription_registry import SubscriptionDelta


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    if normalized not in {"stocks", "crypto", "forex", "futures", "commodities"}:
        return "stocks"
    return normalized


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _to_ms(ts: datetime) -> int:
    return int(ts.astimezone(UTC).timestamp() * 1000)


def _from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)


def _to_float(value: Decimal | float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


@dataclass(frozen=True, slots=True)
class RuntimeBar:
    """Serialized bar object for REST/API consumers."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataRuntime:
    """Shared runtime for 1m bars, aggregated bars, quotes and factor cache."""

    def __init__(
        self,
        *,
        redis_store: RedisMarketDataStore | None = None,
        subscription_store: RedisSubscriptionStore | None = None,
    ) -> None:
        timeframes = tuple(
            timeframe.strip()
            for timeframe in settings.market_data_aggregate_timeframes_csv.split(",")
            if timeframe.strip()
        )
        self._aggregator = BarAggregator(
            timeframes=timeframes or ("5m", "15m", "1h", "1d"),
            timezone=settings.market_data_aggregate_timezone,
        )
        self._factor_cache = FactorCache(max_entries=settings.market_data_factor_cache_max_entries)
        self._rings_1m: dict[tuple[str, str], OhlcvRing] = {}
        self._rings_agg: dict[tuple[str, str, str], OhlcvRing] = {}
        self._quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._checkpoints: dict[str, int] = {}
        self._redis_store = redis_store or redis_market_data_store
        self._subscription_store = subscription_store or redis_subscription_store
        self._metrics: dict[str, int] = {
            "redis_read_hits": 0,
            "redis_read_misses": 0,
            "redis_read_errors": 0,
            "redis_write_errors": 0,
        }
        self._lock = RLock()

    @property
    def factor_cache(self) -> FactorCache:
        return self._factor_cache

    def _memory_cache_enabled(self) -> bool:
        return bool(settings.market_data_memory_cache_enabled)

    def _redis_write_enabled(self) -> bool:
        return bool(settings.effective_market_data_redis_write_enabled)

    def _redis_read_enabled(self) -> bool:
        return bool(settings.effective_market_data_redis_read_enabled)

    def _redis_subs_enabled(self) -> bool:
        # Subscription state is fully externalized to Redis across all environments.
        return True

    def _increment_metric(self, key: str, count: int = 1) -> None:
        with self._lock:
            self._metrics[key] = int(self._metrics.get(key, 0)) + max(1, int(count))

    def runtime_metrics(self) -> dict[str, object]:
        with self._lock:
            payload: dict[str, object] = dict(self._metrics)
        if self._redis_read_enabled() or self._redis_write_enabled():
            refresh = self._redis_store.get_refresh_scheduler_metrics()
            if refresh:
                payload["refresh_scheduler"] = refresh
        return payload

    def refresh_scheduler_metrics(self) -> dict[str, object]:
        return self._redis_store.get_refresh_scheduler_metrics()

    def record_refresh_scheduler_metrics(
        self,
        *,
        scheduled: int,
        deduped: int,
        total: int,
    ) -> None:
        self._redis_store.record_refresh_scheduler_metrics(
            scheduled=scheduled,
            deduped=deduped,
            total=total,
        )

    def redis_data_plane_status(self) -> dict[str, object]:
        enabled = self._redis_write_enabled() or self._redis_read_enabled() or self._redis_subs_enabled()
        if not enabled:
            return {
                "enabled": False,
                "available": True,
                "market_data_store_ok": True,
                "subscription_store_ok": True,
                "last_error": None,
            }
        market_data_store_ok = self._redis_store.ping()
        subscription_store_ok = True
        if self._redis_subs_enabled():
            subscription_store_ok = self._subscription_store.ping()
        return {
            "enabled": True,
            "available": bool(market_data_store_ok and subscription_store_ok),
            "market_data_store_ok": bool(market_data_store_ok),
            "subscription_store_ok": bool(subscription_store_ok),
            "last_error": self._redis_store.last_error() or self._subscription_store.last_error(),
        }

    def redis_read_error_recent(self, *, max_age_seconds: int = 90) -> bool:
        return bool(
            self._redis_store.has_recent_error(max_age_seconds=max_age_seconds)
            or self._subscription_store.has_recent_error(max_age_seconds=max_age_seconds)
        )

    def subscribe(self, subscriber_id: str, symbols: list[str], *, market: str = "stocks") -> SubscriptionDelta:
        normalized_subscriber_id = str(subscriber_id).strip()
        normalized_market = _normalize_market(market)
        requested_symbols = tuple(
            sorted(
                {
                    _normalize_symbol(item)
                    for item in symbols
                    if isinstance(item, str) and item.strip()
                }
            )
        )
        subscriber_instruments = tuple((normalized_market, symbol) for symbol in requested_symbols)
        previous_instruments = self._subscription_store.subscriber_instruments(normalized_subscriber_id)
        ok = self._subscription_store.set_subscriber_instruments(
            normalized_subscriber_id,
            subscriber_instruments,
        )
        if not ok:
            self._increment_metric("redis_write_errors")
        active_symbols = tuple(
            sorted(
                {
                    symbol
                    for _, symbol in self._subscription_store.active_instruments()
                }
            )
        )
        previous_symbols = {symbol for _, symbol in previous_instruments}
        requested_symbol_set = set(requested_symbols)
        return SubscriptionDelta(
            added_symbols=tuple(sorted(requested_symbol_set - previous_symbols)),
            removed_symbols=tuple(sorted(previous_symbols - requested_symbol_set)),
            active_symbols=active_symbols,
        )

    def unsubscribe(self, subscriber_id: str) -> SubscriptionDelta:
        normalized_subscriber_id = str(subscriber_id).strip()
        previous_instruments = self._subscription_store.subscriber_instruments(normalized_subscriber_id)
        ok = self._subscription_store.clear_subscriber(normalized_subscriber_id)
        if not ok:
            self._increment_metric("redis_write_errors")
        removed_symbols: list[str] = []
        for market, symbol in previous_instruments:
            if self._subscription_store.subscribers_for_instrument(
                market=market,
                symbol=symbol,
            ):
                continue
            removed_symbols.append(symbol)
        active_symbols = tuple(
            sorted(
                {
                    symbol
                    for _, symbol in self._subscription_store.active_instruments()
                }
            )
        )
        return SubscriptionDelta(
            added_symbols=(),
            removed_symbols=tuple(sorted(set(removed_symbols))),
            active_symbols=active_symbols,
        )

    def active_subscriptions(self) -> tuple[tuple[str, str], ...]:
        instruments = self._subscription_store.active_instruments()
        if instruments:
            self._increment_metric("redis_read_hits")
            return instruments
        if self._subscription_store.has_recent_error():
            self._increment_metric("redis_read_errors")
        else:
            self._increment_metric("redis_read_misses")
        return ()

    def redis_active_subscriptions(self) -> tuple[tuple[str, str], ...]:
        return self._subscription_store.active_instruments()

    def subscriber_instruments(self, subscriber_id: str) -> tuple[tuple[str, str], ...]:
        instruments = self._subscription_store.subscriber_instruments(subscriber_id)
        if instruments:
            self._increment_metric("redis_read_hits")
            return instruments
        if self._subscription_store.has_recent_error():
            self._increment_metric("redis_read_errors")
        else:
            self._increment_metric("redis_read_misses")
        return ()

    def subscriber_symbols(self, subscriber_id: str) -> tuple[str, ...]:
        instruments = self.subscriber_instruments(subscriber_id)
        return tuple(sorted({symbol for _, symbol in instruments}))

    def upsert_quote(self, *, market: str, symbol: str, quote: QuoteSnapshot) -> None:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        if self._memory_cache_enabled():
            with self._lock:
                key = (normalized_market, normalized_symbol)
                self._quotes[key] = quote
        if self._redis_write_enabled():
            ok = self._redis_store.upsert_quote(
                market=normalized_market,
                symbol=normalized_symbol,
                quote=quote,
            )
            if not ok:
                self._increment_metric("redis_write_errors")

    def ingest_1m_bar(self, *, market: str, symbol: str, bar: OhlcvBar) -> dict[str, RuntimeBar]:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        ts_ms = _to_ms(bar.timestamp)
        closed_runtime: dict[str, RuntimeBar] = {}
        closed_aggregated: tuple[tuple[str, AggregatedBar], ...] = ()
        with self._lock:
            if self._memory_cache_enabled():
                base_key = (normalized_market, normalized_symbol)
                ring = self._rings_1m.get(base_key)
                if ring is None:
                    ring = OhlcvRing.create(settings.market_data_ring_capacity_1m)
                    self._rings_1m[base_key] = ring
                ring.append(
                    ts_ms=ts_ms,
                    open_=_to_float(bar.open),
                    high=_to_float(bar.high),
                    low=_to_float(bar.low),
                    close=_to_float(bar.close),
                    volume=_to_float(bar.volume),
                )
                self._checkpoints[f"{normalized_market}:{normalized_symbol}:1m"] = ts_ms

            closed = self._aggregator.ingest_1m_bar(
                market=normalized_market,
                symbol=normalized_symbol,
                ts_ms=ts_ms,
                open_=_to_float(bar.open),
                high=_to_float(bar.high),
                low=_to_float(bar.low),
                close=_to_float(bar.close),
                volume=_to_float(bar.volume),
            )
            for timeframe, agg_bar in closed.items():
                if self._memory_cache_enabled():
                    self._append_aggregated_bar(
                        market=normalized_market,
                        symbol=normalized_symbol,
                        timeframe=timeframe,
                        bar=agg_bar,
                    )
                closed_runtime[timeframe] = self._runtime_bar_from_agg(agg_bar)
            closed_aggregated = tuple(closed.items())

        if self._redis_write_enabled():
            ok = self._redis_store.append_bar(
                market=normalized_market,
                symbol=normalized_symbol,
                timeframe="1m",
                timestamp=bar.timestamp,
                open_=_to_float(bar.open),
                high=_to_float(bar.high),
                low=_to_float(bar.low),
                close=_to_float(bar.close),
                volume=_to_float(bar.volume),
            )
            if not ok:
                self._increment_metric("redis_write_errors")
            for timeframe, agg_bar in closed_aggregated:
                ok = self._redis_store.append_bar(
                    market=normalized_market,
                    symbol=normalized_symbol,
                    timeframe=timeframe,
                    timestamp=_from_ms(agg_bar.ts_ms),
                    open_=agg_bar.open,
                    high=agg_bar.high,
                    low=agg_bar.low,
                    close=agg_bar.close,
                    volume=agg_bar.volume,
                )
                if not ok:
                    self._increment_metric("redis_write_errors")

        return closed_runtime

    def _append_aggregated_bar(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        bar: AggregatedBar,
    ) -> None:
        key = (market, symbol, timeframe)
        ring = self._rings_agg.get(key)
        if ring is None:
            ring = OhlcvRing.create(settings.market_data_ring_capacity_aggregated)
            self._rings_agg[key] = ring
        ring.append(
            ts_ms=bar.ts_ms,
            open_=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
        )
        self._checkpoints[f"{market}:{symbol}:{timeframe}"] = bar.ts_ms

    def get_latest_quote(self, *, market: str, symbol: str) -> QuoteSnapshot | None:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        if self._redis_read_enabled():
            quote = self._redis_store.get_latest_quote(
                market=normalized_market,
                symbol=normalized_symbol,
            )
            if quote is not None:
                self._increment_metric("redis_read_hits")
                return quote
            if self._redis_store.has_recent_error():
                self._increment_metric("redis_read_errors")
            else:
                self._increment_metric("redis_read_misses")
            return None
        return self._get_latest_quote_from_memory(
            market=normalized_market,
            symbol=normalized_symbol,
        )

    def get_recent_bars(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[RuntimeBar]:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_tf = timeframe.strip().lower()
        if self._redis_read_enabled():
            bars = self._redis_store.get_recent_bars(
                market=normalized_market,
                symbol=normalized_symbol,
                timeframe=normalized_tf,
                limit=limit,
            )
            if bars:
                self._increment_metric("redis_read_hits")
                return self._runtime_bars_from_redis_rows(bars)
            if self._redis_store.has_recent_error():
                self._increment_metric("redis_read_errors")
            else:
                self._increment_metric("redis_read_misses")
            return []
        return self._get_recent_bars_from_memory(
            market=normalized_market,
            symbol=normalized_symbol,
            timeframe=normalized_tf,
            limit=limit,
        )

    def get_checkpoint(self, *, market: str, symbol: str, timeframe: str = "1m") -> int | None:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_tf = timeframe.strip().lower()
        if self._redis_read_enabled():
            value = self._redis_store.get_checkpoint(
                market=normalized_market,
                symbol=normalized_symbol,
                timeframe=normalized_tf,
            )
            if value is not None:
                self._increment_metric("redis_read_hits")
                return value
            if self._redis_store.has_recent_error():
                self._increment_metric("redis_read_errors")
            else:
                self._increment_metric("redis_read_misses")
            return None
        with self._lock:
            return self._checkpoints.get(f"{normalized_market}:{normalized_symbol}:{normalized_tf}")

    def freshest_checkpoint_ms(self, *, timeframe: str = "1m") -> int | None:
        normalized_tf = timeframe.strip().lower()
        if self._redis_read_enabled():
            freshest = self._redis_store.freshest_checkpoint_ms(timeframe=normalized_tf)
            if freshest is not None:
                self._increment_metric("redis_read_hits")
                return freshest
            if self._redis_store.has_recent_error():
                self._increment_metric("redis_read_errors")
            else:
                self._increment_metric("redis_read_misses")
            return None
        checkpoints = self._memory_checkpoints()
        freshest: int | None = None
        suffix = f":{normalized_tf}"
        for key, ts_ms in checkpoints.items():
            if not key.endswith(suffix):
                continue
            if freshest is None or int(ts_ms) > freshest:
                freshest = int(ts_ms)
        return freshest

    def checkpoints(self) -> dict[str, int]:
        if self._redis_read_enabled():
            checkpoints = self._redis_store.list_checkpoints()
            if checkpoints:
                self._increment_metric("redis_read_hits")
                return checkpoints
            if self._redis_store.has_recent_error():
                self._increment_metric("redis_read_errors")
            else:
                self._increment_metric("redis_read_misses")
            return {}
        return self._memory_checkpoints()

    def restore_checkpoints(self, checkpoint: dict[str, int]) -> None:
        with self._lock:
            if self._memory_cache_enabled():
                self._checkpoints = dict(checkpoint)
            else:
                self._checkpoints.clear()
            self._aggregator.restore_checkpoint(checkpoint)

    def reset(self) -> None:
        with self._lock:
            self._rings_1m.clear()
            self._rings_agg.clear()
            self._quotes.clear()
            self._checkpoints.clear()
            self._factor_cache.clear()
            for key in self._metrics:
                self._metrics[key] = 0

    def _get_latest_quote_from_memory(self, *, market: str, symbol: str) -> QuoteSnapshot | None:
        with self._lock:
            return self._quotes.get((market, symbol))

    def _get_recent_bars_from_memory(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[RuntimeBar]:
        if not self._memory_cache_enabled():
            return []
        with self._lock:
            if timeframe == "1m":
                ring = self._rings_1m.get((market, symbol))
            else:
                ring = self._rings_agg.get((market, symbol, timeframe))
            if ring is None:
                return []
            payload = ring.latest(limit)
        return self._runtime_bars_from_arrays(payload)

    def _memory_checkpoints(self) -> dict[str, int]:
        with self._lock:
            return dict(self._checkpoints)

    def _runtime_bars_from_redis_rows(self, rows: list[RedisBar]) -> list[RuntimeBar]:
        return [
            RuntimeBar(
                timestamp=item.timestamp,
                open=float(item.open),
                high=float(item.high),
                low=float(item.low),
                close=float(item.close),
                volume=float(item.volume),
            )
            for item in rows
        ]

    def _runtime_bar_from_agg(self, bar: AggregatedBar) -> RuntimeBar:
        return RuntimeBar(
            timestamp=datetime.fromtimestamp(bar.ts_ms / 1000.0, tz=UTC),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )

    def _runtime_bars_from_arrays(self, payload: dict[str, Any]) -> list[RuntimeBar]:
        ts = payload.get("ts", [])
        open_ = payload.get("open", [])
        high = payload.get("high", [])
        low = payload.get("low", [])
        close = payload.get("close", [])
        volume = payload.get("volume", [])
        rows: list[RuntimeBar] = []
        for idx in range(len(ts)):
            rows.append(
                RuntimeBar(
                    timestamp=datetime.fromtimestamp(int(ts[idx]) / 1000.0, tz=UTC),
                    open=float(open_[idx]),
                    high=float(high[idx]),
                    low=float(low[idx]),
                    close=float(close[idx]),
                    volume=float(volume[idx]),
                )
            )
        return rows


market_data_runtime = MarketDataRuntime()
