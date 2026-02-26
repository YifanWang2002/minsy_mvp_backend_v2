"""Redis-backed market-data store for quotes, bars and checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from threading import RLock
from time import time

from packages.infra.observability.logger import logger
from packages.infra.providers.trading.adapters.base import QuoteSnapshot
from packages.infra.redis.client import get_sync_redis_client
from packages.shared_settings.schema.settings import settings

_SUPPORTED_MARKETS = {"stocks", "crypto", "forex", "futures", "commodities"}


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    if normalized not in _SUPPORTED_MARKETS:
        return "stocks"
    return normalized


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return timeframe.strip().lower() or "1m"


def _to_ms(ts: datetime) -> int:
    return int(ts.astimezone(UTC).timestamp() * 1000)


def _to_float(value: Decimal | float | int | str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_decimal_or_none(value: str | None) -> Decimal | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def _to_datetime_or_none(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class RedisBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class RedisMarketDataStore:
    """Thin Redis persistence layer for market-data runtime objects."""

    def __init__(
        self,
        *,
        key_prefix: str = "md:v1",
        checkpoint_ttl_seconds: int | None = None,
        ring_capacity_1m: int | None = None,
        ring_capacity_aggregated: int | None = None,
    ) -> None:
        self.key_prefix = key_prefix.strip() or "md:v1"
        self.checkpoint_ttl_seconds = (
            max(60, int(checkpoint_ttl_seconds))
            if checkpoint_ttl_seconds is not None
            else int(settings.market_data_checkpoint_ttl_seconds)
        )
        self.ring_capacity_1m = (
            max(10, int(ring_capacity_1m))
            if ring_capacity_1m is not None
            else int(settings.market_data_ring_capacity_1m)
        )
        self.ring_capacity_aggregated = (
            max(10, int(ring_capacity_aggregated))
            if ring_capacity_aggregated is not None
            else int(settings.market_data_ring_capacity_aggregated)
        )
        self._last_error: dict[str, object] | None = None
        self._error_lock = RLock()

    def _quote_key(self, *, market: str, symbol: str) -> str:
        return f"{self.key_prefix}:quote:{market}:{symbol}"

    def _bars_key(self, *, timeframe: str, market: str, symbol: str) -> str:
        return f"{self.key_prefix}:bars:{timeframe}:{market}:{symbol}"

    def _checkpoint_key(self, *, market: str, symbol: str, timeframe: str) -> str:
        return f"{self.key_prefix}:ckpt:{market}:{symbol}:{timeframe}"

    def _instrument_prefix(self) -> str:
        return f"{self.key_prefix}:ckpt:"

    def _bars_maxlen(self, timeframe: str) -> int:
        return self.ring_capacity_1m if timeframe == "1m" else self.ring_capacity_aggregated

    def _refresh_metrics_key(self) -> str:
        return f"{self.key_prefix}:metrics:refresh_active_subscriptions"

    def _record_error(self, operation: str, exc: Exception) -> None:
        with self._error_lock:
            self._last_error = {
                "operation": operation,
                "error_type": type(exc).__name__,
                "message": str(exc)[:240],
                "ts": int(time()),
            }

    def _clear_error(self) -> None:
        with self._error_lock:
            self._last_error = None

    def has_recent_error(self, *, max_age_seconds: int = 90) -> bool:
        with self._error_lock:
            payload = dict(self._last_error) if isinstance(self._last_error, dict) else None
        if payload is None:
            return False
        try:
            ts = int(payload.get("ts", 0))
        except (TypeError, ValueError):
            return True
        return int(time()) - ts <= max(1, int(max_age_seconds))

    def last_error(self) -> dict[str, object] | None:
        with self._error_lock:
            if not isinstance(self._last_error, dict):
                return None
            return dict(self._last_error)

    def ping(self) -> bool:
        try:
            client = get_sync_redis_client()
            ok = bool(client.ping())
            if ok:
                self._clear_error()
            return ok
        except Exception as exc:  # noqa: BLE001
            self._record_error("ping", exc)
            logger.warning("redis market-data ping failed error=%s", type(exc).__name__)
            return False

    def upsert_quote(self, *, market: str, symbol: str, quote: QuoteSnapshot) -> bool:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        key = self._quote_key(market=normalized_market, symbol=normalized_symbol)
        payload = {
            "symbol": normalized_symbol,
            "bid": "" if quote.bid is None else str(quote.bid),
            "ask": "" if quote.ask is None else str(quote.ask),
            "last": "" if quote.last is None else str(quote.last),
            "ts_ms": str(_to_ms(quote.timestamp)),
        }
        try:
            client = get_sync_redis_client()
            client.hset(key, mapping=payload)
            self._clear_error()
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error("upsert_quote", exc)
            logger.warning(
                "redis market-data quote write failed market=%s symbol=%s error=%s",
                normalized_market,
                normalized_symbol,
                type(exc).__name__,
            )
            return False

    def get_latest_quote(self, *, market: str, symbol: str) -> QuoteSnapshot | None:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        key = self._quote_key(market=normalized_market, symbol=normalized_symbol)
        try:
            client = get_sync_redis_client()
            payload = client.hgetall(key)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("get_latest_quote", exc)
            logger.warning(
                "redis market-data quote read failed market=%s symbol=%s error=%s",
                normalized_market,
                normalized_symbol,
                type(exc).__name__,
            )
            return None
        if not payload:
            return None
        timestamp = _to_datetime_or_none(payload.get("ts_ms")) or datetime.now(UTC)
        return QuoteSnapshot(
            symbol=str(payload.get("symbol") or normalized_symbol),
            bid=_to_decimal_or_none(payload.get("bid")),
            ask=_to_decimal_or_none(payload.get("ask")),
            last=_to_decimal_or_none(payload.get("last")),
            timestamp=timestamp,
            raw={
                "market": normalized_market,
                "source": "redis",
            },
        )

    def append_bar(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        open_: float | Decimal | int,
        high: float | Decimal | int,
        low: float | Decimal | int,
        close: float | Decimal | int,
        volume: float | Decimal | int,
    ) -> bool:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_tf = _normalize_timeframe(timeframe)
        ts_ms = _to_ms(timestamp)
        encoded_bar = json.dumps(
            {
                "ts_ms": ts_ms,
                "open": _to_float(open_),
                "high": _to_float(high),
                "low": _to_float(low),
                "close": _to_float(close),
                "volume": _to_float(volume),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        bars_key = self._bars_key(
            timeframe=normalized_tf,
            market=normalized_market,
            symbol=normalized_symbol,
        )
        checkpoint_key = self._checkpoint_key(
            market=normalized_market,
            symbol=normalized_symbol,
            timeframe=normalized_tf,
        )
        try:
            client = get_sync_redis_client()
            pipeline = client.pipeline(transaction=True)
            pipeline.lpush(bars_key, encoded_bar)
            pipeline.ltrim(bars_key, 0, self._bars_maxlen(normalized_tf) - 1)
            pipeline.set(checkpoint_key, str(ts_ms), ex=self.checkpoint_ttl_seconds)
            pipeline.execute()
            self._clear_error()
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error("append_bar", exc)
            logger.warning(
                "redis market-data bar write failed market=%s symbol=%s tf=%s error=%s",
                normalized_market,
                normalized_symbol,
                normalized_tf,
                type(exc).__name__,
            )
            return False

    def get_recent_bars(
        self,
        *,
        market: str,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[RedisBar]:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_tf = _normalize_timeframe(timeframe)
        safe_limit = max(1, int(limit))
        key = self._bars_key(
            timeframe=normalized_tf,
            market=normalized_market,
            symbol=normalized_symbol,
        )
        try:
            client = get_sync_redis_client()
            payload = client.lrange(key, 0, safe_limit - 1)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("get_recent_bars", exc)
            logger.warning(
                "redis market-data bars read failed market=%s symbol=%s tf=%s error=%s",
                normalized_market,
                normalized_symbol,
                normalized_tf,
                type(exc).__name__,
            )
            return []
        rows: list[RedisBar] = []
        for raw in reversed(payload):
            try:
                decoded = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, dict):
                continue
            ts_ms = decoded.get("ts_ms")
            try:
                timestamp = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC)
            except (TypeError, ValueError):
                continue
            rows.append(
                RedisBar(
                    timestamp=timestamp,
                    open=_to_float(decoded.get("open")),
                    high=_to_float(decoded.get("high")),
                    low=_to_float(decoded.get("low")),
                    close=_to_float(decoded.get("close")),
                    volume=_to_float(decoded.get("volume")),
                )
            )
        return rows

    def get_checkpoint(self, *, market: str, symbol: str, timeframe: str) -> int | None:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_tf = _normalize_timeframe(timeframe)
        key = self._checkpoint_key(
            market=normalized_market,
            symbol=normalized_symbol,
            timeframe=normalized_tf,
        )
        try:
            client = get_sync_redis_client()
            value = client.get(key)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("get_checkpoint", exc)
            logger.warning(
                "redis checkpoint read failed market=%s symbol=%s tf=%s error=%s",
                normalized_market,
                normalized_symbol,
                normalized_tf,
                type(exc).__name__,
            )
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def list_checkpoints(self) -> dict[str, int]:
        pattern = f"{self._instrument_prefix()}*"
        rows: dict[str, int] = {}
        try:
            client = get_sync_redis_client()
            for key in client.scan_iter(match=pattern, count=200):
                raw_value = client.get(str(key))
                if raw_value is None:
                    continue
                try:
                    value = int(raw_value)
                except (TypeError, ValueError):
                    continue
                try:
                    head, market, symbol, timeframe = str(key).rsplit(":", 3)
                except ValueError:
                    continue
                if not head.endswith(":ckpt"):
                    continue
                rows[f"{market}:{symbol}:{timeframe}"] = value
            self._clear_error()
            return rows
        except Exception as exc:  # noqa: BLE001
            self._record_error("list_checkpoints", exc)
            logger.warning("redis checkpoints scan failed error=%s", type(exc).__name__)
            return {}

    def freshest_checkpoint_ms(self, *, timeframe: str = "1m") -> int | None:
        normalized_tf = _normalize_timeframe(timeframe)
        pattern = f"{self._instrument_prefix()}*:*:{normalized_tf}"
        newest: int | None = None
        try:
            client = get_sync_redis_client()
            for key in client.scan_iter(match=pattern, count=200):
                raw_value = client.get(str(key))
                if raw_value is None:
                    continue
                try:
                    ts_ms = int(raw_value)
                except (TypeError, ValueError):
                    continue
                if newest is None or ts_ms > newest:
                    newest = ts_ms
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("freshest_checkpoint_ms", exc)
            logger.warning("redis freshest checkpoint scan failed error=%s", type(exc).__name__)
            return None
        return newest

    def record_refresh_scheduler_metrics(
        self,
        *,
        scheduled: int,
        deduped: int,
        total: int,
    ) -> bool:
        now_ms = int(time() * 1000)
        payload = {
            "scheduled": str(max(0, int(scheduled))),
            "deduped": str(max(0, int(deduped))),
            "total": str(max(0, int(total))),
            "duplicate_rate_pct": (
                f"{(float(deduped) / float(total) * 100.0):.2f}" if total > 0 else "0.00"
            ),
            "updated_at_ms": str(now_ms),
        }
        key = self._refresh_metrics_key()
        try:
            client = get_sync_redis_client()
            client.hset(key, mapping=payload)
            client.expire(key, max(300, int(settings.market_data_checkpoint_ttl_seconds)))
            self._clear_error()
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error("record_refresh_scheduler_metrics", exc)
            logger.warning(
                "redis refresh scheduler metrics write failed error=%s",
                type(exc).__name__,
            )
            return False

    def get_refresh_scheduler_metrics(self) -> dict[str, object]:
        key = self._refresh_metrics_key()
        try:
            client = get_sync_redis_client()
            payload = client.hgetall(key)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("get_refresh_scheduler_metrics", exc)
            logger.warning(
                "redis refresh scheduler metrics read failed error=%s",
                type(exc).__name__,
            )
            return {}
        if not payload:
            return {}
        output: dict[str, object] = {}
        for key_name, value in payload.items():
            key_text = str(key_name)
            text = str(value)
            if key_text in {"scheduled", "deduped", "total", "updated_at_ms"}:
                try:
                    output[key_text] = int(text)
                except (TypeError, ValueError):
                    output[key_text] = 0
                continue
            if key_text == "duplicate_rate_pct":
                try:
                    output[key_text] = float(text)
                except (TypeError, ValueError):
                    output[key_text] = 0.0
                continue
            output[key_text] = text
        return output

    def clear_prefix(self) -> None:
        pattern = f"{self.key_prefix}:*"
        try:
            client = get_sync_redis_client()
            keys = [str(item) for item in client.scan_iter(match=pattern, count=500)]
            if keys:
                client.delete(*keys)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("clear_prefix", exc)
            logger.warning("redis market-data clear prefix failed error=%s", type(exc).__name__)


redis_market_data_store = RedisMarketDataStore()
