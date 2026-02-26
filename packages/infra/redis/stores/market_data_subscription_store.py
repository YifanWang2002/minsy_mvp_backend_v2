"""Redis-backed subscription index for cross-process visibility."""

from __future__ import annotations

from threading import RLock
from time import time

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_sync_redis_client

_SUPPORTED_MARKETS = {"stocks", "crypto", "forex", "futures", "commodities"}


def _normalize_market(market: str) -> str:
    normalized = market.strip().lower()
    if normalized not in _SUPPORTED_MARKETS:
        return "stocks"
    return normalized


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _instrument_token(market: str, symbol: str) -> str:
    return f"{_normalize_market(market)}|{_normalize_symbol(symbol)}"


def _split_instrument_token(token: str) -> tuple[str, str]:
    market, symbol = token.split("|", 1)
    return _normalize_market(market), _normalize_symbol(symbol)


class RedisSubscriptionStore:
    """Maintains subscriber<->instrument sets in Redis."""

    def __init__(self, *, key_prefix: str = "md:v1") -> None:
        self.key_prefix = key_prefix.strip() or "md:v1"
        self._last_error: dict[str, object] | None = None
        self._error_lock = RLock()

    def _subscriber_key(self, subscriber_id: str) -> str:
        return f"{self.key_prefix}:sub:subscriber:{subscriber_id}"

    def _instrument_key(self, market: str, symbol: str) -> str:
        normalized_market = _normalize_market(market)
        normalized_symbol = _normalize_symbol(symbol)
        return f"{self.key_prefix}:sub:instrument:{normalized_market}:{normalized_symbol}"

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
            logger.warning("redis subscription ping failed error=%s", type(exc).__name__)
            return False

    def set_subscriber_instruments(
        self,
        subscriber_id: str,
        instruments: tuple[tuple[str, str], ...],
    ) -> bool:
        subscriber_key = self._subscriber_key(subscriber_id)
        normalized = {
            _instrument_token(market, symbol)
            for market, symbol in instruments
            if market.strip() and symbol.strip()
        }
        try:
            client = get_sync_redis_client()
            existing = {str(item) for item in client.smembers(subscriber_key)}
            to_remove = existing - normalized
            to_add = normalized - existing

            pipeline = client.pipeline(transaction=True)
            pipeline.delete(subscriber_key)
            if normalized:
                pipeline.sadd(subscriber_key, *sorted(normalized))
            for token in sorted(to_remove):
                market, symbol = _split_instrument_token(token)
                pipeline.srem(self._instrument_key(market, symbol), subscriber_id)
            for token in sorted(to_add):
                market, symbol = _split_instrument_token(token)
                pipeline.sadd(self._instrument_key(market, symbol), subscriber_id)
            pipeline.execute()

            for token in to_remove:
                market, symbol = _split_instrument_token(token)
                key = self._instrument_key(market, symbol)
                if client.scard(key) <= 0:
                    client.delete(key)
            self._clear_error()
            return True
        except Exception as exc:  # noqa: BLE001
            self._record_error("set_subscriber_instruments", exc)
            logger.warning(
                "redis subscription write failed subscriber=%s error=%s",
                subscriber_id,
                type(exc).__name__,
            )
            return False

    def clear_subscriber(self, subscriber_id: str) -> bool:
        return self.set_subscriber_instruments(subscriber_id, ())

    def subscriber_instruments(self, subscriber_id: str) -> tuple[tuple[str, str], ...]:
        key = self._subscriber_key(subscriber_id)
        try:
            client = get_sync_redis_client()
            payload = client.smembers(key)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("subscriber_instruments", exc)
            logger.warning(
                "redis subscription read failed subscriber=%s error=%s",
                subscriber_id,
                type(exc).__name__,
            )
            return ()
        rows: list[tuple[str, str]] = []
        for item in payload:
            token = str(item)
            if "|" not in token:
                continue
            rows.append(_split_instrument_token(token))
        return tuple(sorted(set(rows)))

    def subscribers_for_instrument(self, *, market: str, symbol: str) -> tuple[str, ...]:
        key = self._instrument_key(market, symbol)
        try:
            client = get_sync_redis_client()
            payload = client.smembers(key)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("subscribers_for_instrument", exc)
            logger.warning(
                "redis instrument subscriber read failed market=%s symbol=%s error=%s",
                market,
                symbol,
                type(exc).__name__,
            )
            return ()
        return tuple(sorted(str(item) for item in payload))

    def active_instruments(self) -> tuple[tuple[str, str], ...]:
        pattern = f"{self.key_prefix}:sub:instrument:*"
        rows: list[tuple[str, str]] = []
        try:
            client = get_sync_redis_client()
            for key in client.scan_iter(match=pattern, count=200):
                key_text = str(key)
                try:
                    head, market, symbol = key_text.rsplit(":", 2)
                except ValueError:
                    continue
                if not head.endswith(":sub:instrument"):
                    continue
                if client.scard(key_text) <= 0:
                    continue
                rows.append((_normalize_market(market), _normalize_symbol(symbol)))
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("active_instruments", exc)
            logger.warning("redis active subscription scan failed error=%s", type(exc).__name__)
            return ()
        return tuple(sorted(set(rows)))

    def clear_prefix(self) -> None:
        pattern = f"{self.key_prefix}:sub:*"
        try:
            client = get_sync_redis_client()
            keys = [str(item) for item in client.scan_iter(match=pattern, count=500)]
            if keys:
                client.delete(*keys)
            self._clear_error()
        except Exception as exc:  # noqa: BLE001
            self._record_error("clear_prefix", exc)
            logger.warning("redis subscription clear prefix failed error=%s", type(exc).__name__)


redis_subscription_store = RedisSubscriptionStore()
