"""Shared dedupe guard for market-data refresh scheduling."""

from __future__ import annotations

from threading import Lock
from time import monotonic, time

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_sync_redis_client
from packages.shared_settings.schema.settings import settings

_REFRESH_DEDUPE_FALLBACK: dict[str, float] = {}
_REFRESH_DEDUPE_LOCK = Lock()


def _refresh_dedupe_key(market: str, symbol: str) -> str:
    normalized_market = str(market).strip().lower()
    normalized_symbol = str(symbol).strip().upper()
    return f"md:v1:refresh_dedupe:{normalized_market}:{normalized_symbol}"


def refresh_dedupe_ttl_seconds() -> int:
    interval = max(
        1,
        int(settings.market_data_refresh_active_subscriptions_interval_seconds),
    )
    window = max(1, int(settings.market_data_refresh_dedupe_window_seconds))
    return max(interval, window)


def reserve_market_data_refresh_slot(market: str, symbol: str) -> bool:
    if not settings.market_data_refresh_dedupe_enabled:
        return True

    key = _refresh_dedupe_key(market, symbol)
    ttl = refresh_dedupe_ttl_seconds()
    try:
        client = get_sync_redis_client()
        reserved = client.set(key, str(int(time() * 1000)), nx=True, ex=ttl)
        return bool(reserved)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[market-data] refresh dedupe redis unavailable, using local fallback error=%s",
            type(exc).__name__,
        )

    now = monotonic()
    with _REFRESH_DEDUPE_LOCK:
        expired_keys = [
            item
            for item, expire_at in _REFRESH_DEDUPE_FALLBACK.items()
            if expire_at <= now
        ]
        for item in expired_keys:
            _REFRESH_DEDUPE_FALLBACK.pop(item, None)
        current_expire_at = _REFRESH_DEDUPE_FALLBACK.get(key)
        if current_expire_at is not None and current_expire_at > now:
            return False
        _REFRESH_DEDUPE_FALLBACK[key] = now + float(ttl)
    return True
