"""In-memory cache for pre-strategy regime snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any
from uuid import uuid4

_CACHE_LOCK = RLock()
_CACHE_MAX_ENTRIES = 128


@dataclass(slots=True)
class _CacheEntry:
    payload: dict[str, Any]
    expires_at: datetime
    created_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at


_REGIME_SNAPSHOT_CACHE: dict[str, _CacheEntry] = {}


def _prune_expired_locked() -> None:
    expired_ids = [
        key
        for key, entry in _REGIME_SNAPSHOT_CACHE.items()
        if entry.is_expired
    ]
    for key in expired_ids:
        _REGIME_SNAPSHOT_CACHE.pop(key, None)


def put_snapshot(
    payload: dict[str, Any],
    *,
    ttl_seconds: int,
) -> str:
    ttl = max(int(ttl_seconds), 60)
    now = datetime.now(UTC)
    snapshot_id = str(uuid4())
    entry = _CacheEntry(
        payload=payload,
        expires_at=now + timedelta(seconds=ttl),
        created_at=now,
    )
    with _CACHE_LOCK:
        _prune_expired_locked()
        if len(_REGIME_SNAPSHOT_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest_key = min(
                _REGIME_SNAPSHOT_CACHE,
                key=lambda key: _REGIME_SNAPSHOT_CACHE[key].created_at,
            )
            _REGIME_SNAPSHOT_CACHE.pop(oldest_key, None)
        _REGIME_SNAPSHOT_CACHE[snapshot_id] = entry
    return snapshot_id


def get_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    normalized = str(snapshot_id).strip()
    if not normalized:
        return None
    with _CACHE_LOCK:
        _prune_expired_locked()
        entry = _REGIME_SNAPSHOT_CACHE.get(normalized)
        if entry is None:
            return None
        if entry.is_expired:
            _REGIME_SNAPSHOT_CACHE.pop(normalized, None)
            return None
        return entry.payload

