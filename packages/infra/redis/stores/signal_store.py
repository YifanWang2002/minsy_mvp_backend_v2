"""Signal store for deployment runtime events (Redis + in-memory fallback)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
from dataclasses import dataclass
from datetime import datetime
import json
import os
from threading import RLock
from typing import Any
from uuid import UUID

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_redis_client, init_redis

_DEFAULT_REDIS_PREFIX = "paper_trading:signals"
_DEFAULT_REDIS_INDEX = "paper_trading:signals:index"
_ALLOW_IN_MEMORY_FALLBACK = (
    os.getenv("ALLOW_IN_MEMORY_SIGNAL_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
)


@dataclass(frozen=True, slots=True)
class SignalRecord:
    deployment_id: UUID
    signal: str
    symbol: str
    timeframe: str
    bar_time: datetime
    reason: str
    metadata: dict[str, Any]
    signal_event_id: UUID | None = None


class SignalStore:
    """Bounded signal history keyed by deployment id."""

    def __init__(
        self,
        *,
        max_per_deployment: int = 500,
        redis_prefix: str = _DEFAULT_REDIS_PREFIX,
        redis_index_key: str = _DEFAULT_REDIS_INDEX,
    ) -> None:
        self.max_per_deployment = max_per_deployment
        self.redis_prefix = redis_prefix
        self.redis_index_key = redis_index_key
        self._records: dict[UUID, list[SignalRecord]] = defaultdict(list)
        self._lock = RLock()
        self._fallback_hits = 0
        self._last_fallback_error: str | None = None

    @staticmethod
    async def _get_ready_redis_client():
        try:
            redis = get_redis_client()
        except RuntimeError:
            await init_redis()
            return get_redis_client()
        try:
            await redis.ping()
            return redis
        except RuntimeError:
            await init_redis()
            return get_redis_client()

    def _redis_key(self, deployment_id: UUID) -> str:
        return f"{self.redis_prefix}:{deployment_id}"

    @staticmethod
    def _serialize(record: SignalRecord) -> str:
        payload = {
            "signal_event_id": str(record.signal_event_id) if record.signal_event_id is not None else None,
            "deployment_id": str(record.deployment_id),
            "signal": record.signal,
            "symbol": record.symbol,
            "timeframe": record.timeframe,
            "bar_time": record.bar_time.isoformat(),
            "reason": record.reason,
            "metadata": record.metadata if isinstance(record.metadata, dict) else {},
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _deserialize(raw: str) -> SignalRecord | None:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return None
            return SignalRecord(
                signal_event_id=(
                    UUID(str(payload["signal_event_id"]))
                    if payload.get("signal_event_id") is not None
                    else None
                ),
                deployment_id=UUID(str(payload["deployment_id"])),
                signal=str(payload.get("signal", "")),
                symbol=str(payload.get("symbol", "")),
                timeframe=str(payload.get("timeframe", "")),
                bar_time=datetime.fromisoformat(str(payload.get("bar_time"))),
                reason=str(payload.get("reason", "")),
                metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
            )
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _use_in_memory_fallback() -> bool:
        if _ALLOW_IN_MEMORY_FALLBACK:
            return True
        return "PYTEST_CURRENT_TEST" in os.environ

    def fallback_status(self) -> dict[str, Any]:
        with self._lock:
            in_memory_records = sum(len(rows) for rows in self._records.values())
            fallback_hits = int(self._fallback_hits)
            last_error = self._last_fallback_error
        return {
            "enabled": bool(self._use_in_memory_fallback()),
            "active": bool(fallback_hits > 0),
            "fallback_hits": fallback_hits,
            "entries": in_memory_records,
            "last_error": last_error,
            "mode": "memory",
        }

    def add(self, record: SignalRecord) -> None:
        with self._lock:
            bucket = self._records[record.deployment_id]
            bucket.append(record)
            if len(bucket) > self.max_per_deployment:
                del bucket[: len(bucket) - self.max_per_deployment]

    async def add_async(self, record: SignalRecord) -> None:
        self.add(record)
        try:
            redis = await self._get_ready_redis_client()
            key = self._redis_key(record.deployment_id)
            encoded = self._serialize(record)
            pipeline = redis.pipeline(transaction=True)
            pipeline.sadd(self.redis_index_key, str(record.deployment_id))
            pipeline.lpush(key, encoded)
            pipeline.ltrim(key, 0, self.max_per_deployment - 1)
            await pipeline.execute()
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("signal_store add failed (redis unavailable)")
                raise RuntimeError("Signal storage unavailable.") from exc
            with self._lock:
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning("signal_store add fallback to memory: %s", type(exc).__name__)

    def list_recent(
        self,
        deployment_id: UUID,
        *,
        limit: int = 100,
        cursor: UUID | None = None,
    ) -> list[SignalRecord]:
        with self._lock:
            rows = self._records.get(deployment_id, [])
            if cursor is not None:
                filtered = [row for row in rows if row.signal_event_id is None or row.signal_event_id != cursor]
                return list(filtered[-limit:])
            return list(rows[-limit:])

    async def list_recent_async(
        self,
        deployment_id: UUID,
        *,
        limit: int = 100,
        cursor: UUID | None = None,
    ) -> list[SignalRecord]:
        try:
            redis = await self._get_ready_redis_client()
            key = self._redis_key(deployment_id)
            raw_rows = await redis.lrange(key, 0, max(0, int(limit) - 1))
            if not isinstance(raw_rows, list):
                return self.list_recent(deployment_id, limit=limit, cursor=cursor)
            parsed: list[SignalRecord] = []
            for raw in raw_rows:
                item = self._deserialize(str(raw))
                if item is not None:
                    parsed.append(item)
            if parsed:
                parsed.reverse()
                if cursor is None:
                    return parsed
                return [item for item in parsed if item.signal_event_id is None or item.signal_event_id != cursor]
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("signal_store read failed (redis unavailable)")
                raise RuntimeError("Signal storage unavailable.") from exc
            with self._lock:
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning("signal_store read fallback to memory: %s", type(exc).__name__)
        return self.list_recent(deployment_id, limit=limit, cursor=cursor)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._fallback_hits = 0
            self._last_fallback_error = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Best-effort cleanup when called from sync tests/scripts.
            with contextlib.suppress(Exception):
                asyncio.run(self.clear_async())

    async def clear_async(self) -> None:
        with self._lock:
            self._records.clear()
        try:
            redis = await self._get_ready_redis_client()
            deployment_ids = await redis.smembers(self.redis_index_key)
            keys: list[str] = []
            for item in deployment_ids or []:
                try:
                    keys.append(self._redis_key(UUID(str(item))))
                except Exception:  # noqa: BLE001
                    continue
            if keys:
                await redis.delete(*keys)
            await redis.delete(self.redis_index_key)
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("signal_store clear failed (redis unavailable)")
                raise RuntimeError("Signal storage unavailable.") from exc
            logger.warning("signal_store clear fallback to memory: %s", type(exc).__name__)


signal_store = SignalStore()
