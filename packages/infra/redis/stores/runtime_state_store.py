"""External runtime state store for deployment-level execution metadata."""

from __future__ import annotations

import json
import os
from threading import RLock
from typing import Any
from uuid import UUID

from packages.infra.observability.logger import logger
from packages.infra.redis.client import get_redis_client, init_redis

_DEFAULT_PREFIX = "paper_trading:runtime_state"
_DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7
_ALLOW_IN_MEMORY_FALLBACK = (
    os.getenv("ALLOW_IN_MEMORY_RUNTIME_STATE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
)


class RuntimeStateStore:
    """Persist/retrieve runtime state snapshots outside process memory."""

    def __init__(self, *, key_prefix: str = _DEFAULT_PREFIX, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self.key_prefix = key_prefix
        self.ttl_seconds = max(int(ttl_seconds), 60)
        self._fallback: dict[str, dict[str, Any]] = {}
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

    @staticmethod
    def _use_in_memory_fallback() -> bool:
        if _ALLOW_IN_MEMORY_FALLBACK:
            return True
        return "PYTEST_CURRENT_TEST" in os.environ

    def fallback_status(self) -> dict[str, Any]:
        with self._lock:
            entries = len(self._fallback)
            fallback_hits = int(self._fallback_hits)
            last_error = self._last_fallback_error
        return {
            "enabled": bool(self._use_in_memory_fallback()),
            "active": bool(fallback_hits > 0),
            "fallback_hits": fallback_hits,
            "entries": entries,
            "last_error": last_error,
            "mode": "memory",
        }

    def _key(self, deployment_id: UUID) -> str:
        return f"{self.key_prefix}:{deployment_id}"

    def _health_key(self) -> str:
        return f"{self.key_prefix}:__live_trading_health__"

    async def upsert(
        self,
        deployment_id: UUID,
        state: dict[str, Any],
        *,
        merge: bool = True,
    ) -> None:
        normalized = dict(state) if isinstance(state, dict) else {}
        key = self._key(deployment_id)
        try:
            redis = await self._get_ready_redis_client()
            if merge:
                raw_existing = await redis.get(key)
                if raw_existing:
                    try:
                        existing_payload = json.loads(str(raw_existing))
                        if isinstance(existing_payload, dict):
                            normalized = {
                                **existing_payload,
                                **normalized,
                            }
                    except Exception:  # noqa: BLE001
                        logger.warning("runtime_state_store merge skipped invalid payload.")
            encoded = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
            await redis.set(key, encoded, ex=self.ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("runtime_state_store upsert failed (redis unavailable)")
                raise RuntimeError("Runtime state storage unavailable.") from exc
            with self._lock:
                if merge:
                    existing = self._fallback.get(key)
                    if isinstance(existing, dict):
                        self._fallback[key] = {**existing, **normalized}
                    else:
                        self._fallback[key] = normalized
                else:
                    self._fallback[key] = normalized
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning("runtime_state_store fallback to memory: %s", type(exc).__name__)

    async def get(self, deployment_id: UUID) -> dict[str, Any] | None:
        key = self._key(deployment_id)
        try:
            redis = await self._get_ready_redis_client()
            raw = await redis.get(key)
            if not raw:
                return None
            parsed = json.loads(raw if isinstance(raw, str) else str(raw))
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("runtime_state_store read failed (redis unavailable)")
                raise RuntimeError("Runtime state storage unavailable.") from exc
            with self._lock:
                value = self._fallback.get(key)
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning("runtime_state_store read fallback to memory: %s", type(exc).__name__)
            return dict(value) if isinstance(value, dict) else None

    async def publish_live_trading_health(self, payload: dict[str, Any]) -> None:
        normalized = dict(payload) if isinstance(payload, dict) else {}
        key = self._health_key()
        try:
            redis = await self._get_ready_redis_client()
            encoded = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
            await redis.set(key, encoded, ex=self.ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.warning(
                    "runtime_state_store live-trading health publish skipped: %s",
                    type(exc).__name__,
                )
                return
            with self._lock:
                self._fallback[key] = normalized
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning(
                "runtime_state_store live-trading health fallback to memory: %s",
                type(exc).__name__,
            )

    async def get_live_trading_health(self) -> dict[str, Any] | None:
        key = self._health_key()
        try:
            redis = await self._get_ready_redis_client()
            raw = await redis.get(key)
            if not raw:
                return None
            parsed = json.loads(raw if isinstance(raw, str) else str(raw))
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.warning(
                    "runtime_state_store live-trading health read skipped: %s",
                    type(exc).__name__,
                )
                return None
            with self._lock:
                value = self._fallback.get(key)
                self._fallback_hits += 1
                self._last_fallback_error = type(exc).__name__
            logger.warning(
                "runtime_state_store live-trading health read fallback to memory: %s",
                type(exc).__name__,
            )
            return dict(value) if isinstance(value, dict) else None

    async def clear(self) -> None:
        pattern = f"{self.key_prefix}:*"
        with self._lock:
            self._fallback.clear()
            self._fallback_hits = 0
            self._last_fallback_error = None
        try:
            redis = await self._get_ready_redis_client()
            cursor = 0
            keys_to_delete: list[str] = []
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    keys_to_delete.extend(str(item) for item in keys)
                if cursor == 0:
                    break
            if keys_to_delete:
                await redis.delete(*keys_to_delete)
        except Exception as exc:  # noqa: BLE001
            if not self._use_in_memory_fallback():
                logger.exception("runtime_state_store clear failed (redis unavailable)")
                raise RuntimeError("Runtime state storage unavailable.") from exc
            logger.warning("runtime_state_store clear fallback to memory: %s", type(exc).__name__)


runtime_state_store = RuntimeStateStore()
