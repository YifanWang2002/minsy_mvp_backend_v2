"""Redis-backed deployment runtime lock with in-process fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from uuid import UUID, uuid4

from src.config import settings
from src.models.redis import get_redis_client, init_redis
from src.util.logger import logger

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

_REFRESH_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""


@dataclass(frozen=True, slots=True)
class DeploymentLockLease:
    key: str
    token: str
    via_fallback: bool = False


class DeploymentRuntimeLock:
    """Per-deployment mutual exclusion lock for runtime/signal processing."""

    def __init__(self, *, prefix: str = "paper_trading:deployment_lock") -> None:
        self._prefix = prefix
        self._fallback_guard = asyncio.Lock()
        self._fallback_tokens: dict[str, tuple[str, float]] = {}

    def _key(self, deployment_id: UUID) -> str:
        return f"{self._prefix}:{deployment_id}"

    @staticmethod
    def _ttl_ms() -> int:
        return max(1000, int(settings.paper_trading_deployment_lock_ttl_seconds * 1000))

    async def _get_ready_redis(self):
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

    async def acquire(self, deployment_id: UUID) -> DeploymentLockLease | None:
        key = self._key(deployment_id)
        token = uuid4().hex
        ttl_ms = self._ttl_ms()
        try:
            redis = await self._get_ready_redis()
            acquired = await redis.set(key, token, nx=True, px=ttl_ms)
            if acquired:
                return DeploymentLockLease(key=key, token=token, via_fallback=False)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "deployment lock redis unavailable, using in-process fallback: %s",
                type(exc).__name__,
            )
            async with self._fallback_guard:
                existing = self._fallback_tokens.get(key)
                now = time.monotonic()
                if existing is not None and existing[1] > now:
                    return None
                self._fallback_tokens[key] = (token, now + ttl_ms / 1000)
            return DeploymentLockLease(key=key, token=token, via_fallback=True)

    async def refresh(self, lease: DeploymentLockLease) -> bool:
        ttl_ms = self._ttl_ms()
        if lease.via_fallback:
            async with self._fallback_guard:
                current = self._fallback_tokens.get(lease.key)
                now = time.monotonic()
                if current is None or current[0] != lease.token:
                    return False
                self._fallback_tokens[lease.key] = (lease.token, now + ttl_ms / 1000)
            return True

        try:
            redis = await self._get_ready_redis()
            refreshed = await redis.eval(_REFRESH_SCRIPT, 1, lease.key, lease.token, ttl_ms)
            return bool(refreshed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("deployment lock refresh failed: %s", type(exc).__name__)
            return False

    async def release(self, lease: DeploymentLockLease) -> bool:
        if lease.via_fallback:
            async with self._fallback_guard:
                current = self._fallback_tokens.get(lease.key)
                if current is None or current[0] != lease.token:
                    return False
                self._fallback_tokens.pop(lease.key, None)
            return True

        try:
            redis = await self._get_ready_redis()
            released = await redis.eval(_RELEASE_SCRIPT, 1, lease.key, lease.token)
            return bool(released)
        except Exception as exc:  # noqa: BLE001
            logger.warning("deployment lock release failed: %s", type(exc).__name__)
            return False


deployment_runtime_lock = DeploymentRuntimeLock()
