"""Async Redis connection pool management."""

from __future__ import annotations

import asyncio
from threading import Lock

from redis import ConnectionPool as SyncConnectionPool
from redis import Redis as SyncRedis
from redis.asyncio import ConnectionPool, Redis

from packages.shared_settings.schema.settings import settings
from packages.infra.observability.logger import logger

redis_pool: ConnectionPool | None = None
redis_client: Redis | None = None
_redis_loop: asyncio.AbstractEventLoop | None = None
sync_redis_pool: SyncConnectionPool | None = None
sync_redis_client: SyncRedis | None = None
_sync_redis_lock = Lock()


async def init_redis() -> None:
    """Initialize Redis connection pool and verify connectivity."""
    global redis_pool, redis_client, _redis_loop
    current_loop = asyncio.get_running_loop()

    if redis_client is not None and _redis_loop is not current_loop:
        # Recreate client when the app enters a different event loop
        # (e.g. pytest async fixtures + TestClient mixed usage).
        await close_redis()

    if redis_client is None:
        redis_pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        redis_client = Redis(connection_pool=redis_pool)
        _redis_loop = current_loop

    await redis_client.ping()
    logger.info("Redis pool initialized.")


async def close_redis() -> None:
    """Close Redis client and pool."""
    global redis_pool, redis_client, _redis_loop, sync_redis_pool, sync_redis_client

    if redis_client is not None:
        try:
            await redis_client.aclose()
            logger.info("Redis client closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis client close failed: %s", type(exc).__name__)

    if redis_pool is not None:
        try:
            await redis_pool.disconnect(inuse_connections=True)
            logger.info("Redis pool closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis pool disconnect failed: %s", type(exc).__name__)

    if sync_redis_client is not None:
        try:
            sync_redis_client.close()
            logger.info("Sync Redis client closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sync Redis client close failed: %s", type(exc).__name__)

    if sync_redis_pool is not None:
        try:
            sync_redis_pool.disconnect()
            logger.info("Sync Redis pool closed.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sync Redis pool disconnect failed: %s", type(exc).__name__)

    redis_client = None
    redis_pool = None
    _redis_loop = None
    sync_redis_client = None
    sync_redis_pool = None


def get_redis_client() -> Redis:
    """Return initialized Redis client."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized.")
    return redis_client


def get_sync_redis_client() -> SyncRedis:
    """Return sync Redis client, lazily initialized for sync call paths."""
    global sync_redis_pool, sync_redis_client

    if sync_redis_client is not None:
        return sync_redis_client

    with _sync_redis_lock:
        if sync_redis_client is not None:
            return sync_redis_client
        sync_redis_pool = SyncConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        sync_redis_client = SyncRedis(connection_pool=sync_redis_pool)
    return sync_redis_client


async def redis_healthcheck() -> bool:
    """Return True if Redis responds to PING."""
    if redis_client is None:
        return False

    try:
        return bool(await redis_client.ping())
    except Exception:
        logger.exception("Redis health check failed.")
        return False
