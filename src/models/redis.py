"""Async Redis connection pool management."""

from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis

from src.config import settings
from src.util.logger import logger

redis_pool: ConnectionPool | None = None
redis_client: Redis | None = None


async def init_redis() -> None:
    """Initialize Redis connection pool and verify connectivity."""
    global redis_pool, redis_client

    if redis_client is None:
        redis_pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        redis_client = Redis(connection_pool=redis_pool)

    await redis_client.ping()
    logger.info("Redis pool initialized.")


async def close_redis() -> None:
    """Close Redis client and pool."""
    global redis_pool, redis_client

    if redis_client is not None:
        await redis_client.aclose()
        logger.info("Redis client closed.")

    if redis_pool is not None:
        await redis_pool.disconnect(inuse_connections=True)
        logger.info("Redis pool closed.")

    redis_client = None
    redis_pool = None


def get_redis_client() -> Redis:
    """Return initialized Redis client."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized.")
    return redis_client


async def redis_healthcheck() -> bool:
    """Return True if Redis responds to PING."""
    if redis_client is None:
        return False

    try:
        return bool(await redis_client.ping())
    except Exception:
        logger.exception("Redis health check failed.")
        return False
