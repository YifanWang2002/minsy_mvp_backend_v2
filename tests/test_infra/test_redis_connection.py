import pytest
from redis.asyncio import Redis

from src.config import settings


@pytest.mark.asyncio
async def test_redis_connection_ping() -> None:
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        pong = await client.ping()
    finally:
        await client.aclose()

    assert pong is True
