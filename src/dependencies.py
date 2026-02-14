"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_db_session
from src.models.redis import get_redis_client
from src.services.openai_stream_service import (
    OpenAIResponsesEventStreamer,
    ResponsesEventStreamer,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    async for session in get_db_session():
        yield session


def get_redis() -> Redis:
    return get_redis_client()


def get_responses_event_streamer() -> ResponsesEventStreamer:
    return OpenAIResponsesEventStreamer()
