"""Redis sliding-window rate limiter dependency."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from apps.api.middleware.auth import get_current_user
from apps.api.dependencies import get_redis
from packages.infra.db.models.user import User


class RateLimiter:
    """Rate limiter using Redis sorted set sliding window."""

    def __init__(self, limit: int = 30, window: int = 60) -> None:
        self.limit = limit
        self.window = window

    async def __call__(
        self,
        request: Request,
        user: User = Depends(get_current_user),
        redis: Redis = Depends(get_redis),
    ) -> None:
        now = time.time()
        window_start = now - self.window
        key = f"rate:{user.id}:{self.window}"
        member = f"{request.method}:{request.url.path}:{now}:{uuid4()}"

        pipeline = redis.pipeline(transaction=True)
        pipeline.zremrangebyscore(key, "-inf", window_start)
        pipeline.zadd(key, {member: now})
        pipeline.zcard(key)
        pipeline.expire(key, self.window + 1)
        _, _, count, _ = await pipeline.execute()

        if int(count) > self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
            )
