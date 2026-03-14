"""FastAPI dependency providers."""

from __future__ import annotations

import hmac
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.orchestration.openai_stream_service import (
    OpenAIResponsesEventStreamer,
    ResponsesEventStreamer,
)
from packages.domain.market_data.incremental.hmac_auth import (
    build_signature_payload,
    sign_payload,
)
from packages.infra.db.session import get_db_session
from packages.infra.redis.client import get_redis_client
from packages.shared_settings.schema.settings import settings


async def get_db() -> AsyncIterator[AsyncSession]:
    async for session in get_db_session():
        yield session


def get_redis() -> Redis:
    return get_redis_client()


def get_responses_event_streamer() -> ResponsesEventStreamer:
    return OpenAIResponsesEventStreamer()


async def require_market_data_incremental_service_auth(
    request: Request,
    x_minsy_service_key_id: Annotated[
        str | None,
        Header(alias="X-Minsy-Service-Key-Id"),
    ] = None,
    x_minsy_service_timestamp: Annotated[
        str | None,
        Header(alias="X-Minsy-Service-Timestamp"),
    ] = None,
    x_minsy_service_signature: Annotated[
        str | None,
        Header(alias="X-Minsy-Service-Signature"),
    ] = None,
) -> dict[str, str]:
    if not settings.market_data_incremental_receiver_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

    expected_key_id = settings.market_data_incremental_hmac_key_id.strip()
    expected_secret = settings.market_data_incremental_hmac_secret.strip()
    if not expected_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Incremental service auth is not configured.",
        )
    if x_minsy_service_key_id is None or x_minsy_service_key_id.strip() != expected_key_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service key id.",
        )
    if x_minsy_service_timestamp is None or x_minsy_service_signature is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing service signature headers.",
        )

    try:
        timestamp_epoch = int(x_minsy_service_timestamp.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service timestamp.",
        ) from exc

    now_epoch = int(time.time())
    max_skew = int(settings.market_data_incremental_hmac_max_skew_seconds)
    if abs(now_epoch - timestamp_epoch) > max_skew:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service timestamp is outside allowed clock skew.",
        )

    body = await request.body()
    payload = build_signature_payload(
        timestamp_epoch_seconds=timestamp_epoch,
        method=request.method,
        path=request.url.path,
        body=body,
    )
    expected_signature = sign_payload(secret=expected_secret, payload=payload)
    if not hmac.compare_digest(expected_signature, x_minsy_service_signature.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service signature.",
        )
    return {"key_id": expected_key_id}
