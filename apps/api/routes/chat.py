"""Chat endpoints for new-thread and message exchange."""

from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.orchestration import ChatOrchestrator
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import ThreadResponse
from apps.api.schemas.requests import ChatSendRequest, NewThreadRequest
from packages.shared_settings.schema.settings import settings
from apps.api.dependencies import get_db, get_responses_event_streamer
from packages.infra.db.models.user import User
from apps.api.orchestration.openai_stream_service import ResponsesEventStreamer
from packages.domain.session.services.session_title_service import read_session_title_from_metadata
from packages.infra.observability.logger import logger
from apps.api.orchestration.chat_debug_trace import (
    CHAT_TRACE_HEADER_ENABLED,
    CHAT_TRACE_HEADER_ID,
    CHAT_TRACE_HEADER_MODE,
    CHAT_TRACE_RESPONSE_HEADER_ID,
    build_chat_debug_trace,
    reset_chat_debug_trace,
    set_chat_debug_trace,
)

router = APIRouter(prefix="/chat", tags=["chat"])
_CHAT_SSE_HEARTBEAT_SECONDS = 12.0


def _resolve_kyc_status(user: User) -> str:
    if user.profiles:
        return user.profiles[0].kyc_status
    return "incomplete"


@router.post("/new-thread", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def new_thread(
    payload: NewThreadRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ThreadResponse:
    orchestrator = ChatOrchestrator(db)
    session = await orchestrator.create_session(
        user_id=user.id,
        parent_session_id=payload.parent_session_id,
        metadata=payload.metadata,
    )
    await db.commit()
    await db.refresh(session)
    title_payload = read_session_title_from_metadata(dict(session.metadata_ or {}))
    return ThreadResponse(
        session_id=session.id,
        phase=session.current_phase,
        status=session.status,
        kyc_status=_resolve_kyc_status(user),
        session_title=title_payload.title,
        session_title_record=title_payload.record,
    )


@router.post("/send-openai-stream")
async def send_message_stream(
    payload: ChatSendRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    streamer: ResponsesEventStreamer = Depends(get_responses_event_streamer),
    language: str = Query("en", description="ISO 639-1 language code from frontend"),
) -> StreamingResponse:
    """Stream a chat turn via the OpenAI Responses API (SSE)."""
    trace = build_chat_debug_trace(
        default_enabled=settings.chat_debug_trace_enabled,
        default_mode=settings.chat_debug_trace_mode,
        header_value=request.headers.get(CHAT_TRACE_HEADER_ENABLED),
        requested_trace_id=request.headers.get(CHAT_TRACE_HEADER_ID),
        requested_mode=request.headers.get(CHAT_TRACE_HEADER_MODE),
    )

    orchestrator = ChatOrchestrator(db)
    stream = orchestrator.handle_message_stream(user, payload, streamer, language=language)

    async def traced_stream():
        token = set_chat_debug_trace(trace)
        producer_queue: asyncio.Queue[str] = asyncio.Queue()
        producer_finished = asyncio.Event()
        producer_error: Exception | None = None
        client_disconnected = False

        async def _produce_chunks() -> None:
            nonlocal producer_error
            try:
                async for chunk in stream:
                    await producer_queue.put(chunk)
            except Exception as exc:  # noqa: BLE001
                producer_error = exc
            finally:
                producer_finished.set()

        producer_task = asyncio.create_task(_produce_chunks())
        try:
            if trace.enabled:
                client_host = request.client.host if request.client is not None else None
                trace.record(
                    stage="frontend_to_backend_request",
                    payload={
                        "path": str(request.url.path),
                        "method": request.method,
                        "client_host": client_host,
                        "language": language,
                        "query_params": dict(request.query_params),
                        "user_id": str(user.id),
                        "payload": payload.model_dump(mode="json"),
                    },
                )
            while True:
                if producer_finished.is_set() and producer_queue.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(
                        producer_queue.get(),
                        timeout=_CHAT_SSE_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if producer_finished.is_set() and producer_queue.empty():
                        break
                    # SSE comment heartbeat (ignored by client parser, but keeps TCP active).
                    yield ": keepalive\n\n"
                    continue
                yield chunk
            if producer_error is not None:
                raise producer_error
        except asyncio.CancelledError:
            # Client disconnected: keep consuming upstream stream so this turn can
            # still reach `done` and persist full assistant output.
            client_disconnected = True
            current_task = asyncio.current_task()
            if (
                current_task is not None
                and hasattr(current_task, "cancelling")
                and hasattr(current_task, "uncancel")
            ):
                while current_task.cancelling():
                    current_task.uncancel()

            while not producer_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(producer_task),
                        timeout=_CHAT_SSE_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    if (
                        current_task is not None
                        and hasattr(current_task, "cancelling")
                        and hasattr(current_task, "uncancel")
                    ):
                        while current_task.cancelling():
                            current_task.uncancel()
                    continue
                except Exception as exc:  # noqa: BLE001
                    producer_error = exc
                    break

            if producer_error is not None:
                logger.warning(
                    "chat.stream producer failed after disconnect user_id=%s error_class=%s error=%s",
                    str(user.id),
                    type(producer_error).__name__,
                    str(producer_error),
                )
            return
        finally:
            if not producer_task.done() and not client_disconnected:
                producer_task.cancel()
                with suppress(Exception):
                    await producer_task
            if trace.enabled:
                trace.record(
                    stage="trace_finished",
                    payload={"path": str(request.url.path), "user_id": str(user.id)},
                )
            reset_chat_debug_trace(token)

    response = StreamingResponse(traced_stream(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache, no-transform"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    if trace.enabled and trace.trace_id:
        response.headers[CHAT_TRACE_RESPONSE_HEADER_ID] = trace.trace_id
    return response
