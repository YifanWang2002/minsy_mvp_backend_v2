"""Chat endpoints for new-thread and message exchange."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import ChatOrchestrator
from src.api.middleware.auth import get_current_user
from src.api.schemas.events import ThreadResponse
from src.api.schemas.requests import ChatSendRequest, NewThreadRequest
from src.config import settings
from src.dependencies import get_db, get_responses_event_streamer
from src.models.user import User
from src.services.openai_stream_service import ResponsesEventStreamer
from src.services.session_title_service import read_session_title_from_metadata
from src.util.chat_debug_trace import (
    CHAT_TRACE_HEADER_ENABLED,
    CHAT_TRACE_HEADER_ID,
    CHAT_TRACE_RESPONSE_HEADER_ID,
    build_chat_debug_trace,
    reset_chat_debug_trace,
    set_chat_debug_trace,
)

router = APIRouter(prefix="/chat", tags=["chat"])


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
        header_value=request.headers.get(CHAT_TRACE_HEADER_ENABLED),
        requested_trace_id=request.headers.get(CHAT_TRACE_HEADER_ID),
    )

    orchestrator = ChatOrchestrator(db)
    stream = orchestrator.handle_message_stream(user, payload, streamer, language=language)

    async def traced_stream():
        token = set_chat_debug_trace(trace)
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
            async for chunk in stream:
                yield chunk
        finally:
            if trace.enabled:
                trace.record(
                    stage="trace_finished",
                    payload={"path": str(request.url.path), "user_id": str(user.id)},
                )
            reset_chat_debug_trace(token)

    response = StreamingResponse(traced_stream(), media_type="text/event-stream")
    if trace.enabled and trace.trace_id:
        response.headers[CHAT_TRACE_RESPONSE_HEADER_ID] = trace.trace_id
    return response
