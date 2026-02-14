"""Chat endpoints for new-thread and message exchange."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.orchestrator import ChatOrchestrator
from src.api.middleware.auth import get_current_user
from src.api.schemas.events import ThreadResponse
from src.api.schemas.requests import ChatSendRequest, NewThreadRequest
from src.dependencies import get_db, get_responses_event_streamer
from src.models.user import User
from src.services.openai_stream_service import ResponsesEventStreamer

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
    return ThreadResponse(
        session_id=session.id,
        phase=session.current_phase,
        status=session.status,
        kyc_status=_resolve_kyc_status(user),
    )


@router.post("/send-openai-stream")
async def send_message_stream(
    payload: ChatSendRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    streamer: ResponsesEventStreamer = Depends(get_responses_event_streamer),
    language: str = Query("en", description="ISO 639-1 language code from frontend"),
) -> StreamingResponse:
    """Stream a chat turn via the OpenAI Responses API (SSE)."""
    orchestrator = ChatOrchestrator(db)
    stream = orchestrator.handle_message_stream(user, payload, streamer, language=language)
    return StreamingResponse(stream, media_type="text/event-stream")
