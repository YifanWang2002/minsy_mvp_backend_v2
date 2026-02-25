"""Session query endpoints for current authenticated user."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.middleware.auth import get_current_user
from src.api.schemas.events import MessageItem, SessionDetailResponse, SessionListItem
from src.dependencies import get_db
from src.models.session import Session
from src.models.user import User
from src.services.session_title_service import read_session_title_from_metadata

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionListItem])
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    archived: bool = Query(default=False, description="When true, list archived sessions."),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SessionListItem]:
    filters = [Session.user_id == user.id]
    if archived:
        filters.append(Session.archived_at.is_not(None))
    else:
        filters.append(Session.archived_at.is_(None))

    stmt = select(Session).where(*filters).order_by(Session.updated_at.desc()).limit(limit)
    sessions = (await db.scalars(stmt)).all()
    output: list[SessionListItem] = []
    for session in sessions:
        title_payload = read_session_title_from_metadata(dict(session.metadata_ or {}))
        output.append(
            SessionListItem(
                session_id=session.id,
                current_phase=session.current_phase,
                status=session.status,
                updated_at=session.updated_at,
                archived_at=session.archived_at,
                session_title=title_payload.title,
                session_title_record=title_payload.record,
            )
        )
    return output


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionDetailResponse:
    stmt = (
        select(Session)
        .where(Session.id == session_id, Session.user_id == user.id)
        .options(selectinload(Session.messages))
    )
    session = await db.scalar(stmt)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    ordered_messages = sorted(session.messages, key=lambda item: item.created_at)
    title_payload = read_session_title_from_metadata(dict(session.metadata_ or {}))
    return SessionDetailResponse(
        session_id=session.id,
        current_phase=session.current_phase,
        status=session.status,
        archived_at=session.archived_at,
        session_title=title_payload.title,
        session_title_record=title_payload.record,
        artifacts=dict(session.artifacts or {}),
        metadata=dict(session.metadata_ or {}),
        last_activity_at=session.last_activity_at,
        messages=[
            MessageItem(
                id=message.id,
                role=message.role,
                content=message.content,
                phase=message.phase,
                created_at=message.created_at,
                tool_calls=message.tool_calls,
                token_usage=message.token_usage,
            )
            for message in ordered_messages
        ],
    )


@router.post("/{session_id}/archive", status_code=status.HTTP_204_NO_CONTENT)
async def archive_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    session = await db.scalar(
        select(Session).where(Session.id == session_id, Session.user_id == user.id),
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if session.archived_at is None:
        session.archived_at = datetime.now(UTC)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/unarchive", status_code=status.HTTP_204_NO_CONTENT)
async def unarchive_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    session = await db.scalar(
        select(Session).where(Session.id == session_id, Session.user_id == user.id),
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if session.archived_at is not None:
        session.archived_at = None
        session.last_activity_at = datetime.now(UTC)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    session = await db.scalar(
        select(Session).where(Session.id == session_id, Session.user_id == user.id),
    )
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    await db.delete(session)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
