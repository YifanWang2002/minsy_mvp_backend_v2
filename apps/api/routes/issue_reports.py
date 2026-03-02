"""Issue report ingestion endpoint for frontend bug submissions."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import IssueReportCreateResponse
from apps.api.schemas.requests import IssueReportCreateRequest
from packages.infra.db.models.session import Session
from packages.infra.db.models.user import User

router = APIRouter(prefix="/issue-reports", tags=["issue-reports"])

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_ISSUE_REPORTS_DIR = _BACKEND_ROOT / "runtime" / "issue_reports"
_MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _decode_png_bytes(raw_base64: str) -> bytes:
    normalized = raw_base64.strip()
    if normalized.lower().startswith("data:image/png;base64,"):
        normalized = normalized.split(",", 1)[1]
    try:
        payload = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid screenshot payload.",
        ) from exc

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Screenshot payload is empty.",
        )
    if len(payload) > _MAX_SCREENSHOT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Screenshot payload is too large.",
        )
    if not payload.startswith(_PNG_SIGNATURE):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Screenshot payload must be a PNG image.",
        )
    return payload


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


async def _load_owned_session(
    db: AsyncSession,
    *,
    user_id: UUID,
    session_id: UUID | None,
) -> Session | None:
    if session_id is None:
        return None
    return await db.scalar(
        select(Session).where(Session.id == session_id, Session.user_id == user_id),
    )


async def _load_latest_session(
    db: AsyncSession,
    *,
    user_id: UUID,
) -> Session | None:
    return await db.scalar(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.updated_at.desc())
        .limit(1)
    )


@router.post(
    "",
    response_model=IssueReportCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_issue_report(
    payload: IssueReportCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> IssueReportCreateResponse:
    screenshot_bytes = _decode_png_bytes(payload.screenshot_png_base64)
    hinted_session = await _load_owned_session(
        db,
        user_id=user.id,
        session_id=payload.session_id_hint,
    )
    latest_session = await _load_latest_session(db, user_id=user.id)

    linked_session_id = hinted_session.id if hinted_session else None
    if linked_session_id is None and latest_session is not None:
        linked_session_id = latest_session.id

    stored_at = datetime.now(UTC)
    issue_report_id = uuid4()
    report_dir = (
        _ISSUE_REPORTS_DIR
        / stored_at.strftime("%Y")
        / stored_at.strftime("%m")
        / stored_at.strftime("%d")
        / str(issue_report_id)
    )
    report_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "issue_report_id": str(issue_report_id),
        "stored_at": _serialize_datetime(stored_at),
        "reported_at": _serialize_datetime(payload.reported_at),
        "received_at": _serialize_datetime(stored_at),
        "user_id": str(user.id),
        "user_email": user.email,
        "client_user_id": payload.client_user_id,
        "linked_session_id": str(linked_session_id) if linked_session_id else None,
        "latest_session_id": str(latest_session.id) if latest_session else None,
        "session_id_hint": str(payload.session_id_hint)
        if payload.session_id_hint
        else None,
        "current_route": payload.current_route,
        "settings_category": payload.settings_category,
        "source": payload.source,
        "locale": payload.locale,
        "theme_mode": payload.theme_mode,
        "platform": payload.platform,
        "description": payload.description,
        "client_context": dict(payload.client_context or {}),
        "screenshot": {
            "filename": "screenshot.png",
            "content_type": "image/png",
            "size_bytes": len(screenshot_bytes),
        },
    }

    (report_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "screenshot.png").write_bytes(screenshot_bytes)

    return IssueReportCreateResponse(
        issue_report_id=issue_report_id,
        resolved_session_id=linked_session_id,
        stored_at=stored_at,
    )
