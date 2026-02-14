"""Response schemas for chat/session APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ThreadResponse(BaseModel):
    """Response after new-thread creation."""

    session_id: UUID
    phase: str
    status: str
    kyc_status: str


class SessionListItem(BaseModel):
    """Light session item for session list response."""

    session_id: UUID
    current_phase: str
    status: str
    updated_at: datetime
    archived_at: datetime | None = None


class MessageItem(BaseModel):
    """Serialized message item in session detail."""

    id: UUID
    role: str
    content: str
    phase: str
    created_at: datetime
    # Mixed payloads persisted with assistant messages:
    # - GenUI blocks (e.g. choice_prompt/tradingview_chart/strategy_card)
    # - MCP tool-call final results (type=mcp_call, status in {success,failure})
    tool_calls: list[dict[str, Any]] | None = None


class SessionDetailResponse(BaseModel):
    """Detailed session response with messages and artifacts."""

    session_id: UUID
    current_phase: str
    status: str
    archived_at: datetime | None = None
    artifacts: dict[str, Any]
    metadata: dict[str, Any]
    last_activity_at: datetime
    messages: list[MessageItem] = Field(default_factory=list)


class StrategyConfirmResponse(BaseModel):
    """Response for frontend strategy confirmation + optional auto backtest turn."""

    session_id: UUID
    strategy_id: UUID
    phase: str
    metadata: dict[str, Any]
    auto_started: bool = False
    auto_message: str | None = None
    auto_assistant_text: str | None = None
    auto_done_payload: dict[str, Any] | None = None
    auto_error: str | None = None


class StrategyDetailResponse(BaseModel):
    """Strategy detail payload for frontend rendering/query by id."""

    strategy_id: UUID
    session_id: UUID
    version: int
    status: str
    dsl_json: dict[str, Any]
    metadata: dict[str, Any]
