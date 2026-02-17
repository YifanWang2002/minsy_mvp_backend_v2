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
    session_title: str | None = None
    session_title_record: dict[str, Any] | None = None


class SessionListItem(BaseModel):
    """Light session item for session list response."""

    session_id: UUID
    current_phase: str
    status: str
    updated_at: datetime
    archived_at: datetime | None = None
    session_title: str | None = None
    session_title_record: dict[str, Any] | None = None


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
    session_title: str | None = None
    session_title_record: dict[str, Any] | None = None
    artifacts: dict[str, Any]
    metadata: dict[str, Any]
    last_activity_at: datetime
    messages: list[MessageItem] = Field(default_factory=list)


class StrategyBacktestSummary(BaseModel):
    """Backtest summary attached to one strategy/version view."""

    job_id: UUID | None = None
    status: str | None = None
    strategy_version: int | None = None
    total_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe_ratio: float | None = None
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    completed_at: datetime | None = None


class StrategyListItemResponse(BaseModel):
    """One strategy row in the authenticated user's strategy list."""

    strategy_id: UUID
    session_id: UUID
    version: int
    status: str
    dsl_json: dict[str, Any]
    metadata: dict[str, Any]
    latest_backtest: StrategyBacktestSummary | None = None


class StrategyVersionItemResponse(BaseModel):
    """One historical DSL snapshot with optional backtest summary."""

    strategy_id: UUID
    version: int
    dsl_json: dict[str, Any]
    revision: dict[str, Any]
    backtest: StrategyBacktestSummary | None = None


class StrategyVersionDiffItem(BaseModel):
    """Display-friendly diff item between two strategy versions."""

    op: str
    path: str
    old_value: Any | None = None
    new_value: Any | None = None


class StrategyVersionDiffResponse(BaseModel):
    """Version-to-version diff payload for frontend rendering."""

    strategy_id: UUID
    from_version: int
    to_version: int
    patch_op_count: int
    patch_ops: list[dict[str, Any]] = Field(default_factory=list)
    diff_items: list[StrategyVersionDiffItem] = Field(default_factory=list)
    from_payload_hash: str
    to_payload_hash: str


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


class StrategyDraftDetailResponse(BaseModel):
    """Temporary strategy draft payload for pre-confirmation rendering."""

    strategy_draft_id: UUID
    session_id: UUID
    dsl_json: dict[str, Any]
    expires_at: datetime
    metadata: dict[str, Any]
