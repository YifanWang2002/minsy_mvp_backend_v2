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
    token_usage: dict[str, Any] | None = None


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
    stream_recovery: dict[str, Any] | None = None
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


class SocialConnectorItem(BaseModel):
    """One social connector status entry for settings page."""

    provider: str
    status: str
    connected_account: str | None = None
    connected_at: datetime | None = None
    supports_connect: bool = False


class TelegramConnectLinkResponse(BaseModel):
    """Telegram connect-link payload returned to frontend."""

    provider: str
    connect_url: str
    expires_at: datetime


class TelegramActivityItem(BaseModel):
    """Telegram interaction event item for settings timeline."""

    id: UUID
    event_type: str
    choice_value: str | None = None
    message_text: str | None = None
    created_at: datetime


class TelegramActivitiesResponse(BaseModel):
    """Telegram connector activity list."""

    provider: str
    items: list[TelegramActivityItem] = Field(default_factory=list)


class NotificationPreferencesResponse(BaseModel):
    """Notification preference payload for settings page."""

    user_id: UUID
    telegram_enabled: bool
    backtest_completed_enabled: bool
    deployment_started_enabled: bool
    position_opened_enabled: bool
    position_closed_enabled: bool
    risk_triggered_enabled: bool
    execution_anomaly_enabled: bool


class TradingPreferenceResponse(BaseModel):
    """Trading execution mode + approval configuration payload."""

    user_id: UUID
    execution_mode: str
    approval_channel: str
    approval_timeout_seconds: int
    approval_scope: str


class TradeApprovalRequestResponse(BaseModel):
    """Trade approval request detail payload."""

    trade_approval_request_id: UUID
    user_id: UUID
    deployment_id: UUID
    execution_order_id: UUID | None = None
    signal: str
    side: str
    symbol: str
    qty: float
    mark_price: float
    reason: str
    timeframe: str
    bar_time: datetime | None = None
    approval_channel: str
    status: str
    approval_key: str
    requested_at: datetime
    expires_at: datetime
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    expired_at: datetime | None = None
    executed_at: datetime | None = None
    approved_via: str | None = None
    decision_actor: str | None = None
    execution_error: str | None = None
    intent_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TelegramTestTargetResponse(BaseModel):
    """Resolved Telegram test-target diagnostics."""

    configured_email: str
    resolved_user_exists: bool
    resolved_binding_connected: bool
    resolved_chat_id_masked: str | None = None
    resolved_binding_id: UUID | None = None
    resolved_username: str | None = None
    resolved_user_id: UUID | None = None


class TelegramTestSendResponse(BaseModel):
    """Telegram debug send result payload."""

    ok: bool
    actual_target: str
    message_id: str | None = None
    detail: str | None = None


class BrokerAccountResponse(BaseModel):
    """Broker account payload for API responses."""

    broker_account_id: UUID
    user_id: UUID
    provider: str
    mode: str
    status: str
    key_fingerprint: str | None = None
    encryption_version: str | None = None
    updated_source: str | None = None
    last_validated_at: datetime | None = None
    last_validated_status: str | None = None
    validation_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DeploymentRunResponse(BaseModel):
    """Deployment runtime snapshot."""

    deployment_run_id: UUID
    deployment_id: UUID
    strategy_id: UUID
    broker_account_id: UUID
    status: str
    last_bar_time: datetime | None = None
    timeframe_seconds: int | None = None
    last_trigger_bucket: int | None = None
    last_enqueued_at: datetime | None = None
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DeploymentResponse(BaseModel):
    """Deployment detail payload."""

    deployment_id: UUID
    strategy_id: UUID
    user_id: UUID
    mode: str
    status: str
    market: str | None = None
    symbols: list[str] = Field(default_factory=list)
    timeframe: str | None = None
    capital_allocated: float
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    deployed_at: datetime | None = None
    stopped_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    run: DeploymentRunResponse | None = None


class DeploymentActionResponse(BaseModel):
    """Lifecycle transition response payload."""

    deployment: DeploymentResponse
    queued_task_id: str | None = None


class OrderResponse(BaseModel):
    """Order list/detail payload."""

    order_id: UUID
    deployment_id: UUID
    provider_order_id: str | None = None
    client_order_id: str
    symbol: str
    side: str
    type: str
    qty: float
    price: float | None = None
    status: str
    provider_status: str | None = None
    reject_reason: str | None = None
    last_sync_at: datetime | None = None
    submitted_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class FillResponse(BaseModel):
    """Fill list payload."""

    fill_id: UUID
    order_id: UUID
    provider_fill_id: str | None = None
    fill_price: float
    fill_qty: float
    fee: float
    filled_at: datetime
    created_at: datetime
    updated_at: datetime


class PositionResponse(BaseModel):
    """Position list payload."""

    position_id: UUID
    deployment_id: UUID
    symbol: str
    side: str
    qty: float
    avg_entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    created_at: datetime
    updated_at: datetime


class PnlSnapshotResponse(BaseModel):
    """PnL snapshot payload."""

    pnl_snapshot_id: UUID
    deployment_id: UUID
    source: str = "platform_estimate"
    equity: float
    cash: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    snapshot_time: datetime
    created_at: datetime
    updated_at: datetime


class ManualTradeActionResponse(BaseModel):
    """Manual trade action payload."""

    manual_trade_action_id: UUID
    user_id: UUID
    deployment_id: UUID
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: datetime
    updated_at: datetime


class MarketDataQuoteResponse(BaseModel):
    """Quote snapshot payload for frontend polling."""

    market: str
    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    timestamp: datetime
    source: str = "runtime_cache"


class MarketDataBarResponse(BaseModel):
    """One normalized OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataBarsResponse(BaseModel):
    """Bar list payload."""

    market: str
    symbol: str
    timeframe: str
    bars: list[MarketDataBarResponse] = Field(default_factory=list)


class MarketDataSubscriptionResponse(BaseModel):
    """Subscription dedup result payload."""

    subscriber_id: str
    added_symbols: list[str] = Field(default_factory=list)
    removed_symbols: list[str] = Field(default_factory=list)
    active_symbols: list[str] = Field(default_factory=list)


class SignalResponse(BaseModel):
    """Live signal event payload."""

    signal_event_id: UUID | None = None
    deployment_id: UUID
    signal: str
    symbol: str
    timeframe: str
    bar_time: datetime
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentSignalProcessResponse(BaseModel):
    """Response after processing one deployment bar-close cycle."""

    deployment_id: UUID
    execution_event_id: UUID | None = None
    signal: str
    reason: str
    order_id: UUID | None = None
    idempotent_hit: bool = False
    bar_time: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrokerAccountSnapshotResponse(BaseModel):
    """Broker-reported account snapshot attached to deployment runtime."""

    provider: str
    source: str
    sync_status: str
    fetched_at: datetime | None = None
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    margin_used: float | None = None
    unrealized_pnl: float | None = None
    realized_pnl: float | None = None
    positions_count: int | None = None
    symbols: list[str] = Field(default_factory=list)
    error: str | None = None
    updated_at: datetime | None = None


class PortfolioResponse(BaseModel):
    """Portfolio aggregate payload for one deployment."""

    deployment_id: UUID
    metrics_source: str = "platform_estimate"
    equity: float
    cash: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    snapshot_time: datetime
    broker_account: BrokerAccountSnapshotResponse | None = None
    positions: list[PositionResponse] = Field(default_factory=list)
