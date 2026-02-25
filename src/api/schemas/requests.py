"""Common request schemas for chat/session APIs."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class NewThreadRequest(BaseModel):
    """Create a new workflow session."""

    parent_session_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimePolicy(BaseModel):
    """Per-turn runtime controls for handler prompt/tool exposure."""

    phase_stage: str | None = Field(default=None, min_length=1, max_length=64)
    tool_mode: Literal["append", "replace"] = "append"
    allowed_tools: list[dict[str, Any]] | None = None

    @field_validator("phase_stage")
    @classmethod
    def validate_phase_stage(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized


class ChatSendRequest(BaseModel):
    """Send one user message to orchestrator."""

    session_id: UUID | None = None
    message: str = Field(min_length=1, max_length=4000)
    runtime_policy: RuntimePolicy | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be empty.")
        return normalized


class StrategyConfirmRequest(BaseModel):
    """Persist a reviewed strategy DSL and optionally auto-start backtest turn."""

    session_id: UUID
    dsl_json: dict[str, Any] | str
    strategy_id: UUID | None = None
    auto_start_backtest: bool = False
    # Deprecated compatibility field. Currently ignored: strategy iteration
    # remains in strategy phase until dedicated stress-test tools are available.
    advance_to_stress_test: bool = False
    language: str = Field(default="en", min_length=2, max_length=16)
    auto_message: str | None = Field(default=None, max_length=4000)

    @field_validator("dsl_json", mode="before")
    @classmethod
    def validate_dsl_json(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("dsl_json cannot be empty.")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"dsl_json must be valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("dsl_json must decode to a JSON object.")
            return parsed
        raise ValueError("dsl_json must be a JSON object or JSON string.")

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            return "en"
        return normalized

    @field_validator("auto_message")
    @classmethod
    def validate_auto_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized


class TelegramConnectLinkRequest(BaseModel):
    """Request payload for Telegram connect-link generation."""

    locale: str = Field(default="en", min_length=2, max_length=16)

    @field_validator("locale")
    @classmethod
    def validate_locale(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if normalized.startswith("zh"):
            return "zh"
        return "en"


class BrokerAccountCreateRequest(BaseModel):
    """Create one broker account binding for current user."""

    provider: Literal["alpaca", "ccxt"]
    mode: Literal["paper"] = "paper"
    credentials: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("credentials cannot be empty.")
        return value


class BrokerAccountCredentialsUpdateRequest(BaseModel):
    """Rotate credentials for an existing broker account."""

    credentials: dict[str, Any] = Field(default_factory=dict)

    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("credentials cannot be empty.")
        return value


class DeploymentCreateRequest(BaseModel):
    """Create one paper/live deployment runtime."""

    strategy_id: UUID
    broker_account_id: UUID
    mode: Literal["paper"] = "paper"
    capital_allocated: Decimal = Field(default=Decimal("0"), ge=0)
    risk_limits: dict[str, Any] = Field(default_factory=dict)
    runtime_state: dict[str, Any] = Field(default_factory=dict)


class ManualTradeActionRequest(BaseModel):
    """Submit user manual action for one deployment."""

    action: Literal["open", "close", "reduce", "stop"]
    payload: dict[str, Any] = Field(default_factory=dict)


class MarketDataSubscriptionRequest(BaseModel):
    """Subscribe current user to market-data symbols."""

    symbols: list[str] = Field(default_factory=list)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
        normalized = [symbol.strip().upper() for symbol in value if symbol.strip()]
        if not normalized:
            raise ValueError("symbols cannot be empty.")
        return normalized


class NotificationPreferencesUpdateRequest(BaseModel):
    """Patch payload for notification preference toggles."""

    telegram_enabled: bool | None = None
    backtest_completed_enabled: bool | None = None
    deployment_started_enabled: bool | None = None
    position_opened_enabled: bool | None = None
    position_closed_enabled: bool | None = None
    risk_triggered_enabled: bool | None = None
    execution_anomaly_enabled: bool | None = None


class TradingPreferencesUpdateRequest(BaseModel):
    """Patch payload for runtime execution/approval mode."""

    execution_mode: Literal["auto_execute", "approval_required"] | None = None
    approval_channel: Literal["telegram", "discord", "slack", "whatsapp"] | None = None
    approval_timeout_seconds: int | None = Field(default=None, ge=1, le=86_400)
    approval_scope: Literal["open_only", "open_and_close"] | None = None


class TradeApprovalDecisionRequest(BaseModel):
    """Decision payload for approve/reject operations."""

    note: str | None = Field(default=None, max_length=500)

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized


class TelegramTestSendRequest(BaseModel):
    """Payload for Telegram debug test-send endpoint."""

    message: str = Field(min_length=1, max_length=3000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be empty.")
        return normalized
