"""Common request schemas for chat/session APIs."""

from __future__ import annotations

import json
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
