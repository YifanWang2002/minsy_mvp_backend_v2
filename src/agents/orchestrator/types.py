"""Dataclasses used across orchestrator modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.handler_protocol import PhaseContext


@dataclass(slots=True)
class _TurnPreparation:
    phase_before: str
    phase_turn_count: int
    prompt_user_message: str
    handler: Any
    artifacts: dict[str, Any]
    pre_strategy_instrument_before: str | None
    ctx: PhaseContext
    prompt: Any
    tools: list[dict[str, Any]]


@dataclass(slots=True)
class _TurnStreamState:
    full_text: str = ""
    text_delta_emitted: bool = False
    request_model: str | None = None
    completed_model: str | None = None
    completed_usage: dict[str, Any] = field(default_factory=dict)
    stream_error_message: str | None = None
    stream_error_detail: dict[str, Any] | None = None
    mcp_call_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_call_order: list[str] = field(default_factory=list)
    mcp_fallback_counter: int = 0


@dataclass(slots=True)
class _TurnPostProcessResult:
    assistant_text: str
    cleaned_text: str
    filtered_genui_payloads: list[dict[str, Any]]
    persisted_tool_calls: list[dict[str, Any]]
    missing_fields: list[str]
    kyc_status: str
    transitioned: bool
    stop_criteria_delta: str | None
