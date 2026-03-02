"""Core data types for orchestrator test observations.

These dataclasses capture the complete state of each turn in an orchestrator
conversation, including:
- All inputs sent to the AI (instructions, tools, enriched input)
- All outputs from the AI (text, patches, genui, tool calls)
- Performance metrics (tokens, latency)
- Phase transitions and artifact mutations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class TurnObservation:
    """Complete observation data for a single conversation turn.

    Captures everything that happened during one user->assistant exchange,
    including what was sent to the AI, what it returned, and the resulting
    state changes.
    """

    turn_number: int
    timestamp: datetime

    # === Input Side ===
    user_message: str
    phase: str
    phase_stage: str | None
    phase_turn_count: int
    session_state_snapshot: dict[str, Any]

    # === Sent to AI ===
    instructions: str
    instructions_sent: bool  # Whether instructions were actually sent this turn
    enriched_input: str  # [SESSION STATE] + user message
    tools: list[dict[str, Any]]
    tool_choice: dict[str, Any] | None
    model: str
    max_output_tokens: int | None
    reasoning_config: dict[str, Any] | None

    # === AI Response ===
    raw_response_text: str
    cleaned_text: str
    extracted_patches: list[dict[str, Any]]
    extracted_genui: list[dict[str, Any]]
    mcp_tool_calls: list[dict[str, Any]]

    # === Post-Processing Results ===
    artifacts_before: dict[str, Any]
    artifacts_after: dict[str, Any]
    missing_fields: list[str]
    phase_transition: tuple[str, str] | None  # (from_phase, to_phase)

    # === Performance Metrics ===
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    stream_start_time: datetime | None = None
    stream_end_time: datetime | None = None

    # === Error State ===
    stream_error: str | None = None
    stream_error_detail: dict[str, Any] | None = None

    # === Response Metadata ===
    response_id: str | None = None
    assistant_message_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "phase": self.phase,
            "phase_stage": self.phase_stage,
            "phase_turn_count": self.phase_turn_count,
            "session_state_snapshot": self.session_state_snapshot,
            "instructions": self.instructions,
            "instructions_sent": self.instructions_sent,
            "enriched_input": self.enriched_input,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_config": self.reasoning_config,
            "raw_response_text": self.raw_response_text,
            "cleaned_text": self.cleaned_text,
            "extracted_patches": self.extracted_patches,
            "extracted_genui": self.extracted_genui,
            "mcp_tool_calls": self.mcp_tool_calls,
            "artifacts_before": self.artifacts_before,
            "artifacts_after": self.artifacts_after,
            "missing_fields": self.missing_fields,
            "phase_transition": self.phase_transition,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "stream_start_time": (
                self.stream_start_time.isoformat() if self.stream_start_time else None
            ),
            "stream_end_time": (
                self.stream_end_time.isoformat() if self.stream_end_time else None
            ),
            "stream_error": self.stream_error,
            "stream_error_detail": self.stream_error_detail,
            "response_id": self.response_id,
            "assistant_message_id": (
                str(self.assistant_message_id) if self.assistant_message_id else None
            ),
        }


@dataclass
class ConversationObservation:
    """Complete observation data for an entire conversation.

    Aggregates all turn observations and provides summary statistics.
    """

    session_id: UUID
    user_id: UUID
    started_at: datetime
    ended_at: datetime | None = None

    turns: list[TurnObservation] = field(default_factory=list)

    # === Summary Statistics ===
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0

    # === Final State ===
    final_phase: str = ""
    final_artifacts: dict[str, Any] = field(default_factory=dict)
    phases_visited: list[str] = field(default_factory=list)
    phase_transitions: list[tuple[str, str]] = field(default_factory=list)

    # === Error Summary ===
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add_turn(self, turn: TurnObservation) -> None:
        """Add a turn observation and update aggregates."""
        self.turns.append(turn)
        self.total_input_tokens += turn.input_tokens
        self.total_output_tokens += turn.output_tokens
        self.total_tokens += turn.total_tokens
        self.total_duration_ms += turn.latency_ms

        if turn.phase not in self.phases_visited:
            self.phases_visited.append(turn.phase)

        if turn.phase_transition:
            self.phase_transitions.append(turn.phase_transition)

        if turn.stream_error:
            self.errors.append(
                {
                    "turn": turn.turn_number,
                    "error": turn.stream_error,
                    "detail": turn.stream_error_detail,
                }
            )

        self.final_phase = turn.phase
        self.final_artifacts = turn.artifacts_after

    def finalize(self, ended_at: datetime | None = None) -> None:
        """Mark the conversation as complete."""
        self.ended_at = ended_at or datetime.now()
        if self.turns:
            last_turn = self.turns[-1]
            if last_turn.phase_transition:
                self.final_phase = last_turn.phase_transition[1]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "session_id": str(self.session_id),
            "user_id": str(self.user_id),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turns": [t.to_dict() for t in self.turns],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "final_phase": self.final_phase,
            "final_artifacts": self.final_artifacts,
            "phases_visited": self.phases_visited,
            "phase_transitions": self.phase_transitions,
            "errors": self.errors,
        }


@dataclass
class TurnPreparationCapture:
    """Captured data from _prepare_turn_context."""

    turn_id: str
    user_message_id: UUID
    phase_before: str
    phase_turn_count: int
    prompt_user_message: str
    artifacts: dict[str, Any]
    instructions: str
    enriched_input: str
    tools: list[dict[str, Any]]
    tool_choice: dict[str, Any] | None
    model: str | None
    max_output_tokens: int | None
    reasoning: dict[str, Any] | None
    phase_stage: str | None


@dataclass
class TurnStreamCapture:
    """Captured data from _stream_openai_and_collect."""

    full_text: str
    instructions_sent: bool
    request_model: str | None
    completed_model: str | None
    completed_usage: dict[str, Any]
    stream_error_message: str | None
    stream_error_detail: dict[str, Any] | None
    mcp_call_records: dict[str, dict[str, Any]]
    mcp_call_order: list[str]
    response_id: str | None
    stream_start_time: datetime
    stream_end_time: datetime


@dataclass
class TurnPostProcessCapture:
    """Captured data from _post_process_turn."""

    assistant_text: str
    cleaned_text: str
    filtered_genui_payloads: list[dict[str, Any]]
    persisted_tool_calls: list[dict[str, Any]]
    missing_fields: list[str]
    transitioned: bool
    transition_from_phase: str | None
    transition_to_phase: str | None
    artifacts_after: dict[str, Any]
