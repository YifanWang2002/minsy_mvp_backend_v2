"""Turn observer for capturing orchestrator interaction data.

The TurnObserver collects data at each stage of the orchestrator pipeline:
1. Preparation: instructions, tools, enriched input
2. Streaming: raw response, token usage, latency
3. Post-processing: patches, genui, artifacts, transitions

This data is assembled into TurnObservation objects for analysis.
"""

from __future__ import annotations

import copy
from datetime import datetime, UTC
from typing import Any
from uuid import UUID

from .observation_types import (
    ConversationObservation,
    TurnObservation,
    TurnPreparationCapture,
    TurnStreamCapture,
    TurnPostProcessCapture,
)


class TurnObserver:
    """Observes and captures orchestrator turn data.

    Usage:
        observer = TurnObserver(session_id, user_id)
        # ... orchestrator calls observer.capture_* methods ...
        observation = observer.finalize()
    """

    def __init__(self, session_id: UUID, user_id: UUID) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._started_at = datetime.now(UTC)
        self._turn_number = 0

        # Current turn captures (reset each turn)
        self._current_preparation: TurnPreparationCapture | None = None
        self._current_stream: TurnStreamCapture | None = None
        self._current_post_process: TurnPostProcessCapture | None = None
        self._current_user_message: str = ""
        self._current_artifacts_before: dict[str, Any] = {}

        # Accumulated observations
        self._conversation = ConversationObservation(
            session_id=session_id,
            user_id=user_id,
            started_at=self._started_at,
        )

    def start_turn(self, user_message: str, artifacts_before: dict[str, Any]) -> None:
        """Mark the start of a new turn."""
        self._turn_number += 1
        self._current_preparation = None
        self._current_stream = None
        self._current_post_process = None
        self._current_user_message = user_message
        self._current_artifacts_before = copy.deepcopy(artifacts_before)

    def capture_preparation(
        self,
        *,
        turn_id: str,
        user_message_id: UUID,
        phase_before: str,
        phase_turn_count: int,
        prompt_user_message: str,
        artifacts: dict[str, Any],
        instructions: str,
        enriched_input: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        model: str | None,
        max_output_tokens: int | None,
        reasoning: dict[str, Any] | None,
        phase_stage: str | None,
    ) -> None:
        """Capture data from _prepare_turn_context."""
        self._current_preparation = TurnPreparationCapture(
            turn_id=turn_id,
            user_message_id=user_message_id,
            phase_before=phase_before,
            phase_turn_count=phase_turn_count,
            prompt_user_message=prompt_user_message,
            artifacts=copy.deepcopy(artifacts),
            instructions=instructions,
            enriched_input=enriched_input,
            tools=copy.deepcopy(tools or []),
            tool_choice=copy.deepcopy(tool_choice) if tool_choice else None,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning=copy.deepcopy(reasoning) if reasoning else None,
            phase_stage=phase_stage,
        )

    def capture_stream_start(self) -> None:
        """Mark the start of streaming."""
        # Initialize stream capture with start time
        self._stream_start_time = datetime.now(UTC)

    def capture_stream_result(
        self,
        *,
        full_text: str,
        instructions_sent: bool,
        request_model: str | None,
        completed_model: str | None,
        completed_usage: dict[str, Any],
        stream_error_message: str | None,
        stream_error_detail: dict[str, Any] | None,
        mcp_call_records: dict[str, dict[str, Any]],
        mcp_call_order: list[str],
        response_id: str | None,
    ) -> None:
        """Capture data from _stream_openai_and_collect."""
        stream_end_time = datetime.now(UTC)
        stream_start_time = getattr(self, "_stream_start_time", stream_end_time)

        self._current_stream = TurnStreamCapture(
            full_text=full_text,
            instructions_sent=instructions_sent,
            request_model=request_model,
            completed_model=completed_model,
            completed_usage=copy.deepcopy(completed_usage),
            stream_error_message=stream_error_message,
            stream_error_detail=(
                copy.deepcopy(stream_error_detail) if stream_error_detail else None
            ),
            mcp_call_records=copy.deepcopy(mcp_call_records),
            mcp_call_order=list(mcp_call_order),
            response_id=response_id,
            stream_start_time=stream_start_time,
            stream_end_time=stream_end_time,
        )

    def capture_post_process(
        self,
        *,
        assistant_text: str,
        cleaned_text: str,
        filtered_genui_payloads: list[dict[str, Any]],
        persisted_tool_calls: list[dict[str, Any]],
        missing_fields: list[str],
        transitioned: bool,
        transition_from_phase: str | None,
        transition_to_phase: str | None,
        artifacts_after: dict[str, Any],
    ) -> None:
        """Capture data from _post_process_turn."""
        self._current_post_process = TurnPostProcessCapture(
            assistant_text=assistant_text,
            cleaned_text=cleaned_text,
            filtered_genui_payloads=copy.deepcopy(filtered_genui_payloads),
            persisted_tool_calls=copy.deepcopy(persisted_tool_calls),
            missing_fields=list(missing_fields),
            transitioned=transitioned,
            transition_from_phase=transition_from_phase,
            transition_to_phase=transition_to_phase,
            artifacts_after=copy.deepcopy(artifacts_after),
        )

    def finalize_turn(self, assistant_message_id: UUID | None = None) -> TurnObservation:
        """Assemble captured data into a TurnObservation."""
        if self._current_preparation is None:
            raise ValueError("No preparation data captured for this turn")

        prep = self._current_preparation
        stream = self._current_stream
        post = self._current_post_process

        # Extract token usage
        usage = stream.completed_usage if stream else {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Calculate latency
        latency_ms = 0.0
        stream_start = None
        stream_end = None
        if stream:
            stream_start = stream.stream_start_time
            stream_end = stream.stream_end_time
            latency_ms = (stream_end - stream_start).total_seconds() * 1000

        # Build MCP tool calls list
        mcp_tool_calls: list[dict[str, Any]] = []
        if stream:
            for call_id in stream.mcp_call_order:
                record = stream.mcp_call_records.get(call_id)
                if record:
                    mcp_tool_calls.append(record)

        # Determine phase transition
        phase_transition: tuple[str, str] | None = None
        if post and post.transitioned and post.transition_from_phase and post.transition_to_phase:
            phase_transition = (post.transition_from_phase, post.transition_to_phase)

        # Extract genui from post-process (excluding MCP tool calls)
        extracted_genui = []
        if post:
            extracted_genui = [
                p for p in post.filtered_genui_payloads
                if p.get("type") != "mcp_tool_call"
            ]

        observation = TurnObservation(
            turn_number=self._turn_number,
            timestamp=datetime.now(UTC),
            user_message=self._current_user_message,
            phase=prep.phase_before,
            phase_stage=prep.phase_stage,
            phase_turn_count=prep.phase_turn_count,
            session_state_snapshot=self._current_artifacts_before,
            instructions=prep.instructions,
            instructions_sent=stream.instructions_sent if stream else False,
            enriched_input=prep.enriched_input,
            tools=prep.tools,
            tool_choice=prep.tool_choice,
            model=stream.completed_model or stream.request_model or prep.model or "unknown",
            max_output_tokens=prep.max_output_tokens,
            reasoning_config=prep.reasoning,
            raw_response_text=stream.full_text if stream else "",
            cleaned_text=post.cleaned_text if post else "",
            extracted_patches=[],  # Patches are processed internally
            extracted_genui=extracted_genui,
            mcp_tool_calls=mcp_tool_calls,
            artifacts_before=self._current_artifacts_before,
            artifacts_after=post.artifacts_after if post else prep.artifacts,
            missing_fields=post.missing_fields if post else [],
            phase_transition=phase_transition,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            stream_start_time=stream_start,
            stream_end_time=stream_end,
            stream_error=stream.stream_error_message if stream else None,
            stream_error_detail=stream.stream_error_detail if stream else None,
            response_id=stream.response_id if stream else None,
            assistant_message_id=assistant_message_id,
        )

        self._conversation.add_turn(observation)
        return observation

    def finalize(self) -> ConversationObservation:
        """Finalize and return the complete conversation observation."""
        self._conversation.finalize(datetime.now(UTC))
        return self._conversation

    @property
    def current_turn_number(self) -> int:
        """Get the current turn number."""
        return self._turn_number

    @property
    def conversation(self) -> ConversationObservation:
        """Get the conversation observation (may be incomplete)."""
        return self._conversation
