"""Observable ChatOrchestrator wrapper for testing.

Extends ChatOrchestrator to inject observation hooks at each stage of the
message handling pipeline, capturing all data for test analysis.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from datetime import datetime, UTC
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.orchestration.core import ChatOrchestrator
from apps.api.orchestration.types import (
    _TurnPreparation,
    _TurnStreamState,
    _TurnPostProcessResult,
)
from apps.api.orchestration.openai_stream_service import ResponsesEventStreamer
from apps.api.schemas.requests import ChatSendRequest
from packages.infra.db.models.session import Session
from packages.infra.db.models.user import User

from .observer import TurnObserver
from .observation_types import TurnObservation


class ObservableChatOrchestrator(ChatOrchestrator):
    """ChatOrchestrator with observation hooks for testing.

    This class wraps the production ChatOrchestrator and captures all
    interaction data at each stage of the pipeline:

    1. _prepare_turn_context: Captures instructions, tools, enriched input
    2. _stream_openai_and_collect: Captures raw response, tokens, latency
    3. _post_process_turn: Captures patches, genui, artifacts, transitions

    Usage:
        observer = TurnObserver(session_id, user_id)
        orchestrator = ObservableChatOrchestrator(db, observer)

        async for event in orchestrator.handle_message_stream(...):
            # Process events as normal
            ...

        # After the turn, get the observation
        turn_obs = observer.finalize_turn()
    """

    def __init__(self, db: AsyncSession, observer: TurnObserver) -> None:
        super().__init__(db)
        self._observer = observer
        self._current_session: Session | None = None
        self._current_user_message: str = ""

    async def handle_message_stream(
        self,
        user: User,
        payload: ChatSendRequest,
        streamer: ResponsesEventStreamer,
        *,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """Stream a turn with observation hooks.

        Wraps the parent implementation to capture data at each stage.
        """
        # Capture user message for observation
        self._current_user_message = payload.message

        # Get session to capture artifacts before the turn
        session = await self._resolve_session(
            user_id=user.id, session_id=payload.session_id
        )
        self._current_session = session

        # Start turn observation
        artifacts_before = copy.deepcopy(session.artifacts or {})
        self._observer.start_turn(payload.message, artifacts_before)

        # Delegate to parent implementation
        async for event in super().handle_message_stream(
            user, payload, streamer, language=language
        ):
            yield event

    async def _prepare_turn_context(
        self,
        *,
        session: Session,
        user: User,
        payload: ChatSendRequest,
        language: str,
        phase_before: str,
        user_runtime_policy: Any,
        prompt_user_message: str,
        phase_turn_count: int,
        handler: Any,
        turn_request_id: str,
        turn_id: str,
        user_message_id: UUID,
    ) -> _TurnPreparation:
        """Capture preparation data before delegating to parent."""
        preparation = await super()._prepare_turn_context(
            session=session,
            user=user,
            payload=payload,
            language=language,
            phase_before=phase_before,
            user_runtime_policy=user_runtime_policy,
            prompt_user_message=prompt_user_message,
            phase_turn_count=phase_turn_count,
            handler=handler,
            turn_request_id=turn_request_id,
            turn_id=turn_id,
            user_message_id=user_message_id,
        )

        # Capture preparation data
        self._observer.capture_preparation(
            turn_id=preparation.turn_id,
            user_message_id=preparation.user_message_id,
            phase_before=preparation.phase_before,
            phase_turn_count=preparation.phase_turn_count,
            prompt_user_message=preparation.prompt_user_message,
            artifacts=preparation.artifacts,
            instructions=preparation.prompt.instructions,
            enriched_input=preparation.prompt.enriched_input,
            tools=preparation.tools,
            tool_choice=preparation.prompt.tool_choice,
            model=preparation.prompt.model,
            max_output_tokens=None,  # Not available in PromptPieces
            reasoning=preparation.prompt.reasoning,
            phase_stage=preparation.ctx.runtime_policy.phase_stage,
        )

        return preparation

    async def _stream_openai_and_collect(
        self,
        *,
        session: Session,
        streamer: ResponsesEventStreamer,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
    ) -> AsyncIterator[str]:
        """Capture stream data while delegating to parent."""
        # Mark stream start
        self._observer.capture_stream_start()

        # Delegate to parent
        async for event in super()._stream_openai_and_collect(
            session=session,
            streamer=streamer,
            preparation=preparation,
            stream_state=stream_state,
        ):
            yield event

        # Capture stream result after completion
        self._observer.capture_stream_result(
            full_text=stream_state.full_text,
            instructions_sent=True,  # Instructions are always sent in current implementation
            request_model=stream_state.request_model,
            completed_model=stream_state.completed_model,
            completed_usage=stream_state.completed_usage,
            stream_error_message=stream_state.stream_error_message,
            stream_error_detail=stream_state.stream_error_detail,
            mcp_call_records=stream_state.mcp_call_records,
            mcp_call_order=stream_state.mcp_call_order,
            response_id=session.previous_response_id,
        )

    async def _post_process_turn(
        self,
        *,
        session: Session,
        user: User,
        payload: ChatSendRequest,
        language: str,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
    ) -> _TurnPostProcessResult:
        """Capture post-process data after delegating to parent."""
        result = await super()._post_process_turn(
            session=session,
            user=user,
            payload=payload,
            language=language,
            preparation=preparation,
            stream_state=stream_state,
        )

        # Capture post-process result
        self._observer.capture_post_process(
            assistant_text=result.assistant_text,
            cleaned_text=result.cleaned_text,
            filtered_genui_payloads=result.filtered_genui_payloads,
            persisted_tool_calls=result.persisted_tool_calls,
            missing_fields=result.missing_fields,
            transitioned=result.transitioned,
            transition_from_phase=result.transition_from_phase,
            transition_to_phase=result.transition_to_phase,
            artifacts_after=session.artifacts or {},
        )

        return result

    async def _emit_tail_events_and_persist(
        self,
        *,
        session: Session,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
        post_process_result: _TurnPostProcessResult,
        turn_id: str,
    ) -> AsyncIterator[str]:
        """Finalize turn observation after persistence."""
        # Delegate to parent first
        assistant_message_id: UUID | None = None

        async for event in super()._emit_tail_events_and_persist(
            session=session,
            preparation=preparation,
            stream_state=stream_state,
            post_process_result=post_process_result,
            turn_id=turn_id,
        ):
            # Try to extract assistant_message_id from done event
            if '"type": "done"' in event or '"type":"done"' in event:
                import json
                try:
                    # Parse SSE event
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data.get("type") == "done":
                                msg_id = data.get("assistant_message_id")
                                if msg_id:
                                    assistant_message_id = UUID(msg_id)
                except (json.JSONDecodeError, ValueError):
                    pass
            yield event

        # Finalize the turn observation
        self._observer.finalize_turn(assistant_message_id)


class MockResponsesEventStreamer:
    """Mock streamer for testing without real OpenAI calls.

    Useful for unit testing the observation infrastructure itself.
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    async def stream_events(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield pre-configured mock events."""
        self._call_count += 1
        for event in self._responses:
            yield event

    @property
    def call_count(self) -> int:
        """Number of times stream_events was called."""
        return self._call_count

    @staticmethod
    def create_simple_response(text: str, model: str = "gpt-4o") -> list[dict[str, Any]]:
        """Create a simple mock response sequence."""
        return [
            {"type": "response.created", "response": {"id": f"resp_{datetime.now(UTC).timestamp()}"}},
            {"type": "response.output_text.delta", "delta": text, "sequence_number": 1},
            {"type": "response.output_text.done", "text": text, "sequence_number": 2},
            {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{datetime.now(UTC).timestamp()}",
                    "model": model,
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                },
                "sequence_number": 3,
            },
        ]
