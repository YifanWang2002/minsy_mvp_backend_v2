"""Session orchestrator for chat phases and profile persistence.

Dispatches to pluggable :class:`PhaseHandler` instances via a registry,
so adding a new phase only requires writing a handler and registering it.

This module exclusively uses the OpenAI **Responses API** with
``previous_response_id`` for conversation context and separated
static instructions / dynamic state for prompt-caching efficiency.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agents.genui_registry import normalize_genui_payloads
from src.agents.handler_protocol import PhaseContext
from src.agents.handler_protocol import RuntimePolicy as HandlerRuntimePolicy
from src.agents.handler_registry import WORKFLOW_PHASES, get_handler, init_all_artifacts
from src.agents.handlers.kyc_handler import KYCHandler
from src.agents.phases import Phase, SessionStatus, can_transition
from src.agents.skills.pre_strategy_skills import (
    get_tradingview_symbol_for_market_instrument,
)
from src.api.schemas.requests import ChatSendRequest
from src.config import settings
from src.engine.strategy import validate_strategy_payload
from src.mcp.context_auth import MCP_CONTEXT_HEADER, create_mcp_context_token
from src.models.phase_transition import PhaseTransition
from src.models.session import Message, Session
from src.models.strategy import Strategy
from src.models.user import User, UserProfile
from src.services.openai_stream_service import ResponsesEventStreamer
from src.services.session_title_service import refresh_session_title
from src.util.chat_debug_trace import get_chat_debug_trace, record_chat_debug_trace
from src.util.logger import log_agent

_AGENT_UI_TAG = "AGENT_UI_JSON"
_AGENT_STATE_PATCH_TAG = "AGENT_STATE_PATCH"
_STRATEGY_CARD_GENUI_TYPE = "strategy_card"
_STRATEGY_REF_GENUI_TYPE = "strategy_ref"
_BACKTEST_CHARTS_GENUI_TYPE = "backtest_charts"
_STOP_CRITERIA_TURN_LIMIT = 10
_PHASE_CARRYOVER_TAG = "PHASE CARRYOVER MEMORY"
_PHASE_CARRYOVER_MAX_TURNS = 4
_PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE = 220
_PHASE_CARRYOVER_META_KEY = "phase_carryover_memory"
_OPENAI_STREAM_HARD_TIMEOUT_SECONDS = 300.0
_UUID_CANDIDATE_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_BACKTEST_COMPLETED_TOOL_NAMES: frozenset[str] = frozenset(
    {"backtest_get_job", "backtest_create_job"}
)
_BACKTEST_DEFAULT_CHARTS: tuple[str, ...] = (
    "equity_curve",
    "underwater_curve",
    "monthly_return_table",
    "holding_period_pnl_bins",
)
_STRATEGY_SCHEMA_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
)
_STRATEGY_ARTIFACT_OPS_TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_upsert_dsl",
    "strategy_get_dsl",
    "strategy_list_tunable_params",
    "strategy_patch_dsl",
    "strategy_list_versions",
    "strategy_get_version_dsl",
    "strategy_diff_versions",
    "strategy_rollback_dsl",
    "get_indicator_detail",
    "get_indicator_catalog",
)
_MARKET_DATA_MINIMAL_TOOL_NAMES: tuple[str, ...] = (
    "get_symbol_data_coverage",
)
_BACKTEST_BOOTSTRAP_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
)
_BACKTEST_FEEDBACK_TOOL_NAMES: tuple[str, ...] = (
    "backtest_create_job",
    "backtest_get_job",
    "backtest_entry_hour_pnl_heatmap",
    "backtest_entry_weekday_pnl",
    "backtest_monthly_return_table",
    "backtest_holding_period_pnl_bins",
    "backtest_long_short_breakdown",
    "backtest_exit_reason_breakdown",
    "backtest_underwater_curve",
    "backtest_rolling_metrics",
)
_MCP_CONTEXT_ENABLED_SERVER_LABELS: frozenset[str] = frozenset(
    {"strategy", "backtest", "market_data"}
)

# Singleton for KYC-specific helpers (profile loading from UserProfile)
_kyc_handler = KYCHandler()


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


class ChatOrchestrator:
    """Coordinates message flow and phase transitions.

    All AI communication goes through the OpenAI Responses API via
    :class:`ResponsesEventStreamer`.  The ``previous_response_id`` stored on
    each :class:`Session` gives the model full conversation history without
    us having to re-send messages manually.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        user_id: UUID,
        parent_session_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        profile = await self.db.scalar(
            select(UserProfile).where(UserProfile.user_id == user_id),
        )
        initial_phase = self._compute_initial_phase(profile)

        # Build phase-keyed artifacts from the registry
        artifacts = init_all_artifacts()

        # If user already has KYC data, seed the KYC artifact block
        if profile is not None:
            kyc_profile = _kyc_handler.build_profile_from_user_profile(profile)
            kyc_missing = _kyc_handler._compute_missing(kyc_profile)
            artifacts[Phase.KYC.value]["profile"] = kyc_profile
            artifacts[Phase.KYC.value]["missing_fields"] = kyc_missing

        session = Session(
            user_id=user_id,
            parent_session_id=parent_session_id,
            current_phase=initial_phase,
            status=SessionStatus.ACTIVE.value,
            artifacts=artifacts,
            metadata_=metadata or {},
            last_activity_at=datetime.now(UTC),
        )
        self.db.add(session)
        await self.db.flush()
        await refresh_session_title(db=self.db, session=session)
        return session

    # ------------------------------------------------------------------
    # /send-stream  (Responses API streaming – the single entry point)
    # ------------------------------------------------------------------

    async def handle_message_stream(
        self,
        user: User,
        payload: ChatSendRequest,
        streamer: ResponsesEventStreamer,
        *,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """Stream a single user->assistant turn via the Responses API.

        This is the **only** method the router calls.  It:
        1. Persists the user message.
        2. Looks up the phase handler for the current phase.
        3. Builds prompt pieces via the handler.
        4. Streams events from the Responses API while forwarding as SSE.
        5. Extracts phase state patches / GenUI payloads from the AI text.
        6. Delegates post-processing to the handler.
        7. Persists the assistant message and updates session/profile state.
        """
        session = await self._resolve_session(
            user_id=user.id, session_id=payload.session_id
        )
        await self._enforce_strategy_only_boundary(session=session)
        phase_before = session.current_phase
        user_runtime_policy = self._build_runtime_policy(payload)
        carryover_block = self._consume_phase_carryover_memory(
            session=session, phase=phase_before
        )
        prompt_user_message = (
            f"{carryover_block}{payload.message}"
            if isinstance(carryover_block, str) and carryover_block.strip()
            else payload.message
        )

        # -- persist user message ----------------------------------------
        self.db.add(
            Message(
                session_id=session.id,
                role="user",
                content=payload.message,
                phase=phase_before,
            )
        )
        phase_turn_count = self._increment_phase_turn_count(
            session=session,
            phase=phase_before,
        )

        yield self._sse(
            "stream",
            {
                "type": "stream_start",
                "session_id": str(session.id),
                "phase": phase_before,
            },
        )

        # -- resolve handler for current phase ---------------------------
        handler = get_handler(phase_before)
        if handler is None:
            # Terminal phases (completed, error) or unknown
            assistant_text = f"{phase_before} phase has no active handler."
            self.db.add(
                Message(
                    session_id=session.id,
                    role="assistant",
                    content=assistant_text,
                    phase=session.current_phase,
                )
            )
            session.last_activity_at = datetime.now(UTC)
            title_payload = await refresh_session_title(db=self.db, session=session)
            await self.db.commit()
            await self.db.refresh(session)

            yield self._sse("stream", {"type": "text_delta", "delta": assistant_text})
            yield self._sse(
                "stream",
                {
                    "type": "done",
                    "session_id": str(session.id),
                    "phase": session.current_phase,
                    "status": session.status,
                    "kyc_status": await self._fetch_kyc_status(user.id),
                    "missing_fields": [],
                    "session_title": title_payload.title,
                    "session_title_record": title_payload.record,
                },
            )
            log_agent(
                "orchestrator",
                f"session={session.id} phase={session.current_phase} (no handler)",
            )
            return

        preparation = await self._prepare_turn_context(
            session=session,
            user=user,
            payload=payload,
            language=language,
            phase_before=phase_before,
            user_runtime_policy=user_runtime_policy,
            prompt_user_message=prompt_user_message,
            phase_turn_count=phase_turn_count,
            handler=handler,
        )
        stream_state = _TurnStreamState()
        async for stream_event in self._stream_openai_and_collect(
            session=session,
            streamer=streamer,
            preparation=preparation,
            stream_state=stream_state,
        ):
            yield stream_event

        post_process_result = await self._post_process_turn(
            session=session,
            user=user,
            payload=payload,
            language=language,
            preparation=preparation,
            stream_state=stream_state,
        )
        async for tail_event in self._emit_tail_events_and_persist(
            session=session,
            preparation=preparation,
            stream_state=stream_state,
            post_process_result=post_process_result,
        ):
            yield tail_event

        log_agent("orchestrator", f"session={session.id} phase={session.current_phase}")

    async def _prepare_turn_context(
        self,
        *,
        session: Session,
        user: User,
        payload: ChatSendRequest,
        language: str,
        phase_before: str,
        user_runtime_policy: HandlerRuntimePolicy,
        prompt_user_message: str,
        phase_turn_count: int,
        handler: Any,
    ) -> _TurnPreparation:
        artifacts = copy.deepcopy(session.artifacts or {})
        artifacts = self._ensure_phase_keyed(artifacts)
        await self._hydrate_strategy_context(
            session=session,
            user_id=user.id,
            phase=phase_before,
            artifacts=artifacts,
            user_message=payload.message,
        )
        pre_strategy_instrument_before: str | None = None
        if phase_before == Phase.PRE_STRATEGY.value:
            pre_data = artifacts.get(Phase.PRE_STRATEGY.value)
            if isinstance(pre_data, dict):
                pre_profile = pre_data.get("profile")
                if isinstance(pre_profile, dict):
                    value = pre_profile.get("target_instrument")
                    if isinstance(value, str) and value.strip():
                        pre_strategy_instrument_before = value.strip()

        runtime_policy = self._resolve_runtime_policy(
            phase=phase_before,
            artifacts=artifacts,
            user_runtime_policy=user_runtime_policy,
        )
        ctx = PhaseContext(
            user_id=user.id,
            session_artifacts=artifacts,
            session_id=session.id,
            language=language,
            runtime_policy=runtime_policy,
        )
        prompt = handler.build_prompt(ctx, prompt_user_message)
        tools = self._merge_tools(
            base_tools=prompt.tools,
            runtime_policy=runtime_policy,
        )
        tools = self._attach_mcp_context_headers(
            tools=tools,
            user_id=user.id,
            session_id=session.id,
            phase=phase_before,
        )
        return _TurnPreparation(
            phase_before=phase_before,
            phase_turn_count=phase_turn_count,
            prompt_user_message=prompt_user_message,
            handler=handler,
            artifacts=artifacts,
            pre_strategy_instrument_before=pre_strategy_instrument_before,
            ctx=ctx,
            prompt=prompt,
            tools=tools,
        )

    async def _stream_openai_and_collect(
        self,
        *,
        session: Session,
        streamer: ResponsesEventStreamer,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
    ) -> AsyncIterator[str]:
        stream_request_kwargs: dict[str, Any] = {
            "model": preparation.prompt.model or settings.openai_response_model,
            "input_text": preparation.prompt.enriched_input,
            "instructions": preparation.prompt.instructions,
            "previous_response_id": session.previous_response_id,
            "tools": preparation.tools,
            "tool_choice": preparation.prompt.tool_choice,
            "reasoning": preparation.prompt.reasoning,
        }
        trace_stream_request_kwargs = self._redact_stream_request_kwargs_for_trace(
            stream_request_kwargs
        )
        record_chat_debug_trace(
            "orchestrator_to_openai_request",
            {
                "session_id": str(session.id),
                "phase": preparation.phase_before,
                **trace_stream_request_kwargs,
            },
        )
        try:
            async with asyncio.timeout(_OPENAI_STREAM_HARD_TIMEOUT_SECONDS):
                async for event in streamer.stream_events(**stream_request_kwargs):
                    record_chat_debug_trace(
                        "openai_to_orchestrator_event",
                        {
                            "session_id": str(session.id),
                            "phase": preparation.phase_before,
                            "event": event,
                        },
                    )
                    event_type = str(event.get("type", "unknown"))
                    seq = event.get("sequence_number")
                    stream_state.mcp_fallback_counter = self._collect_mcp_records_from_event(
                        event_type=event_type,
                        event=event,
                        sequence_number=seq,
                        records=stream_state.mcp_call_records,
                        order=stream_state.mcp_call_order,
                        fallback_counter=stream_state.mcp_fallback_counter,
                    )
                    yield self._sse(
                        "openai_event",
                        {
                            "type": "openai_event",
                            "openai_type": event_type,
                            "sequence_number": seq,
                            "payload": event,
                        },
                    )
                    if event_type == "response.stream_error":
                        err = event.get("error")
                        if isinstance(err, dict):
                            stream_state.stream_error_detail = copy.deepcopy(err)
                            msg = err.get("message")
                            if isinstance(msg, str) and msg.strip():
                                stream_state.stream_error_message = msg.strip()
                            if not stream_state.stream_error_message:
                                diagnostics = err.get("diagnostics")
                                if isinstance(diagnostics, dict):
                                    fallback_message = diagnostics.get("upstream_message")
                                    if (
                                        isinstance(fallback_message, str)
                                        and fallback_message.strip()
                                    ):
                                        stream_state.stream_error_message = (
                                            fallback_message.strip()
                                        )
                        if not stream_state.stream_error_message:
                            stream_state.stream_error_message = (
                                "Upstream stream interrupted."
                            )
                        continue
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            stream_state.full_text += delta
                            yield self._sse(
                                "stream",
                                {
                                    "type": "text_delta",
                                    "delta": delta,
                                    "sequence_number": seq,
                                },
                            )
                            stream_state.text_delta_emitted = True
                    elif event_type == "response.output_text.done":
                        done_text = event.get("text")
                        if (
                            isinstance(done_text, str)
                            and done_text
                            and not stream_state.full_text.strip()
                        ):
                            stream_state.full_text = done_text
                            yield self._sse(
                                "stream",
                                {
                                    "type": "text_delta",
                                    "delta": done_text,
                                    "sequence_number": seq,
                                },
                            )
                            stream_state.text_delta_emitted = True
                    if "mcp_" in event_type:
                        yield self._sse(
                            "stream",
                            {
                                "type": "mcp_event",
                                "openai_type": event_type,
                                "sequence_number": seq,
                                "payload": event,
                            },
                        )
                    if event_type == "response.completed":
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict):
                            usage = response_obj.get("usage")
                            if isinstance(usage, dict):
                                stream_state.completed_usage = usage
                            resp_id = response_obj.get("id")
                            if isinstance(resp_id, str) and resp_id:
                                session.previous_response_id = resp_id
        except TimeoutError:
            stream_state.stream_error_message = (
                "OpenAI stream timed out before completion. "
                f"(>{int(_OPENAI_STREAM_HARD_TIMEOUT_SECONDS)}s)"
            )
            stream_state.stream_error_detail = {
                "class": "TimeoutError",
                "message": stream_state.stream_error_message,
                "diagnostics": {"category": "orchestrator_stream_timeout"},
            }
        except Exception as exc:  # noqa: BLE001
            stream_state.stream_error_message = f"{type(exc).__name__}: {exc}"
            stream_state.stream_error_detail = {
                "class": type(exc).__name__,
                "message": str(exc),
                "diagnostics": {"category": "orchestrator_stream_runtime_error"},
            }

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
        cleaned_text, genui_payloads, raw_patches = self._extract_wrapped_payloads(
            stream_state.full_text
        )
        record_chat_debug_trace(
            "openai_to_orchestrator_full_text",
            {
                "session_id": str(session.id),
                "phase": preparation.phase_before,
                "full_text": stream_state.full_text,
                "cleaned_text": cleaned_text,
                "stream_error": stream_state.stream_error_message,
                "stream_error_detail": stream_state.stream_error_detail,
            },
        )
        assistant_text = cleaned_text.strip()
        if stream_state.stream_error_message and not assistant_text:
            assistant_text = (
                "The upstream AI stream was interrupted before completion. "
                f"Reason: {stream_state.stream_error_message}. Please retry this step."
            )

        final_mcp_tool_calls = self._build_persistable_mcp_tool_calls(
            records=stream_state.mcp_call_records,
            order=stream_state.mcp_call_order,
        )
        selected_genui_payloads = normalize_genui_payloads(
            genui_payloads,
            allow_passthrough_unregistered=True,
        )
        selected_genui_payloads = self._maybe_auto_wrap_strategy_genui(
            phase=preparation.phase_before,
            artifacts=preparation.artifacts,
            assistant_text=assistant_text,
            existing_genui=selected_genui_payloads,
            mcp_tool_calls=final_mcp_tool_calls,
        )
        selected_genui_payloads = self._maybe_auto_wrap_backtest_charts_genui(
            phase=preparation.phase_before,
            existing_genui=selected_genui_payloads,
            mcp_tool_calls=final_mcp_tool_calls,
        )

        result = await preparation.handler.post_process(
            preparation.ctx, raw_patches, self.db
        )
        session.artifacts = result.artifacts
        if result.completed:
            selected_genui_payloads = []

        post_process_ctx = PhaseContext(
            user_id=preparation.ctx.user_id,
            session_artifacts=result.artifacts,
            session_id=preparation.ctx.session_id,
            language=preparation.ctx.language,
            runtime_policy=preparation.ctx.runtime_policy,
        )
        filtered_genui_payloads: list[dict[str, Any]] = []
        for genui_payload in selected_genui_payloads:
            filtered = preparation.handler.filter_genui(genui_payload, post_process_ctx)
            if filtered is not None:
                filtered_genui_payloads.append(filtered)
        filtered_genui_payloads = self._ensure_required_choice_prompt_payload(
            handler=preparation.handler,
            ctx=post_process_ctx,
            missing_fields=result.missing_fields,
            genui_payloads=filtered_genui_payloads,
        )
        filtered_genui_payloads = self._ensure_pre_strategy_chart_payload(
            phase_before=preparation.phase_before,
            artifacts=result.artifacts,
            genui_payloads=filtered_genui_payloads,
            instrument_before=preparation.pre_strategy_instrument_before,
        )
        persisted_tool_calls = [*filtered_genui_payloads, *final_mcp_tool_calls]

        transitioned = False
        if result.completed and result.next_phase:
            await self._transition_phase(
                session=session,
                to_phase=result.next_phase,
                trigger="ai_output",
                metadata={"reason": result.transition_reason or "phase_completed"},
            )
            transitioned = True

        assistant_text = self._resolve_assistant_text_fallbacks(
            assistant_text=assistant_text,
            filtered_genui_payloads=filtered_genui_payloads,
            full_text=stream_state.full_text,
        )

        kyc_status = result.phase_status.get("kyc_status")
        if kyc_status is None:
            kyc_status = await self._fetch_kyc_status(user.id)

        assistant_text, stop_criteria_delta = self._maybe_apply_stop_criteria_placeholder(
            session=session,
            phase=preparation.phase_before,
            phase_turn_count=preparation.phase_turn_count,
            language=language,
            assistant_text=assistant_text,
        )
        if not assistant_text.strip():
            assistant_text = self._build_empty_turn_fallback_text(
                phase=preparation.phase_before,
                missing_fields=result.missing_fields,
                language=language,
            )
            log_agent(
                "orchestrator",
                (
                    f"session={session.id} phase={preparation.phase_before} "
                    "empty_model_output_fallback"
                ),
            )
        if transitioned:
            await self._store_phase_carryover_memory(
                session=session,
                from_phase=preparation.phase_before,
                to_phase=session.current_phase,
                user_message=payload.message,
                assistant_message=assistant_text,
            )

        return _TurnPostProcessResult(
            assistant_text=assistant_text,
            cleaned_text=cleaned_text,
            filtered_genui_payloads=filtered_genui_payloads,
            persisted_tool_calls=persisted_tool_calls,
            missing_fields=result.missing_fields,
            kyc_status=kyc_status,
            transitioned=transitioned,
            stop_criteria_delta=stop_criteria_delta,
        )

    def _resolve_assistant_text_fallbacks(
        self,
        *,
        assistant_text: str,
        filtered_genui_payloads: list[dict[str, Any]],
        full_text: str,
    ) -> str:
        if assistant_text:
            return assistant_text
        if not filtered_genui_payloads or not full_text.strip():
            return assistant_text

        fallback_choice = next(
            (
                payload
                for payload in reversed(filtered_genui_payloads)
                if payload.get("type") == "choice_prompt"
            ),
            None,
        )
        question = fallback_choice.get("question") if fallback_choice is not None else None
        if isinstance(question, str) and question.strip():
            return question.strip()
        return assistant_text

    async def _emit_tail_events_and_persist(
        self,
        *,
        session: Session,
        preparation: _TurnPreparation,
        stream_state: _TurnStreamState,
        post_process_result: _TurnPostProcessResult,
    ) -> AsyncIterator[str]:
        self.db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content=post_process_result.assistant_text,
                phase=preparation.phase_before,
                response_id=session.previous_response_id,
                tool_calls=post_process_result.persisted_tool_calls or None,
                token_usage=stream_state.completed_usage or None,
            )
        )
        session.last_activity_at = datetime.now(UTC)
        title_payload = await refresh_session_title(db=self.db, session=session)
        await self.db.commit()
        await self.db.refresh(session)

        for genui_payload in post_process_result.filtered_genui_payloads:
            yield self._sse("stream", {"type": "genui", "payload": genui_payload})

        if post_process_result.transitioned:
            yield self._sse(
                "stream",
                {
                    "type": "phase_change",
                    "from_phase": preparation.phase_before,
                    "to_phase": session.current_phase,
                },
            )

        stop_criteria_delta = post_process_result.stop_criteria_delta
        if isinstance(stop_criteria_delta, str) and stop_criteria_delta.strip():
            if stream_state.text_delta_emitted and post_process_result.cleaned_text.strip():
                yield self._sse(
                    "stream",
                    {
                        "type": "text_delta",
                        "delta": stop_criteria_delta,
                    },
                )
            else:
                stop_criteria_delta = None

        should_emit_terminal_text = (
            post_process_result.assistant_text.strip()
            and (
                not stream_state.text_delta_emitted
                or not post_process_result.cleaned_text.strip()
            )
        )
        if should_emit_terminal_text:
            yield self._sse(
                "stream",
                {
                    "type": "text_delta",
                    "delta": post_process_result.assistant_text,
                },
            )
            stream_state.text_delta_emitted = True

        yield self._sse(
            "stream",
            {
                "type": "done",
                "session_id": str(session.id),
                "phase": session.current_phase,
                "status": session.status,
                "kyc_status": post_process_result.kyc_status,
                "missing_fields": post_process_result.missing_fields,
                "session_title": title_payload.title,
                "session_title_record": title_payload.record,
                "usage": stream_state.completed_usage,
                "stream_error": stream_state.stream_error_message,
                "stream_error_detail": stream_state.stream_error_detail,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers – session / DB
    # ------------------------------------------------------------------

    async def _resolve_session(
        self, *, user_id: UUID, session_id: UUID | None
    ) -> Session:
        if session_id is None:
            return await self.create_session(user_id=user_id)

        stmt = (
            select(Session)
            .where(Session.id == session_id, Session.user_id == user_id)
            .options(selectinload(Session.messages))
        )
        session = await self.db.scalar(stmt)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found."
            )
        if session.archived_at is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Session is archived. Unarchive it before sending new messages.",
            )
        return session

    async def _fetch_kyc_status(self, user_id: UUID) -> str:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        profile = await self.db.scalar(stmt)
        if profile is None:
            return "incomplete"
        return profile.kyc_status

    async def _transition_phase(
        self,
        *,
        session: Session,
        to_phase: str,
        trigger: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from_phase = session.current_phase
        if from_phase == to_phase:
            return
        if not can_transition(from_phase, to_phase):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Invalid phase transition: {from_phase} -> {to_phase}",
            )

        session.current_phase = to_phase
        # Reset Responses API conversation chain on phase boundary to bound
        # context growth and keep phase prompts/tools isolated. A concise
        # carryover text block will be attached to the next phase turn.
        session.previous_response_id = None
        next_meta = dict(session.metadata_ or {})
        next_meta.update(metadata or {})
        next_meta["phase_transition_at"] = datetime.now(UTC).isoformat()
        session.metadata_ = next_meta

        self.db.add(
            PhaseTransition(
                session_id=session.id,
                from_phase=from_phase,
                to_phase=to_phase,
                trigger=trigger,
                metadata_=metadata or {},
            )
        )

    # ------------------------------------------------------------------
    # Private helpers – initial phase computation
    # ------------------------------------------------------------------

    def _compute_initial_phase(self, profile: UserProfile | None) -> str:
        """Determine which phase to start a new session in.

        KYC only needs to be done once per user; completed-KYC users
        always start from PRE_STRATEGY.
        """
        if _kyc_handler.is_profile_complete(profile):
            return Phase.PRE_STRATEGY.value
        return Phase.KYC.value

    # ------------------------------------------------------------------
    # Private helpers – artifact migration / legacy compat
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_phase_keyed(artifacts: dict[str, Any]) -> dict[str, Any]:
        """Ensure session artifacts are phase-keyed and initialized."""
        normalized = dict(artifacts)
        for phase in WORKFLOW_PHASES:
            phase_block = normalized.get(phase)
            if isinstance(phase_block, dict) and "profile" in phase_block:
                continue
            handler = get_handler(phase)
            if handler:
                normalized[phase] = handler.init_artifacts()
        return normalized

    # ------------------------------------------------------------------
    # Private helpers – AI output parsing
    # ------------------------------------------------------------------

    def _extract_wrapped_payloads(
        self,
        text: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse AGENT_UI_JSON and AGENT_STATE_PATCH blocks from AI text.

        Returns (cleaned_text, genui_payloads, raw_patch_payloads).
        """
        genui_payloads = self._extract_json_by_tag(text, _AGENT_UI_TAG)
        patch_payloads = self._extract_json_by_tag(text, _AGENT_STATE_PATCH_TAG)

        cleaned = self._strip_tag_blocks(text, _AGENT_UI_TAG)
        cleaned = self._strip_tag_blocks(cleaned, _AGENT_STATE_PATCH_TAG)
        cleaned = self._strip_session_state_echo(cleaned)
        cleaned = self._strip_mcp_pseudo_tool_tags(cleaned)

        return cleaned, genui_payloads, patch_payloads

    def _maybe_auto_wrap_strategy_genui(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        assistant_text: str,
        existing_genui: list[dict[str, Any]],
        mcp_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if phase != Phase.STRATEGY.value:
            return existing_genui
        if self._contains_strategy_genui(existing_genui):
            return existing_genui

        strategy_profile = self._extract_phase_profile(
            artifacts=artifacts,
            phase=Phase.STRATEGY.value,
        )
        strategy_id = strategy_profile.get("strategy_id")
        if isinstance(strategy_id, str) and strategy_id.strip():
            return existing_genui

        strategy_draft_id = self._extract_strategy_draft_id_from_mcp_calls(
            mcp_tool_calls
        )
        if strategy_draft_id is not None:
            wrapped_ref = {
                "type": _STRATEGY_REF_GENUI_TYPE,
                "strategy_draft_id": strategy_draft_id,
                "source": "strategy_validate_dsl",
                "display_mode": "draft",
            }
            return self._append_genui_if_new(
                existing_genui=existing_genui, candidate=wrapped_ref
            )

        dsl_payload = self._try_extract_strategy_dsl_from_text(assistant_text)
        if not isinstance(dsl_payload, dict):
            return existing_genui
        wrapped = {
            "type": _STRATEGY_CARD_GENUI_TYPE,
            "dsl_json": dsl_payload,
            "source": "auto_detected_first_dsl",
            "display_mode": "draft",
        }
        return self._append_genui_if_new(
            existing_genui=existing_genui, candidate=wrapped
        )

    @staticmethod
    def _append_genui_if_new(
        *,
        existing_genui: list[dict[str, Any]],
        candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        dedupe_key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        existing_keys = {
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in existing_genui
        }
        if dedupe_key in existing_keys:
            return existing_genui
        return [*existing_genui, candidate]

    def _extract_strategy_draft_id_from_mcp_calls(
        self,
        mcp_tool_calls: list[dict[str, Any]],
    ) -> str | None:
        for item in reversed(mcp_tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue
            if str(item.get("name", "")).strip() != "strategy_validate_dsl":
                continue
            if str(item.get("status", "")).strip().lower() != "success":
                continue

            output_payload = self._coerce_json_object(item.get("output"))
            if not isinstance(output_payload, dict):
                continue
            if output_payload.get("ok") is not True:
                continue
            draft_id = self._coerce_uuid_text(output_payload.get("strategy_draft_id"))
            if draft_id is None:
                data_payload = self._coerce_json_object(output_payload.get("data"))
                if isinstance(data_payload, dict):
                    draft_id = self._coerce_uuid_text(
                        data_payload.get("strategy_draft_id")
                    )
            if draft_id is None:
                continue
            return draft_id
        return None

    @staticmethod
    def _coerce_json_object(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _contains_strategy_genui(payloads: list[dict[str, Any]]) -> bool:
        for payload in payloads:
            payload_type = payload.get("type")
            if not isinstance(payload_type, str):
                continue
            normalized = payload_type.strip().lower()
            if normalized in {_STRATEGY_CARD_GENUI_TYPE, _STRATEGY_REF_GENUI_TYPE}:
                return True
        return False

    @staticmethod
    def _contains_backtest_charts_genui(payloads: list[dict[str, Any]]) -> bool:
        for payload in payloads:
            payload_type = payload.get("type")
            if not isinstance(payload_type, str):
                continue
            if payload_type.strip().lower() == _BACKTEST_CHARTS_GENUI_TYPE:
                return True
        return False

    def _maybe_auto_wrap_backtest_charts_genui(
        self,
        *,
        phase: str,
        existing_genui: list[dict[str, Any]],
        mcp_tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if phase != Phase.STRATEGY.value:
            return existing_genui
        if self._contains_backtest_charts_genui(existing_genui):
            return existing_genui

        resolved = self._extract_backtest_done_from_mcp_calls(mcp_tool_calls)
        if resolved is None:
            return existing_genui

        job_id, strategy_id, source = resolved
        payload: dict[str, Any] = {
            "type": _BACKTEST_CHARTS_GENUI_TYPE,
            "job_id": job_id,
            "charts": list(_BACKTEST_DEFAULT_CHARTS),
            "sampling": "eod",
            "max_points": 365,
            "source": source,
        }
        if strategy_id is not None:
            payload["strategy_id"] = strategy_id
        return self._append_genui_if_new(
            existing_genui=existing_genui,
            candidate=payload,
        )

    def _extract_backtest_done_from_mcp_calls(
        self,
        mcp_tool_calls: list[dict[str, Any]],
    ) -> tuple[str, str | None, str] | None:
        for item in reversed(mcp_tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue

            name = str(item.get("name", "")).strip()
            if name not in _BACKTEST_COMPLETED_TOOL_NAMES:
                continue
            if str(item.get("status", "")).strip().lower() != "success":
                continue

            output_payload = self._coerce_json_object(item.get("output"))
            if not isinstance(output_payload, dict):
                continue
            if output_payload.get("ok") is not True:
                continue
            if str(output_payload.get("status", "")).strip().lower() != "done":
                continue

            job_id = self._coerce_uuid_text(output_payload.get("job_id"))
            if job_id is None:
                continue
            strategy_id = self._coerce_uuid_text(output_payload.get("strategy_id"))
            return job_id, strategy_id, name
        return None

    def _try_extract_strategy_dsl_from_text(self, text: str) -> dict[str, Any] | None:
        if not isinstance(text, str):
            return None
        body = text.strip()
        if not body:
            return None

        candidates = self._extract_json_candidates_from_text(body)
        for raw_candidate in candidates:
            try:
                parsed = json.loads(raw_candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if not self._looks_like_strategy_dsl(parsed):
                continue

            validation = validate_strategy_payload(parsed)
            if validation.is_valid:
                return parsed
        return None

    @staticmethod
    def _looks_like_strategy_dsl(payload: dict[str, Any]) -> bool:
        required_keys = ("strategy", "universe", "timeframe", "trade")
        for key in required_keys:
            if key not in payload:
                return False
        return isinstance(payload.get("strategy"), dict) and isinstance(
            payload.get("trade"), dict
        )

    @staticmethod
    def _extract_json_candidates_from_text(text: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)```",
            text,
            flags=re.IGNORECASE,
        ):
            block = match.group(1)
            if isinstance(block, str):
                _add(block)

        depth = 0
        in_string = False
        escaped = False
        start_index: int | None = None
        max_candidates = 40

        for idx, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if ch == "{":
                if depth == 0:
                    start_index = idx
                depth += 1
                continue

            if ch == "}":
                if depth <= 0:
                    continue
                depth -= 1
                if depth == 0 and start_index is not None:
                    _add(text[start_index : idx + 1])
                    if len(candidates) >= max_candidates:
                        break
                    start_index = None

        return candidates

    def _extract_json_by_tag(self, text: str, tag: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        pattern = self._build_tag_pattern(tag)
        for matched in pattern.findall(text):
            try:
                data = json.loads(matched.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                payloads.append(data)
        return payloads

    def _strip_tag_blocks(self, text: str, tag: str) -> str:
        return self._build_tag_pattern(tag).sub("", text)

    def _strip_session_state_echo(self, text: str) -> str:
        # Remove full [SESSION STATE] ... [/SESSION STATE] echoes.
        output = re.sub(
            r"\[\s*SESSION\s+STATE\s*\][\s\S]*?\[\s*/\s*SESSION\s+STATE\s*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Remove truncated echoes like a lone "[SESSION STATE]" line and attached
        # state bullets (when model starts echoing but stops mid-block).
        output = re.sub(
            r"\[\s*SESSION\s+STATE\s*\]\s*\n(?:\s*-\s*[^\n]*\n)*",
            "",
            output,
            flags=re.IGNORECASE,
        )
        # Remove any remaining standalone marker lines.
        output = re.sub(
            r"(?im)^\s*\[\s*/?\s*SESSION\s+STATE\s*\]\s*$\n?",
            "",
            output,
        )
        return output

    def _strip_mcp_pseudo_tool_tags(self, text: str) -> str:
        """Strip hallucinated pseudo-tool XML tags like <mcp_xxx>{...}</mcp_xxx>."""
        output = re.sub(
            r"<\s*(mcp_[a-z0-9_:\-]+)\s*>[\s\S]*?<\s*/\s*\1\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )

        open_matches = list(
            re.finditer(r"<\s*mcp_[a-z0-9_:\-]+\s*>", output, flags=re.IGNORECASE)
        )
        if open_matches:
            open_idx = open_matches[-1].start()
            tail = output[open_idx:]
            has_close = re.search(
                r"<\s*/\s*mcp_[a-z0-9_:\-]+\s*>",
                tail,
                flags=re.IGNORECASE,
            )
            if has_close is None:
                output = output[:open_idx]

        output = re.sub(
            r"</?\s*mcp_[a-z0-9_:\-]+\s*>",
            "",
            output,
            flags=re.IGNORECASE,
        )
        return output

    @staticmethod
    def _build_tag_pattern(tag: str) -> re.Pattern[str]:
        escaped_tag = re.escape(tag)
        return re.compile(
            rf"<\s*{escaped_tag}\s*>(.*?)<\s*/\s*{escaped_tag}\s*>",
            flags=re.DOTALL | re.IGNORECASE,
        )

    def _ensure_required_choice_prompt_payload(
        self,
        *,
        handler: Any,
        ctx: PhaseContext,
        missing_fields: list[str],
        genui_payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not missing_fields:
            return genui_payloads
        if any(payload.get("type") == "choice_prompt" for payload in genui_payloads):
            return genui_payloads

        fallback_builder = getattr(handler, "build_fallback_choice_prompt", None)
        if not callable(fallback_builder):
            return genui_payloads

        try:
            candidate = fallback_builder(
                missing_fields=list(missing_fields),
                ctx=ctx,
            )
        except Exception as exc:  # noqa: BLE001
            phase_name = getattr(handler, "phase_name", "unknown")
            log_agent(
                "orchestrator",
                f"session={ctx.session_id} phase={phase_name} "
                f"fallback_choice_prompt_error={type(exc).__name__}",
            )
            return genui_payloads

        if not isinstance(candidate, dict):
            return genui_payloads

        normalized = normalize_genui_payloads(
            [candidate],
            allow_passthrough_unregistered=True,
        )
        if not normalized:
            return genui_payloads

        filtered = handler.filter_genui(normalized[0], ctx)
        if not isinstance(filtered, dict):
            return genui_payloads

        phase_name = getattr(handler, "phase_name", "unknown")
        choice_id = filtered.get("choice_id")
        log_agent(
            "orchestrator",
            f"session={ctx.session_id} phase={phase_name} "
            f"injected_fallback_choice_prompt choice_id={choice_id}",
        )
        return [*genui_payloads, filtered]

    def _ensure_pre_strategy_chart_payload(
        self,
        *,
        phase_before: str,
        artifacts: dict[str, Any],
        genui_payloads: list[dict[str, Any]],
        instrument_before: str | None,
    ) -> list[dict[str, Any]]:
        if phase_before != Phase.PRE_STRATEGY.value:
            return genui_payloads
        if any(
            payload.get("type") == "tradingview_chart" for payload in genui_payloads
        ):
            return genui_payloads
        if not any(
            payload.get("type") == "choice_prompt" for payload in genui_payloads
        ):
            return genui_payloads

        pre_strategy_data = artifacts.get(Phase.PRE_STRATEGY.value)
        if not isinstance(pre_strategy_data, dict):
            return genui_payloads
        profile = pre_strategy_data.get("profile")
        if not isinstance(profile, dict):
            return genui_payloads
        instrument = profile.get("target_instrument")
        if not isinstance(instrument, str) or not instrument.strip():
            return genui_payloads
        instrument = instrument.strip()
        if instrument_before == instrument:
            return genui_payloads
        market = profile.get("target_market")
        if not isinstance(market, str) or not market.strip():
            return genui_payloads

        symbol = get_tradingview_symbol_for_market_instrument(
            market=market,
            instrument=instrument,
        )
        if not isinstance(symbol, str) or not symbol.strip():
            return genui_payloads

        chart_payload = {
            "type": "tradingview_chart",
            "symbol": symbol,
            "interval": "D",
        }
        return [chart_payload, *genui_payloads]

    def _collect_mcp_records_from_event(
        self,
        *,
        event_type: str,
        event: dict[str, Any],
        sequence_number: Any,
        records: dict[str, dict[str, Any]],
        order: list[str],
        fallback_counter: int,
    ) -> int:
        candidates: list[dict[str, Any]] = []

        if event_type in {"response.output_item.added", "response.output_item.done"}:
            raw_item = event.get("item")
            if isinstance(raw_item, dict):
                candidates.append(raw_item)

        if event_type == "response.completed":
            response_obj = event.get("response")
            if isinstance(response_obj, dict):
                output_items = response_obj.get("output")
                if isinstance(output_items, list):
                    for output_item in output_items:
                        if isinstance(output_item, dict):
                            candidates.append(output_item)

        if "mcp_call" in event_type:
            candidates.append(event)

        for candidate in candidates:
            fallback_counter = self._upsert_mcp_record(
                candidate=candidate,
                event_type=event_type,
                sequence_number=sequence_number,
                records=records,
                order=order,
                fallback_counter=fallback_counter,
            )
        return fallback_counter

    def _upsert_mcp_record(
        self,
        *,
        candidate: dict[str, Any],
        event_type: str,
        sequence_number: Any,
        records: dict[str, dict[str, Any]],
        order: list[str],
        fallback_counter: int,
    ) -> int:
        event_type_norm = event_type.strip().lower()
        payload_type_raw = candidate.get("type")
        payload_type = (
            payload_type_raw.strip().lower()
            if isinstance(payload_type_raw, str)
            else ""
        )

        if payload_type == "mcp_list_tools" or "mcp_list_tools" in event_type_norm:
            return fallback_counter

        is_payload_mcp_call = payload_type == "mcp_call"
        is_mcp_call_event = "mcp_call" in event_type_norm
        if not is_payload_mcp_call and not is_mcp_call_event:
            return fallback_counter

        call_id, fallback_counter = self._resolve_mcp_record_id(
            candidate=candidate,
            sequence_number=sequence_number,
            fallback_counter=fallback_counter,
        )

        existing = records.get(call_id, {})
        next_status = self._normalize_mcp_status(existing.get("status")) or "running"
        status_from_payload = self._normalize_mcp_status(candidate.get("status"))
        status_from_event = self._status_from_mcp_event_type(event_type_norm)
        merged_status = status_from_payload or status_from_event
        error_text = self._normalize_mcp_error(candidate.get("error"))
        if error_text:
            next_status = "failure"
        elif merged_status == "failure":
            next_status = "failure"
        elif merged_status in {"running", "success"} and next_status != "failure":
            next_status = merged_status

        arguments = candidate.get("arguments")
        if arguments is None:
            arguments = candidate.get("args")
        next_arguments = (
            arguments if arguments is not None else existing.get("arguments")
        )

        output = candidate.get("output")
        if output is None:
            output = candidate.get("result")
        if output is None:
            output = candidate.get("response")
        next_output = output if output is not None else existing.get("output")

        tool_name = candidate.get("name")
        if isinstance(tool_name, str):
            cleaned_name = tool_name.strip()
        else:
            cleaned_name = ""
        if not cleaned_name:
            existing_name = existing.get("name")
            cleaned_name = (
                existing_name.strip()
                if isinstance(existing_name, str) and existing_name.strip()
                else "mcp_call"
            )

        next_error = error_text or self._normalize_mcp_error(existing.get("error"))

        next_record: dict[str, Any] = {
            "type": "mcp_call",
            "id": call_id,
            "call_id": call_id,
            "name": cleaned_name,
            "status": next_status,
        }
        if next_arguments is not None:
            next_record["arguments"] = next_arguments
        if next_output is not None:
            next_record["output"] = next_output
        if next_error:
            next_record["error"] = next_error
            next_record["status"] = "failure"

        records[call_id] = next_record
        if call_id not in order:
            order.append(call_id)
        return fallback_counter

    @staticmethod
    def _resolve_mcp_record_id(
        *,
        candidate: dict[str, Any],
        sequence_number: Any,
        fallback_counter: int,
    ) -> tuple[str, int]:
        for key in ("call_id", "item_id", "output_item_id", "id"):
            raw = candidate.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text, fallback_counter

        output_index = candidate.get("output_index")
        if output_index is not None:
            output_text = str(output_index).strip()
            if output_text:
                return f"output_{output_text}", fallback_counter

        if sequence_number is not None:
            seq_text = str(sequence_number).strip()
            if seq_text:
                return f"seq_{seq_text}", fallback_counter

        next_counter = fallback_counter + 1
        return f"mcp_call_{next_counter}", next_counter

    @staticmethod
    def _normalize_mcp_status(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized in {"in_progress", "running"}:
            return "running"
        if normalized in {"completed", "success"}:
            return "success"
        if normalized in {"failed", "error", "failure"}:
            return "failure"
        return None

    @staticmethod
    def _status_from_mcp_event_type(event_type: str) -> str | None:
        normalized = event_type.strip().lower()
        if normalized.endswith(".in_progress"):
            return "running"
        if normalized.endswith(".completed"):
            return "success"
        if normalized.endswith(".failed"):
            return "failure"
        return None

    @staticmethod
    def _normalize_mcp_error(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
        text = text.strip()
        return text or None

    def _build_persistable_mcp_tool_calls(
        self,
        *,
        records: dict[str, dict[str, Any]],
        order: list[str],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for call_id in order:
            record = records.get(call_id)
            if not isinstance(record, dict):
                continue

            status = self._normalize_mcp_status(record.get("status"))
            error = self._normalize_mcp_error(record.get("error"))
            if error:
                status = "failure"

            if status not in {"success", "failure"}:
                continue

            name_raw = record.get("name")
            name = (
                str(name_raw).strip()
                if name_raw is not None and str(name_raw).strip()
                else "mcp_call"
            )
            item: dict[str, Any] = {
                "type": "mcp_call",
                "id": call_id,
                "call_id": call_id,
                "name": name,
                "status": status,
            }
            if "arguments" in record:
                item["arguments"] = record["arguments"]
            if "output" in record:
                item["output"] = record["output"]
            if error:
                item["error"] = error
            output.append(item)

        # Suppress transient retry failures when the same tool+arguments
        # succeeds later in the same assistant turn.
        success_signatures: set[tuple[str, str]] = set()
        success_names: set[str] = set()
        for item in output:
            if item.get("status") != "success":
                continue
            name = str(item.get("name", "")).strip()
            success_names.add(name)
            success_signatures.add(
                (
                    name,
                    self._stable_json_signature(item.get("arguments")),
                )
            )

        filtered: list[dict[str, Any]] = []
        for item in output:
            status = str(item.get("status", "")).strip().lower()
            signature = (
                str(item.get("name", "")).strip(),
                self._stable_json_signature(item.get("arguments")),
            )
            if status == "failure" and (
                signature in success_signatures or signature[0] in success_names
            ):
                continue
            filtered.append(item)
        return filtered

    @staticmethod
    def _stable_json_signature(value: Any) -> str:
        if value is None:
            return ""
        try:
            return json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except TypeError:
            return str(value)

    # ------------------------------------------------------------------
    # Private helpers – misc
    # ------------------------------------------------------------------

    async def _enforce_strategy_only_boundary(self, *, session: Session) -> None:
        """Redirect legacy stress-test sessions back into strategy phase.

        Product boundary today keeps performance iteration inside strategy.
        If an old session is still parked in ``stress_test``, we migrate it
        before prompt/tool selection so the model never executes stress phase
        instructions.
        """
        if session.current_phase != Phase.STRESS_TEST.value:
            return

        artifacts = self._ensure_phase_keyed(copy.deepcopy(session.artifacts or {}))
        strategy_block = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": ["strategy_id"]},
        )
        stress_block = artifacts.setdefault(
            Phase.STRESS_TEST.value,
            {
                "profile": {},
                "missing_fields": ["strategy_id", "backtest_job_id", "backtest_status"],
            },
        )

        strategy_profile_raw = strategy_block.get("profile")
        strategy_profile = (
            dict(strategy_profile_raw) if isinstance(strategy_profile_raw, dict) else {}
        )
        stress_profile_raw = stress_block.get("profile")
        stress_profile = (
            dict(stress_profile_raw) if isinstance(stress_profile_raw, dict) else {}
        )

        resolved_strategy_id = self._coerce_uuid_text(
            strategy_profile.get("strategy_id")
        )
        if resolved_strategy_id is None:
            resolved_strategy_id = self._coerce_uuid_text(
                stress_profile.get("strategy_id")
            )
        if resolved_strategy_id is not None:
            strategy_profile["strategy_id"] = resolved_strategy_id
            raw_missing = strategy_block.get("missing_fields")
            if isinstance(raw_missing, list):
                strategy_block["missing_fields"] = [
                    normalized
                    for item in raw_missing
                    if (normalized := str(item).strip()) and normalized != "strategy_id"
                ]
            else:
                strategy_block["missing_fields"] = []

        strategy_block["profile"] = strategy_profile
        stress_block["profile"] = stress_profile
        session.artifacts = artifacts

        await self._transition_phase(
            session=session,
            to_phase=Phase.STRATEGY.value,
            trigger="system",
            metadata={"reason": "stress_test_disabled_redirect_to_strategy"},
        )

    async def _hydrate_strategy_context(
        self,
        *,
        session: Session,
        user_id: UUID,
        phase: str,
        artifacts: dict[str, Any],
        user_message: str,
    ) -> None:
        if phase not in {Phase.STRATEGY.value, Phase.STRESS_TEST.value}:
            return

        strategy_block = artifacts.setdefault(
            Phase.STRATEGY.value,
            {"profile": {}, "missing_fields": ["strategy_id"]},
        )
        strategy_profile_raw = strategy_block.get("profile")
        strategy_profile = (
            dict(strategy_profile_raw) if isinstance(strategy_profile_raw, dict) else {}
        )

        stress_block = artifacts.setdefault(
            Phase.STRESS_TEST.value,
            {
                "profile": {},
                "missing_fields": ["strategy_id", "backtest_job_id", "backtest_status"],
            },
        )
        stress_profile_raw = stress_block.get("profile")
        stress_profile = (
            dict(stress_profile_raw) if isinstance(stress_profile_raw, dict) else {}
        )

        resolved_strategy_id = self._coerce_uuid_text(
            strategy_profile.get("strategy_id")
        )
        if resolved_strategy_id is None:
            resolved_strategy_id = self._coerce_uuid_text(
                stress_profile.get("strategy_id")
            )
        if resolved_strategy_id is None:
            metadata = dict(session.metadata_ or {})
            resolved_strategy_id = self._coerce_uuid_text(metadata.get("strategy_id"))

        if resolved_strategy_id is None:
            for candidate in self._extract_uuid_candidates(text=user_message):
                if await self._strategy_belongs_to_user(
                    user_id=user_id, strategy_id=candidate
                ):
                    resolved_strategy_id = candidate
                    break
        if resolved_strategy_id is None:
            resolved_strategy_id = await self._resolve_latest_session_strategy_id(
                user_id=user_id,
                session_id=session.id,
            )

        strategy_block["profile"] = strategy_profile
        stress_block["profile"] = stress_profile
        if resolved_strategy_id is None:
            return

        strategy_profile["strategy_id"] = resolved_strategy_id
        if "strategy_id" not in stress_profile:
            stress_profile["strategy_id"] = resolved_strategy_id

        raw_strategy_missing = strategy_block.get("missing_fields")
        if isinstance(raw_strategy_missing, list):
            strategy_block["missing_fields"] = [
                normalized
                for item in raw_strategy_missing
                if (normalized := str(item).strip()) and normalized != "strategy_id"
            ]
        else:
            strategy_block["missing_fields"] = []

        raw_stress_missing = stress_block.get("missing_fields")
        if isinstance(raw_stress_missing, list):
            stress_block["missing_fields"] = [
                normalized
                for item in raw_stress_missing
                if (normalized := str(item).strip()) and normalized != "strategy_id"
            ]

    async def _strategy_belongs_to_user(
        self,
        *,
        user_id: UUID,
        strategy_id: str,
    ) -> bool:
        try:
            strategy_uuid = UUID(strategy_id)
        except ValueError:
            return False

        owned = await self.db.scalar(
            select(Strategy.id).where(
                Strategy.id == strategy_uuid,
                Strategy.user_id == user_id,
            )
        )
        return owned is not None

    async def _resolve_latest_session_strategy_id(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
    ) -> str | None:
        strategy_uuid = await self.db.scalar(
            select(Strategy.id)
            .where(
                Strategy.user_id == user_id,
                Strategy.session_id == session_id,
            )
            .order_by(
                Strategy.updated_at.desc(),
                Strategy.created_at.desc(),
            )
            .limit(1)
        )
        if strategy_uuid is None:
            return None
        return self._coerce_uuid_text(str(strategy_uuid))

    @staticmethod
    def _coerce_uuid_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return str(UUID(text))
        except ValueError:
            return None

    @staticmethod
    def _extract_uuid_candidates(*, text: str) -> list[str]:
        if not isinstance(text, str):
            return []

        output: list[str] = []
        seen: set[str] = set()
        for candidate in _UUID_CANDIDATE_PATTERN.findall(text):
            normalized = ChatOrchestrator._coerce_uuid_text(candidate)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    def _resolve_runtime_policy(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        user_runtime_policy: HandlerRuntimePolicy,
    ) -> HandlerRuntimePolicy:
        phase_policy = self._build_phase_runtime_policy(
            phase=phase, artifacts=artifacts
        )
        return self._merge_runtime_policies(
            phase_policy=phase_policy,
            user_policy=user_runtime_policy,
        )

    def _build_phase_runtime_policy(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        if phase == Phase.STRATEGY.value:
            return self._build_strategy_runtime_policy(artifacts=artifacts)
        if phase == Phase.STRESS_TEST.value:
            return self._build_stress_test_runtime_policy(artifacts=artifacts)
        return HandlerRuntimePolicy()

    def _build_strategy_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(
            artifacts=artifacts, phase=Phase.STRATEGY.value
        )
        strategy_id = self._coerce_uuid_text(profile.get("strategy_id"))
        if strategy_id is None:
            stress_profile = self._extract_phase_profile(
                artifacts=artifacts,
                phase=Phase.STRESS_TEST.value,
            )
            strategy_id = self._coerce_uuid_text(stress_profile.get("strategy_id"))

        if strategy_id is not None:
            return HandlerRuntimePolicy(
                phase_stage="artifact_ops",
                tool_mode="replace",
                allowed_tools=self._build_strategy_artifact_ops_allowed_tools(),
            )

        return HandlerRuntimePolicy(
            phase_stage="schema_only",
            tool_mode="replace",
            allowed_tools=[
                self._build_strategy_tool_def(
                    allowed_tools=list(_STRATEGY_SCHEMA_ONLY_TOOL_NAMES),
                )
            ],
        )

    def _build_stress_test_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(
            artifacts=artifacts, phase=Phase.STRESS_TEST.value
        )
        raw_status = profile.get("backtest_status")
        status = raw_status.strip().lower() if isinstance(raw_status, str) else ""

        if status == "done":
            return HandlerRuntimePolicy(
                phase_stage="feedback",
                tool_mode="replace",
                allowed_tools=self._build_stress_feedback_allowed_tools(),
            )

        return HandlerRuntimePolicy(
            phase_stage="bootstrap",
            tool_mode="replace",
            allowed_tools=[
                self._build_market_data_tool_def(
                    allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
                ),
                self._build_backtest_tool_def(
                    allowed_tools=list(_BACKTEST_BOOTSTRAP_TOOL_NAMES),
                ),
            ],
        )

    def _build_strategy_artifact_ops_allowed_tools(self) -> list[dict[str, Any]]:
        return [
            self._build_strategy_tool_def(
                allowed_tools=list(_STRATEGY_ARTIFACT_OPS_TOOL_NAMES),
            ),
            self._build_market_data_tool_def(
                allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
            ),
            self._build_backtest_tool_def(
                allowed_tools=list(_BACKTEST_FEEDBACK_TOOL_NAMES),
            ),
        ]

    def _build_stress_feedback_allowed_tools(self) -> list[dict[str, Any]]:
        return [
            self._build_market_data_tool_def(
                allowed_tools=list(_MARKET_DATA_MINIMAL_TOOL_NAMES),
            ),
            self._build_backtest_tool_def(
                allowed_tools=list(_BACKTEST_FEEDBACK_TOOL_NAMES),
            ),
            self._build_strategy_tool_def(
                allowed_tools=list(_STRATEGY_ARTIFACT_OPS_TOOL_NAMES),
            ),
        ]

    @staticmethod
    def _extract_phase_profile(
        *, artifacts: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        phase_block = artifacts.get(phase)
        if not isinstance(phase_block, dict):
            return {}
        profile = phase_block.get("profile")
        if not isinstance(profile, dict):
            return {}
        return dict(profile)

    @staticmethod
    def _build_strategy_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "strategy",
            "server_url": settings.strategy_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    @staticmethod
    def _build_backtest_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "backtest",
            "server_url": settings.backtest_mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    @staticmethod
    def _build_market_data_tool_def(*, allowed_tools: list[str]) -> dict[str, Any]:
        return {
            "type": "mcp",
            "server_label": "market_data",
            "server_url": settings.mcp_server_url,
            "allowed_tools": allowed_tools,
            "require_approval": "never",
        }

    def _merge_tools(
        self,
        *,
        base_tools: list[dict[str, Any]] | None,
        runtime_policy: HandlerRuntimePolicy,
    ) -> list[dict[str, Any]] | None:
        merged = list(base_tools or [])

        requested_tools = runtime_policy.allowed_tools or []
        if requested_tools:
            if runtime_policy.tool_mode == "replace":
                merged = list(requested_tools)
            else:
                merged.extend(requested_tools)

        return merged or None

    def _attach_mcp_context_headers(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        user_id: UUID,
        session_id: UUID,
        phase: str,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return tools

        trace = get_chat_debug_trace()
        trace_id = trace.trace_id if trace and isinstance(trace.trace_id, str) else None
        token = create_mcp_context_token(
            user_id=user_id,
            session_id=session_id,
            ttl_seconds=settings.mcp_context_ttl_seconds,
            trace_id=trace_id,
            phase=phase,
        )

        decorated: list[dict[str, Any]] = []
        for item in tools:
            if not isinstance(item, dict):
                decorated.append(item)
                continue
            if str(item.get("type", "")).strip().lower() != "mcp":
                decorated.append(dict(item))
                continue
            server_label = str(item.get("server_label", "")).strip().lower()
            if server_label not in _MCP_CONTEXT_ENABLED_SERVER_LABELS:
                decorated.append(dict(item))
                continue

            next_tool = dict(item)
            raw_headers = item.get("headers")
            headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
            headers[MCP_CONTEXT_HEADER] = token
            next_tool["headers"] = headers
            decorated.append(next_tool)
        return decorated

    @staticmethod
    def _redact_stream_request_kwargs_for_trace(
        stream_request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload = copy.deepcopy(stream_request_kwargs)
        tools = payload.get("tools")
        if not isinstance(tools, list):
            return payload

        redacted_tools: list[Any] = []
        for item in tools:
            if not isinstance(item, dict):
                redacted_tools.append(item)
                continue

            next_tool = dict(item)
            raw_headers = next_tool.get("headers")
            if isinstance(raw_headers, dict):
                sanitized_headers: dict[str, Any] = {}
                for key, value in raw_headers.items():
                    normalized_key = str(key).strip().lower()
                    if normalized_key in {"authorization", MCP_CONTEXT_HEADER} or (
                        "token" in normalized_key
                    ):
                        sanitized_headers[str(key)] = "***"
                        continue
                    sanitized_headers[str(key)] = value
                next_tool["headers"] = sanitized_headers
            redacted_tools.append(next_tool)
        payload["tools"] = redacted_tools
        return payload

    @staticmethod
    def _merge_runtime_policies(
        *,
        phase_policy: HandlerRuntimePolicy,
        user_policy: HandlerRuntimePolicy,
    ) -> HandlerRuntimePolicy:
        phase_stage = user_policy.phase_stage or phase_policy.phase_stage
        tool_mode = phase_policy.tool_mode
        allowed_tools = list(phase_policy.allowed_tools or [])

        user_tools = list(user_policy.allowed_tools or [])
        if user_tools:
            if user_policy.tool_mode == "replace":
                tool_mode = "replace"
                allowed_tools = user_tools
            else:
                if tool_mode != "replace":
                    tool_mode = "append"
                allowed_tools.extend(user_tools)
        elif not allowed_tools:
            tool_mode = user_policy.tool_mode

        return HandlerRuntimePolicy(
            phase_stage=phase_stage,
            tool_mode=tool_mode,
            allowed_tools=allowed_tools or None,
        )

    def _consume_phase_carryover_memory(
        self, *, session: Session, phase: str
    ) -> str | None:
        metadata = dict(session.metadata_ or {})
        raw = metadata.get(_PHASE_CARRYOVER_META_KEY)
        if not isinstance(raw, dict):
            return None

        target_phase = raw.get("target_phase")
        block = raw.get("block")
        if not isinstance(target_phase, str) or not isinstance(block, str):
            metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
            session.metadata_ = metadata
            return None

        if target_phase != phase:
            metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
            session.metadata_ = metadata
            return None

        metadata.pop(_PHASE_CARRYOVER_META_KEY, None)
        session.metadata_ = metadata
        return block

    async def _store_phase_carryover_memory(
        self,
        *,
        session: Session,
        from_phase: str,
        to_phase: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        block = await self._build_phase_carryover_block(
            session_id=session.id,
            from_phase=from_phase,
            user_message=user_message,
            assistant_message=assistant_message,
        )
        if not isinstance(block, str) or not block.strip():
            return

        metadata = dict(session.metadata_ or {})
        metadata[_PHASE_CARRYOVER_META_KEY] = {
            "target_phase": to_phase,
            "from_phase": from_phase,
            "created_at": datetime.now(UTC).isoformat(),
            "block": block,
        }
        session.metadata_ = metadata

    async def _build_phase_carryover_block(
        self,
        *,
        session_id: UUID,
        from_phase: str,
        user_message: str,
        assistant_message: str,
    ) -> str | None:
        stmt = (
            select(Message.role, Message.content)
            .where(
                Message.session_id == session_id,
                Message.phase == from_phase,
                Message.role.in_(("user", "assistant")),
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(_PHASE_CARRYOVER_MAX_TURNS * 2)
        )
        rows = (await self.db.execute(stmt)).all()

        entries: list[tuple[str, str]] = []
        for role, content in reversed(rows):
            normalized = self._normalize_carryover_utterance(content)
            if normalized:
                entries.append((str(role), normalized))

        current_user = self._normalize_carryover_utterance(user_message)
        current_assistant = self._normalize_carryover_utterance(assistant_message)
        if current_user:
            entries.append(("user", current_user))
        if current_assistant:
            entries.append(("assistant", current_assistant))

        deduped: list[tuple[str, str]] = []
        for role, content in entries:
            if deduped and deduped[-1] == (role, content):
                continue
            deduped.append((role, content))

        tail = deduped[-(_PHASE_CARRYOVER_MAX_TURNS * 2) :]
        if not tail:
            return None

        lines = [
            f"[{_PHASE_CARRYOVER_TAG}]",
            "- note: previous phase dialogue snippets, reference only",
            f"- from_phase: {from_phase}",
        ]
        for role, content in tail:
            lines.append(f"- {role}: {content}")
        lines.append(f"[/{_PHASE_CARRYOVER_TAG}]")
        return "\n".join(lines) + "\n\n"

    def _normalize_carryover_utterance(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        if len(cleaned) > _PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE:
            trimmed = cleaned[:_PHASE_CARRYOVER_MAX_CHARS_PER_UTTERANCE].rstrip()
            cleaned = f"{trimmed}..."
        return cleaned

    def _increment_phase_turn_count(self, *, session: Session, phase: str) -> int:
        metadata = dict(session.metadata_ or {})
        raw_counts = metadata.get("phase_turn_counts")
        counts = dict(raw_counts) if isinstance(raw_counts, dict) else {}

        current = counts.get(phase, 0)
        try:
            current_value = int(current)
        except (TypeError, ValueError):
            current_value = 0

        next_value = max(0, current_value) + 1
        counts[phase] = next_value
        metadata["phase_turn_counts"] = counts
        session.metadata_ = metadata
        return next_value

    def _maybe_apply_stop_criteria_placeholder(
        self,
        *,
        session: Session,
        phase: str,
        phase_turn_count: int,
        language: str,
        assistant_text: str,
    ) -> tuple[str, str | None]:
        if phase not in {Phase.STRATEGY.value, Phase.STRESS_TEST.value}:
            return assistant_text, None
        if phase_turn_count < _STOP_CRITERIA_TURN_LIMIT:
            return assistant_text, None

        metadata = dict(session.metadata_ or {})
        raw_alerted = metadata.get("stop_criteria_alerted_phases")
        alerted_phases = (
            {
                item.strip()
                for item in raw_alerted
                if isinstance(item, str) and item.strip()
            }
            if isinstance(raw_alerted, list)
            else set()
        )

        if phase in alerted_phases:
            return assistant_text, None

        alerted_phases.add(phase)
        metadata["stop_criteria_alerted_phases"] = sorted(alerted_phases)
        metadata["stop_criteria_placeholder"] = {
            "enabled": True,
            "max_turns_per_phase": _STOP_CRITERIA_TURN_LIMIT,
            "performance_threshold_todo": True,
            "last_triggered_phase": phase,
            "last_triggered_at": datetime.now(UTC).isoformat(),
        }
        session.metadata_ = metadata

        if language == "zh":
            hint = (
                "提示：当前策略迭代轮次已较多。你可以考虑更换策略方向或重置一次。"
                "（占位逻辑：后续将接入真实绩效阈值判断）"
            )
        else:
            hint = (
                "Hint: this strategy has gone through many iterations. "
                "Consider trying a new strategy direction or restarting once. "
                "(Placeholder logic: real performance-threshold checks will be added later.)"
            )

        if not assistant_text.strip():
            return hint, hint
        return f"{assistant_text}\n\n{hint}", f"\n\n{hint}"

    @staticmethod
    def _build_empty_turn_fallback_text(
        *,
        phase: str,
        missing_fields: list[str],
        language: str,
    ) -> str:
        field = missing_fields[0] if missing_fields else ""
        is_zh = isinstance(language, str) and language.strip().lower().startswith("zh")

        if is_zh:
            prompts = {
                "trading_years_bucket": "我这轮没有收到可显示的回复。请告诉我你的交易经验年限（例如：5年以上）。",
                "risk_tolerance": "我这轮没有收到可显示的回复。请告诉我你的风险偏好（保守/中等/激进/非常激进）。",
                "return_expectation": "我这轮没有收到可显示的回复。请告诉我你的收益预期（保本/平衡增长/增长/高增长）。",
                "target_market": "我这轮没有收到可显示的回复。请告诉我你想交易的市场（如：美股、加密、外汇、期货）。",
                "target_instrument": "我这轮没有收到可显示的回复。请告诉我你想交易的标的（例如：SPY 或 GBPUSD）。",
                "opportunity_frequency_bucket": "我这轮没有收到可显示的回复。请告诉我你期望的机会频率（每月少量/每周几次/每日/每日多次）。",
                "holding_period_bucket": "我这轮没有收到可显示的回复。请告诉我你的持仓周期（超短线/日内/波段数天/持有数周以上）。",
            }
            if field in prompts:
                return prompts[field]
            return "我这轮没有收到可显示的回复。请再发送一次你的答案，我们继续。"

        prompts = {
            "trading_years_bucket": "I did not receive a displayable reply this turn. Please share your trading experience bucket (for example: 5+ years).",
            "risk_tolerance": "I did not receive a displayable reply this turn. Please share your risk tolerance (conservative/moderate/aggressive/very aggressive).",
            "return_expectation": "I did not receive a displayable reply this turn. Please share your return expectation (capital preservation/balanced growth/growth/high growth).",
            "target_market": "I did not receive a displayable reply this turn. Please tell me your target market (us_stocks/crypto/forex/futures).",
            "target_instrument": "I did not receive a displayable reply this turn. Please tell me your target instrument (for example: SPY or GBPUSD).",
            "opportunity_frequency_bucket": "I did not receive a displayable reply this turn. Please share your opportunity frequency (few_per_month/few_per_week/daily/multiple_per_day).",
            "holding_period_bucket": "I did not receive a displayable reply this turn. Please share your holding period bucket (intraday_scalp/intraday/swing_days/position_weeks_plus).",
        }
        if field in prompts:
            return prompts[field]
        if phase == Phase.STRATEGY.value:
            return "I did not receive a displayable reply this turn. Please restate your strategy request and I will continue."
        return "I did not receive a displayable reply this turn. Please resend your answer and we can continue."

    @staticmethod
    def _build_runtime_policy(payload: ChatSendRequest) -> HandlerRuntimePolicy:
        runtime = payload.runtime_policy
        if runtime is None:
            return HandlerRuntimePolicy()
        return HandlerRuntimePolicy(
            phase_stage=runtime.phase_stage,
            tool_mode=runtime.tool_mode,
            allowed_tools=list(runtime.allowed_tools or []),
        )

    def _sse(self, event: str, payload: dict[str, Any]) -> str:
        record_chat_debug_trace(
            "orchestrator_to_frontend_sse",
            {
                "event": event,
                "payload": payload,
            },
        )
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
