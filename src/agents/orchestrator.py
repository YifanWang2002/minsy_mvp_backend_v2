"""Session orchestrator for chat phases and profile persistence.

Dispatches to pluggable :class:`PhaseHandler` instances via a registry,
so adding a new phase only requires writing a handler and registering it.

This module exclusively uses the OpenAI **Responses API** with
``previous_response_id`` for conversation context and separated
static instructions / dynamic state for prompt-caching efficiency.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import AsyncIterator
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
from src.models.phase_transition import PhaseTransition
from src.models.session import Message, Session
from src.models.user import User, UserProfile
from src.services.openai_stream_service import ResponsesEventStreamer
from src.util.logger import log_agent

_AGENT_UI_TAG = "AGENT_UI_JSON"
_AGENT_STATE_PATCH_TAG = "AGENT_STATE_PATCH"
_STOP_CRITERIA_TURN_LIMIT = 10

# Singleton for KYC-specific helpers (profile loading from UserProfile)
_kyc_handler = KYCHandler()


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
        session = await self._resolve_session(user_id=user.id, session_id=payload.session_id)
        phase_before = session.current_phase
        user_runtime_policy = self._build_runtime_policy(payload)

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

        yield self._sse("stream", {
            "type": "stream_start",
            "session_id": str(session.id),
            "phase": phase_before,
        })

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
            await self.db.commit()
            await self.db.refresh(session)

            yield self._sse("stream", {"type": "text_delta", "delta": assistant_text})
            yield self._sse("stream", {
                "type": "done",
                "session_id": str(session.id),
                "phase": session.current_phase,
                "status": session.status,
                "kyc_status": await self._fetch_kyc_status(user.id),
                "missing_fields": [],
            })
            log_agent("orchestrator", f"session={session.id} phase={session.current_phase} (no handler)")
            return

        # -- migrate artifacts if needed ---------------------------------
        artifacts = copy.deepcopy(session.artifacts or {})
        artifacts = self._ensure_phase_keyed(artifacts)
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

        # -- build prompt via handler ------------------------------------
        ctx = PhaseContext(
            user_id=user.id,
            session_artifacts=artifacts,
            language=language,
            runtime_policy=runtime_policy,
        )
        prompt = handler.build_prompt(ctx, payload.message)

        tools = self._merge_tools(
            base_tools=prompt.tools,
            runtime_policy=runtime_policy,
            message=payload.message,
        )

        # -- stream from Responses API -----------------------------------
        full_text = ""
        completed_usage: dict[str, Any] = {}
        stream_error_message: str | None = None
        try:
            async for event in streamer.stream_events(
                model=prompt.model or settings.openai_response_model,
                input_text=prompt.enriched_input,
                instructions=prompt.instructions,
                previous_response_id=session.previous_response_id,
                tools=tools,
                tool_choice=prompt.tool_choice,
                reasoning=prompt.reasoning,
            ):
                event_type = str(event.get("type", "unknown"))
                seq = event.get("sequence_number")

                # Forward raw OpenAI event to frontend
                yield self._sse("openai_event", {
                    "type": "openai_event",
                    "openai_type": event_type,
                    "sequence_number": seq,
                    "payload": event,
                })

                if event_type == "response.stream_error":
                    err = event.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message")
                        if isinstance(msg, str) and msg.strip():
                            stream_error_message = msg.strip()
                    if not stream_error_message:
                        stream_error_message = "Upstream stream interrupted."
                    continue

                # Accumulate text deltas
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        full_text += delta
                        yield self._sse("stream", {
                            "type": "text_delta",
                            "delta": delta,
                            "sequence_number": seq,
                        })

                # Forward MCP events
                if "mcp_" in event_type:
                    yield self._sse("stream", {
                        "type": "mcp_event",
                        "openai_type": event_type,
                        "sequence_number": seq,
                        "payload": event,
                    })

                # Capture response id + usage on completion
                if event_type == "response.completed":
                    response_obj = event.get("response")
                    if isinstance(response_obj, dict):
                        usage = response_obj.get("usage")
                        if isinstance(usage, dict):
                            completed_usage = usage
                        resp_id = response_obj.get("id")
                        if isinstance(resp_id, str) and resp_id:
                            session.previous_response_id = resp_id
        except Exception as exc:  # noqa: BLE001
            stream_error_message = f"{type(exc).__name__}: {exc}"

        # -- post-processing: extract patches / genui ---------------------
        cleaned_text, genui_payloads, raw_patches = self._extract_wrapped_payloads(full_text)
        assistant_text = cleaned_text.strip() or full_text.strip()
        if stream_error_message and not assistant_text:
            assistant_text = (
                "The upstream AI stream was interrupted before completion. "
                "Please retry this step."
            )

        selected_genui_payloads = normalize_genui_payloads(
            genui_payloads,
            allow_passthrough_unregistered=True,
        )

        # -- delegate post-processing to handler --------------------------
        result = await handler.post_process(ctx, raw_patches, self.db)

        # Write back updated artifacts
        session.artifacts = result.artifacts

        # Suppress genui when phase is complete
        if result.completed:
            selected_genui_payloads = []

        # Apply handler's genui filter
        filtered_genui_payloads: list[dict[str, Any]] = []
        for genui_payload in selected_genui_payloads:
            filtered = handler.filter_genui(genui_payload, ctx)
            if filtered is not None:
                filtered_genui_payloads.append(filtered)
        filtered_genui_payloads = self._ensure_pre_strategy_chart_payload(
            phase_before=phase_before,
            artifacts=result.artifacts,
            genui_payloads=filtered_genui_payloads,
            instrument_before=pre_strategy_instrument_before,
        )

        # Handle transition
        transitioned = False
        if result.completed and result.next_phase:
            await self._transition_phase(
                session=session,
                to_phase=result.next_phase,
                trigger="ai_output",
                metadata={"reason": result.transition_reason or "phase_completed"},
            )
            transitioned = True

        # Fallback: use genui question as assistant text
        if not assistant_text and filtered_genui_payloads:
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
                assistant_text = question.strip()

        # Fetch kyc_status for the done event
        kyc_status = result.phase_status.get("kyc_status")
        if kyc_status is None:
            kyc_status = await self._fetch_kyc_status(user.id)

        assistant_text, stop_criteria_delta = self._maybe_apply_stop_criteria_placeholder(
            session=session,
            phase=phase_before,
            phase_turn_count=phase_turn_count,
            language=language,
            assistant_text=assistant_text,
        )

        # -- persist assistant message ------------------------------------
        self.db.add(
            Message(
                session_id=session.id,
                role="assistant",
                content=assistant_text,
                phase=phase_before,
                response_id=session.previous_response_id,
                tool_calls=filtered_genui_payloads or None,
                token_usage=completed_usage or None,
            )
        )

        session.last_activity_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(session)

        # -- tail events --------------------------------------------------
        for genui_payload in filtered_genui_payloads:
            yield self._sse("stream", {"type": "genui", "payload": genui_payload})

        if transitioned:
            yield self._sse("stream", {
                "type": "phase_change",
                "from_phase": phase_before,
                "to_phase": session.current_phase,
            })

        if isinstance(stop_criteria_delta, str) and stop_criteria_delta.strip():
            yield self._sse("stream", {
                "type": "text_delta",
                "delta": stop_criteria_delta,
            })

        if stream_error_message and not full_text.strip() and assistant_text.strip():
            yield self._sse("stream", {
                "type": "text_delta",
                "delta": assistant_text,
            })
        yield self._sse("stream", {
            "type": "done",
            "session_id": str(session.id),
            "phase": session.current_phase,
            "status": session.status,
            "kyc_status": kyc_status,
            "missing_fields": result.missing_fields,
            "usage": completed_usage,
            "stream_error": stream_error_message,
        })

        log_agent("orchestrator", f"session={session.id} phase={session.current_phase}")

    # ------------------------------------------------------------------
    # Private helpers – session / DB
    # ------------------------------------------------------------------

    async def _resolve_session(self, *, user_id: UUID, session_id: UUID | None) -> Session:
        if session_id is None:
            return await self.create_session(user_id=user_id)

        stmt = (
            select(Session)
            .where(Session.id == session_id, Session.user_id == user_id)
            .options(selectinload(Session.messages))
        )
        session = await self.db.scalar(stmt)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
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

        return cleaned, genui_payloads, patch_payloads

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

    @staticmethod
    def _build_tag_pattern(tag: str) -> re.Pattern[str]:
        escaped_tag = re.escape(tag)
        return re.compile(
            rf"<\s*{escaped_tag}\s*>(.*?)<\s*/\s*{escaped_tag}\s*>",
            flags=re.DOTALL | re.IGNORECASE,
        )

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
        if any(payload.get("type") == "tradingview_chart" for payload in genui_payloads):
            return genui_payloads
        if not any(payload.get("type") == "choice_prompt" for payload in genui_payloads):
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

    # ------------------------------------------------------------------
    # Private helpers – misc
    # ------------------------------------------------------------------

    def _resolve_runtime_policy(
        self,
        *,
        phase: str,
        artifacts: dict[str, Any],
        user_runtime_policy: HandlerRuntimePolicy,
    ) -> HandlerRuntimePolicy:
        phase_policy = self._build_phase_runtime_policy(phase=phase, artifacts=artifacts)
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
        profile = self._extract_phase_profile(artifacts=artifacts, phase=Phase.STRATEGY.value)
        strategy_id = profile.get("strategy_id")

        if isinstance(strategy_id, str) and strategy_id.strip():
            return HandlerRuntimePolicy(
                phase_stage="artifact_ops",
                tool_mode="replace",
                allowed_tools=[
                    self._build_strategy_tool_def(
                        allowed_tools=[
                            "strategy_validate_dsl",
                            "strategy_upsert_dsl",
                            "get_indicator_detail",
                            "get_indicator_catalog",
                        ],
                    )
                ],
            )

        return HandlerRuntimePolicy(
            phase_stage="schema_only",
            tool_mode="replace",
            allowed_tools=[
                self._build_strategy_tool_def(
                    allowed_tools=["strategy_validate_dsl"],
                )
            ],
        )

    def _build_stress_test_runtime_policy(
        self,
        *,
        artifacts: dict[str, Any],
    ) -> HandlerRuntimePolicy:
        profile = self._extract_phase_profile(artifacts=artifacts, phase=Phase.STRESS_TEST.value)
        raw_status = profile.get("backtest_status")
        status = raw_status.strip().lower() if isinstance(raw_status, str) else ""

        if status == "done":
            return HandlerRuntimePolicy(
                phase_stage="feedback",
                tool_mode="replace",
                allowed_tools=[
                    self._build_backtest_tool_def(
                        allowed_tools=["backtest_create_job", "backtest_get_job"],
                    ),
                    self._build_strategy_tool_def(
                        allowed_tools=[
                            "strategy_validate_dsl",
                            "strategy_upsert_dsl",
                            "get_indicator_detail",
                            "get_indicator_catalog",
                        ],
                    ),
                ],
            )

        return HandlerRuntimePolicy(
            phase_stage="bootstrap",
            tool_mode="replace",
            allowed_tools=[
                self._build_backtest_tool_def(
                    allowed_tools=["backtest_create_job", "backtest_get_job"],
                )
            ],
        )

    @staticmethod
    def _extract_phase_profile(*, artifacts: dict[str, Any], phase: str) -> dict[str, Any]:
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

    def _build_optional_tools(self, message: str) -> list[dict[str, Any]] | None:
        lower = message.lower()
        dice_keywords = ("dice", "roll", "掷骰", "骰子")
        if not any(kw in lower for kw in dice_keywords):
            return None
        return [
            {
                "type": "mcp",
                "server_label": "dice",
                "server_url": settings.dice_mcp_server_url,
                "allowed_tools": ["roll_dice"],
                "require_approval": "never",
            }
        ]

    def _merge_tools(
        self,
        *,
        base_tools: list[dict[str, Any]] | None,
        runtime_policy: HandlerRuntimePolicy,
        message: str,
    ) -> list[dict[str, Any]] | None:
        merged = list(base_tools or [])

        requested_tools = runtime_policy.allowed_tools or []
        if requested_tools:
            if runtime_policy.tool_mode == "replace":
                merged = list(requested_tools)
            else:
                merged.extend(requested_tools)

        keyword_tools = self._build_optional_tools(message)
        if keyword_tools:
            merged.extend(keyword_tools)

        return merged or None

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
        alerted_phases = {
            item.strip()
            for item in raw_alerted
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_alerted, list) else set()

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
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
