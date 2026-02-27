"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403
from .carryover import CarryoverMixin
from .fallback import FallbackMixin
from .genui_wrapper import GenUiWrapperMixin
from .mcp_records import McpRecordsMixin
from .postprocessor import PostProcessorMixin
from .prompt_builder import PromptBuilderMixin
from .strategy_context import StrategyContextMixin
from .stream_handler import StreamHandlerMixin


class _KycHandlerRef:
    """Lazy reference to KYC handler to avoid circular imports."""

    @staticmethod
    def is_profile_complete(profile: UserProfile | None) -> bool:
        from apps.api.agents.handlers.kyc_handler import KYCHandler

        return KYCHandler().is_profile_complete(profile)

    @staticmethod
    def build_profile_from_user_profile(profile: UserProfile) -> dict[str, Any]:
        from apps.api.agents.handlers.kyc_handler import KYCHandler

        return KYCHandler().build_profile_from_user_profile(profile)

    @staticmethod
    def _compute_missing(profile: dict[str, Any]) -> list[str]:
        from apps.api.agents.handlers.kyc_handler import KYCHandler

        return KYCHandler()._compute_missing(profile)


_kyc_handler = _KycHandlerRef()


class OrchestratorCoreMixin:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    def _compute_initial_phase(self, profile: UserProfile | None) -> str:
        """Determine which phase to start a new session in.

        KYC only needs to be done once per user; completed-KYC users
        always start from PRE_STRATEGY.
        """
        if _kyc_handler.is_profile_complete(profile):
            return Phase.PRE_STRATEGY.value
        return Phase.KYC.value

    @staticmethod
    def _resolve_handler_from_module(phase: str) -> Any:
        # Keep package-level monkeypatch hooks working after package split.
        from apps.api import orchestration as orchestrator_mod

        return orchestrator_mod.get_handler(phase)

    @staticmethod
    def _ensure_phase_keyed(artifacts: dict[str, Any]) -> dict[str, Any]:
        """Ensure session artifacts are phase-keyed and initialized."""
        normalized = dict(artifacts)
        for phase in WORKFLOW_PHASES:
            phase_block = normalized.get(phase)
            if isinstance(phase_block, dict) and "profile" in phase_block:
                continue
            handler = OrchestratorCoreMixin._resolve_handler_from_module(phase)
            if handler:
                normalized[phase] = handler.init_artifacts()
        return normalized

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
        turn_id = self._resolve_turn_id(payload.client_turn_id)
        turn_request_id = turn_id
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
        user_message = Message(
            session_id=session.id,
            role="user",
            content=payload.message,
            phase=phase_before,
        )
        self.db.add(user_message)
        await self.db.flush()
        self._update_stream_recovery(
            session=session,
            turn_id=turn_id,
            state=_STREAM_RECOVERY_STATE_STREAMING,
            user_message_id=str(user_message.id),
        )
        phase_turn_count = self._increment_phase_turn_count(
            session=session,
            phase=phase_before,
        )
        await self.db.commit()
        await self.db.refresh(session)

        yield self._sse(
            "stream",
            {
                "type": "stream_start",
                "session_id": str(session.id),
                "phase": phase_before,
                "turn_id": turn_id,
            },
        )

        # -- resolve handler for current phase ---------------------------
        handler = self._resolve_handler_from_module(phase_before)
        if handler is None:
            # Terminal phases (completed, error) or unknown
            assistant_text = f"{phase_before} phase has no active handler."
            assistant_message = Message(
                session_id=session.id,
                role="assistant",
                content=assistant_text,
                phase=session.current_phase,
            )
            self.db.add(assistant_message)
            await self.db.flush()
            self._update_stream_recovery(
                session=session,
                turn_id=turn_id,
                state=_STREAM_RECOVERY_STATE_COMPLETED,
                assistant_message_id=str(assistant_message.id),
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
                    "turn_id": turn_id,
                    "assistant_message_id": str(assistant_message.id),
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
            turn_request_id=turn_request_id,
            turn_id=turn_id,
            user_message_id=user_message.id,
        )
        stream_state = _TurnStreamState()
        try:
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
                turn_id=turn_id,
            ):
                yield tail_event

            # Auto follow-up for strategy -> deployment transition to avoid
            # requiring user to send "yes deploy" twice.
            if (
                post_process_result.transition_from_phase == Phase.STRATEGY.value
                and post_process_result.transition_to_phase == Phase.DEPLOYMENT.value
            ):
                async for followup_event in self._run_deployment_auto_followup(
                    user=user,
                    session=session,
                    streamer=streamer,
                    language=language,
                ):
                    yield followup_event

            log_agent("orchestrator", f"session={session.id} phase={session.current_phase}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._mark_stream_recovery_failed(
                session=session,
                turn_id=turn_id,
                error=str(exc),
            )
            raise

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

    @staticmethod
    def _resolve_turn_id(client_turn_id: str | None) -> str:
        normalized = client_turn_id.strip() if isinstance(client_turn_id, str) else ""
        if normalized:
            return normalized
        return f"turn_{uuid4().hex}"

    def _update_stream_recovery(
        self,
        *,
        session: Session,
        turn_id: str,
        state: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        metadata = dict(session.metadata_ or {})
        existing = metadata.get(_STREAM_RECOVERY_META_KEY)
        existing_record = dict(existing) if isinstance(existing, dict) else {}
        existing_turn_id = str(existing_record.get("turn_id", "")).strip()
        same_turn = existing_turn_id == turn_id

        now_iso = datetime.now(UTC).isoformat()
        started_at = existing_record.get("started_at") if same_turn else None
        if not isinstance(started_at, str) or not started_at.strip():
            started_at = now_iso

        resolved_user_message_id = (
            user_message_id
            if isinstance(user_message_id, str) and user_message_id.strip()
            else None
        )
        if resolved_user_message_id is None and same_turn:
            raw_user_message_id = existing_record.get("user_message_id")
            if isinstance(raw_user_message_id, str) and raw_user_message_id.strip():
                resolved_user_message_id = raw_user_message_id.strip()

        resolved_assistant_message_id = (
            assistant_message_id
            if isinstance(assistant_message_id, str) and assistant_message_id.strip()
            else None
        )
        if resolved_assistant_message_id is None and same_turn:
            raw_assistant_message_id = existing_record.get("assistant_message_id")
            if isinstance(raw_assistant_message_id, str) and raw_assistant_message_id.strip():
                resolved_assistant_message_id = raw_assistant_message_id.strip()

        record: dict[str, Any] = {
            "turn_id": turn_id,
            "state": state,
            "started_at": started_at,
            "updated_at": now_iso,
        }
        if resolved_user_message_id is not None:
            record["user_message_id"] = resolved_user_message_id
        if resolved_assistant_message_id is not None:
            record["assistant_message_id"] = resolved_assistant_message_id

        error_text = error.strip() if isinstance(error, str) else ""
        if error_text:
            record["error"] = error_text[:600]

        metadata[_STREAM_RECOVERY_META_KEY] = record
        session.metadata_ = metadata
        return record

    async def _mark_stream_recovery_failed(
        self,
        *,
        session: Session,
        turn_id: str,
        error: str | None,
    ) -> None:
        try:
            self._update_stream_recovery(
                session=session,
                turn_id=turn_id,
                state=_STREAM_RECOVERY_STATE_FAILED,
                error=error,
            )
            await self.db.commit()
        except Exception as commit_exc:  # noqa: BLE001
            try:
                await self.db.rollback()
            except Exception:  # noqa: BLE001
                pass
            log_agent(
                "orchestrator",
                (
                    f"session={session.id} turn={turn_id} "
                    f"stream_recovery_failed_commit_error={type(commit_exc).__name__}"
                ),
            )

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

    async def _run_deployment_auto_followup(
        self,
        *,
        user: User,
        session: Session,
        streamer: ResponsesEventStreamer,
        language: str,
    ) -> AsyncIterator[str]:
        """Auto follow-up turn after strategy -> deployment transition.

        This eliminates the need for users to send "yes deploy" twice by
        automatically triggering a deployment turn after the phase transition.
        """
        auto_message = _default_deployment_auto_message(language=language)
        follow_up_payload = ChatSendRequest(
            session_id=session.id,
            message=auto_message,
        )

        log_agent(
            "orchestrator",
            f"session={session.id} deployment_auto_followup_start",
        )

        try:
            async for chunk in self.handle_message_stream(
                user,
                follow_up_payload,
                streamer,
                language=language,
            ):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            err_name = type(exc).__name__
            log_agent(
                "orchestrator",
                f"session={session.id} deployment_auto_followup_error={err_name}",
            )
            # Swallow the error to avoid breaking the main turn; user can
            # manually retry deployment in the next message.


def _default_deployment_auto_message(*, language: str) -> str:
    """Generate the auto follow-up message for deployment phase."""
    if language.startswith("zh"):
        return (
            "用户已确认策略并准备部署。"
            "请检查部署状态，如果一切就绪，创建并启动 paper deployment。"
        )
    return (
        "The user has confirmed the strategy and is ready to deploy. "
        "Please check deployment readiness and create/start a paper deployment."
    )


class ChatOrchestrator(
    OrchestratorCoreMixin,
    PromptBuilderMixin,
    StreamHandlerMixin,
    PostProcessorMixin,
    GenUiWrapperMixin,
    McpRecordsMixin,
    StrategyContextMixin,
    CarryoverMixin,
    FallbackMixin,
):
    """Coordinates message flow and phase transitions."""
