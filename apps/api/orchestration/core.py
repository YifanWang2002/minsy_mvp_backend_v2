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


class OrchestratorCoreMixin:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

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
        turn_request_id = f"turn_{uuid4().hex}"
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
        handler = self._resolve_handler_from_module(phase_before)
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
            turn_request_id=turn_request_id,
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

    def _compute_initial_phase(self, profile: UserProfile | None) -> str:
        """Determine which phase to start a new session in.

        KYC only needs to be done once per user; completed-KYC users
        always start from PRE_STRATEGY.
        """
        if _kyc_handler.is_profile_complete(profile):
            return Phase.PRE_STRATEGY.value
        return Phase.KYC.value

    @staticmethod
    def _ensure_phase_keyed(artifacts: dict[str, Any]) -> dict[str, Any]:
        """Ensure session artifacts are phase-keyed and initialized."""
        normalized = dict(artifacts)
        for phase in WORKFLOW_PHASES:
            phase_block = normalized.get(phase)
            if isinstance(phase_block, dict) and "profile" in phase_block:
                continue
            handler = ChatOrchestrator._resolve_handler_from_module(phase)
            if handler:
                normalized[phase] = handler.init_artifacts()
        return normalized

    @staticmethod
    def _resolve_handler_from_module(phase: str) -> Any:
        # Keep package-level monkeypatch hooks working after package split.
        from apps.api import orchestration as orchestrator_mod

        return orchestrator_mod.get_handler(phase)


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
