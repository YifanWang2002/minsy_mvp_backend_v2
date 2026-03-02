"""Test runner for orchestrator conversations.

Coordinates the ObservableChatOrchestrator with scripted/conditional users
to run complete multi-turn conversations and collect observations.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, UTC
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.orchestration.openai_stream_service import ResponsesEventStreamer
from apps.api.schemas.requests import ChatSendRequest, RuntimePolicy
from packages.infra.db.models.session import Session
from packages.infra.db.models.user import User

from .observable_orchestrator import ObservableChatOrchestrator
from .observer import TurnObserver
from .scripted_user import ScriptedUser, ScriptedReply, ConditionalUser, CompositeUser
from .observation_types import ConversationObservation, TurnObservation

if TYPE_CHECKING:
    from packages.domain.session.services.openai_stream_service import (
        ResponsesEventStreamer as StreamerProtocol,
    )


class OrchestratorTestRunner:
    """Runs orchestrator conversations with full observability.

    Usage:
        # With scripted user
        user = ScriptedUser([
            ScriptedReply("Hello"),
            ScriptedReply("3-5 years experience"),
        ])
        runner = OrchestratorTestRunner(db, streamer, test_user)
        observation = await runner.run_conversation(user)

        # With conditional user
        def decide(turn):
            if "risk" in turn.cleaned_text.lower():
                return ScriptedReply("Medium risk")
            return None

        conditional = ConditionalUser(decide)
        observation = await runner.run_conversation(conditional, initial_message="Hi")
    """

    def __init__(
        self,
        db: AsyncSession,
        streamer: ResponsesEventStreamer,
        user: User,
        *,
        language: str = "en",
        initial_session_id: UUID | None = None,
    ) -> None:
        """Initialize the test runner.

        Args:
            db: Database session
            streamer: OpenAI responses event streamer
            user: The User model instance to use
            language: Language for the conversation
            initial_session_id: Optional existing session to continue
        """
        self._db = db
        self._streamer = streamer
        self._user = user
        self._language = language
        self._initial_session_id = initial_session_id

        # State
        self._session_id: UUID | None = initial_session_id
        self._observer: TurnObserver | None = None
        self._orchestrator: ObservableChatOrchestrator | None = None
        self._collected_events: list[str] = []

    async def run_conversation(
        self,
        scripted_user: ScriptedUser | ConditionalUser | CompositeUser,
        *,
        initial_message: str | None = None,
        max_turns: int = 20,
        stop_on_phase: str | None = None,
        stop_on_transition: bool = False,
        collect_events: bool = True,
    ) -> ConversationObservation:
        """Run a complete conversation and return observations.

        Args:
            scripted_user: User simulation (ScriptedUser, ConditionalUser, or CompositeUser)
            initial_message: First message (required for ConditionalUser)
            max_turns: Maximum turns before stopping
            stop_on_phase: Stop when reaching this phase
            stop_on_transition: Stop after any phase transition
            collect_events: Whether to collect SSE events

        Returns:
            ConversationObservation with all turn data
        """
        self._collected_events = []

        # Initialize observer and orchestrator
        session_id = self._session_id or uuid4()
        self._observer = TurnObserver(session_id, self._user.id)
        self._orchestrator = ObservableChatOrchestrator(self._db, self._observer)

        turn_count = 0
        last_turn: TurnObservation | None = None

        # Get first message
        first_reply = self._get_first_reply(scripted_user, initial_message)
        if first_reply is None:
            return self._observer.finalize()

        current_reply: ScriptedReply | None = first_reply

        while current_reply is not None and turn_count < max_turns:
            turn_count += 1

            # Execute the turn
            turn_obs = await self._execute_turn(
                current_reply,
                collect_events=collect_events,
            )
            last_turn = turn_obs

            # Check stop conditions
            if self._should_stop(turn_obs, stop_on_phase, stop_on_transition):
                break

            # Get next reply
            current_reply = self._get_next_reply(scripted_user, turn_obs)

        return self._observer.finalize()

    async def run_single_turn(
        self,
        message: str,
        *,
        runtime_policy: RuntimePolicy | None = None,
        collect_events: bool = True,
    ) -> TurnObservation:
        """Run a single turn and return the observation.

        Useful for fine-grained control over individual turns.
        """
        # Initialize if needed
        if self._observer is None:
            session_id = self._session_id or uuid4()
            self._observer = TurnObserver(session_id, self._user.id)
            self._orchestrator = ObservableChatOrchestrator(self._db, self._observer)

        reply = ScriptedReply(
            message=message,
            runtime_policy=runtime_policy.model_dump() if runtime_policy else None,
        )

        return await self._execute_turn(reply, collect_events=collect_events)

    async def _execute_turn(
        self,
        reply: ScriptedReply,
        *,
        collect_events: bool = True,
    ) -> TurnObservation:
        """Execute a single turn with the given reply."""
        if self._orchestrator is None or self._observer is None:
            raise RuntimeError("Runner not initialized")

        # Apply delay if specified
        if reply.delay_ms > 0:
            await asyncio.sleep(reply.delay_ms / 1000)

        # Build request
        request_kwargs = reply.to_request_kwargs()
        if self._session_id:
            request_kwargs["session_id"] = self._session_id

        # Convert runtime_policy dict to RuntimePolicy if present
        runtime_policy_dict = request_kwargs.pop("runtime_policy", None)
        runtime_policy = None
        if runtime_policy_dict:
            runtime_policy = RuntimePolicy(**runtime_policy_dict)

        payload = ChatSendRequest(
            session_id=self._session_id,
            message=request_kwargs["message"],
            runtime_policy=runtime_policy,
        )

        # Execute the turn
        events: list[str] = []
        async for event in self._orchestrator.handle_message_stream(
            self._user,
            payload,
            self._streamer,
            language=self._language,
        ):
            if collect_events:
                events.append(event)
                self._collected_events.append(event)

            # Extract session_id from stream_start event
            if '"type": "stream_start"' in event or '"type":"stream_start"' in event:
                import json
                try:
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data.get("type") == "stream_start":
                                sid = data.get("session_id")
                                if sid:
                                    self._session_id = UUID(sid)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Get the turn observation (already finalized by orchestrator)
        return self._observer.conversation.turns[-1]

    def _get_first_reply(
        self,
        user: ScriptedUser | ConditionalUser | CompositeUser,
        initial_message: str | None,
    ) -> ScriptedReply | None:
        """Get the first reply to start the conversation."""
        if isinstance(user, ScriptedUser):
            return user.next_reply()
        elif isinstance(user, CompositeUser):
            return user.get_reply(None)
        elif isinstance(user, ConditionalUser):
            if initial_message is None:
                raise ValueError("initial_message required for ConditionalUser")
            return ScriptedReply(initial_message)
        return None

    def _get_next_reply(
        self,
        user: ScriptedUser | ConditionalUser | CompositeUser,
        last_turn: TurnObservation,
    ) -> ScriptedReply | None:
        """Get the next reply based on the last turn."""
        if isinstance(user, ScriptedUser):
            return user.next_reply()
        elif isinstance(user, CompositeUser):
            return user.get_reply(last_turn)
        elif isinstance(user, ConditionalUser):
            return user.decide_reply(last_turn)
        return None

    def _should_stop(
        self,
        turn: TurnObservation,
        stop_on_phase: str | None,
        stop_on_transition: bool,
    ) -> bool:
        """Check if we should stop the conversation."""
        if stop_on_phase and turn.phase == stop_on_phase:
            return True
        if stop_on_phase and turn.phase_transition:
            _, to_phase = turn.phase_transition
            if to_phase == stop_on_phase:
                return True
        if stop_on_transition and turn.phase_transition:
            return True
        return False

    @property
    def session_id(self) -> UUID | None:
        """Current session ID."""
        return self._session_id

    @property
    def collected_events(self) -> list[str]:
        """All SSE events collected during the conversation."""
        return list(self._collected_events)

    @property
    def observer(self) -> TurnObserver | None:
        """The turn observer (for accessing partial results)."""
        return self._observer

    def reset(self) -> None:
        """Reset the runner for a new conversation."""
        self._session_id = self._initial_session_id
        self._observer = None
        self._orchestrator = None
        self._collected_events = []


class QuickTestRunner:
    """Simplified runner for quick single-turn tests.

    Usage:
        runner = QuickTestRunner(db, streamer, user)
        obs = await runner.send("Hello, I want to create a strategy")
        print(obs.cleaned_text)
    """

    def __init__(
        self,
        db: AsyncSession,
        streamer: ResponsesEventStreamer,
        user: User,
        *,
        language: str = "en",
    ) -> None:
        self._runner = OrchestratorTestRunner(
            db, streamer, user, language=language
        )

    async def send(
        self,
        message: str,
        *,
        runtime_policy: RuntimePolicy | None = None,
    ) -> TurnObservation:
        """Send a message and get the turn observation."""
        return await self._runner.run_single_turn(
            message, runtime_policy=runtime_policy
        )

    async def send_many(
        self,
        messages: list[str],
        *,
        max_turns: int | None = None,
    ) -> ConversationObservation:
        """Send multiple messages in sequence."""
        replies = [ScriptedReply(m) for m in messages]
        user = ScriptedUser(replies)
        return await self._runner.run_conversation(
            user, max_turns=max_turns or len(messages)
        )

    @property
    def session_id(self) -> UUID | None:
        """Current session ID."""
        return self._runner.session_id

    def reset(self) -> None:
        """Reset for a new conversation."""
        self._runner.reset()
