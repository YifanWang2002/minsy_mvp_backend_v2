"""Scripted user simulation for orchestrator testing.

Provides mechanisms to simulate user replies in multi-turn conversations:
- ScriptedReply: A single pre-defined user reply
- ScriptedUser: Sequential replay of pre-defined replies
- ConditionalUser: Dynamic reply selection based on AI responses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .observation_types import TurnObservation


@dataclass
class ScriptedReply:
    """A pre-defined user reply for testing.

    Attributes:
        message: The user's message text
        runtime_policy: Optional runtime policy overrides
        delay_ms: Simulated delay before sending (for realistic timing)
        metadata: Additional metadata for test tracking
    """

    message: str
    runtime_policy: dict[str, Any] | None = None
    delay_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_request_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for ChatSendRequest."""
        kwargs: dict[str, Any] = {"message": self.message}
        if self.runtime_policy:
            kwargs["runtime_policy"] = self.runtime_policy
        return kwargs


class ScriptedUser:
    """A user that provides pre-defined replies in sequence.

    Usage:
        user = ScriptedUser([
            ScriptedReply("Hello"),
            ScriptedReply("I have 3-5 years experience"),
            ScriptedReply("Medium risk tolerance"),
        ])

        while (reply := user.next_reply()) is not None:
            # Send reply to orchestrator
            ...
    """

    def __init__(self, replies: list[ScriptedReply]) -> None:
        self._replies = list(replies)
        self._index = 0
        self._history: list[ScriptedReply] = []

    def next_reply(self) -> ScriptedReply | None:
        """Get the next reply, or None if exhausted."""
        if self._index >= len(self._replies):
            return None
        reply = self._replies[self._index]
        self._index += 1
        self._history.append(reply)
        return reply

    def peek_reply(self) -> ScriptedReply | None:
        """Peek at the next reply without consuming it."""
        if self._index >= len(self._replies):
            return None
        return self._replies[self._index]

    def inject_reply(self, reply: ScriptedReply) -> None:
        """Insert a reply at the current position.

        Useful for conditional branching based on AI responses.
        """
        self._replies.insert(self._index, reply)

    def inject_replies(self, replies: list[ScriptedReply]) -> None:
        """Insert multiple replies at the current position."""
        for i, reply in enumerate(replies):
            self._replies.insert(self._index + i, reply)

    def skip_to_end(self) -> None:
        """Skip all remaining replies."""
        self._index = len(self._replies)

    def reset(self) -> None:
        """Reset to the beginning."""
        self._index = 0
        self._history.clear()

    @property
    def remaining_count(self) -> int:
        """Number of replies remaining."""
        return max(0, len(self._replies) - self._index)

    @property
    def total_count(self) -> int:
        """Total number of replies."""
        return len(self._replies)

    @property
    def current_index(self) -> int:
        """Current position in the reply sequence."""
        return self._index

    @property
    def history(self) -> list[ScriptedReply]:
        """Replies that have been consumed."""
        return list(self._history)

    @property
    def is_exhausted(self) -> bool:
        """Whether all replies have been consumed."""
        return self._index >= len(self._replies)


# Type alias for decision functions
DecisionFn = Callable[["TurnObservation"], ScriptedReply | None]


class ConditionalUser:
    """A user that decides replies based on AI responses.

    Usage:
        def decide(turn: TurnObservation) -> ScriptedReply | None:
            if "risk tolerance" in turn.cleaned_text.lower():
                return ScriptedReply("Medium risk")
            if turn.phase == "pre_strategy":
                return ScriptedReply("US stocks")
            return None  # End conversation

        user = ConditionalUser(decide)
    """

    def __init__(
        self,
        decision_fn: DecisionFn,
        *,
        max_turns: int = 50,
        fallback_reply: ScriptedReply | None = None,
    ) -> None:
        """Initialize a conditional user.

        Args:
            decision_fn: Function that takes a TurnObservation and returns
                the next ScriptedReply, or None to end the conversation.
            max_turns: Maximum turns before auto-stopping (safety limit).
            fallback_reply: Reply to use if decision_fn raises an exception.
        """
        self._decision_fn = decision_fn
        self._max_turns = max_turns
        self._fallback_reply = fallback_reply
        self._turn_count = 0
        self._history: list[tuple[TurnObservation, ScriptedReply | None]] = []

    def decide_reply(self, last_turn: TurnObservation) -> ScriptedReply | None:
        """Decide the next reply based on the last turn.

        Returns None to signal end of conversation.
        """
        self._turn_count += 1

        if self._turn_count > self._max_turns:
            self._history.append((last_turn, None))
            return None

        try:
            reply = self._decision_fn(last_turn)
        except Exception:
            reply = self._fallback_reply

        self._history.append((last_turn, reply))
        return reply

    def reset(self) -> None:
        """Reset state for a new conversation."""
        self._turn_count = 0
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of decisions made."""
        return self._turn_count

    @property
    def history(self) -> list[tuple[TurnObservation, ScriptedReply | None]]:
        """History of (turn, reply) pairs."""
        return list(self._history)


class CompositeUser:
    """Combines scripted and conditional users.

    Starts with scripted replies, then switches to conditional logic
    when the script is exhausted.
    """

    def __init__(
        self,
        scripted: ScriptedUser,
        conditional: ConditionalUser | None = None,
    ) -> None:
        self._scripted = scripted
        self._conditional = conditional
        self._in_conditional_mode = False

    def get_reply(self, last_turn: TurnObservation | None = None) -> ScriptedReply | None:
        """Get the next reply.

        For the first turn (last_turn=None), uses scripted replies.
        After scripted replies are exhausted, switches to conditional mode.
        """
        if not self._in_conditional_mode:
            reply = self._scripted.next_reply()
            if reply is not None:
                return reply
            # Switch to conditional mode
            self._in_conditional_mode = True

        if self._conditional is None or last_turn is None:
            return None

        return self._conditional.decide_reply(last_turn)

    def reset(self) -> None:
        """Reset both users."""
        self._scripted.reset()
        if self._conditional:
            self._conditional.reset()
        self._in_conditional_mode = False

    @property
    def is_in_conditional_mode(self) -> bool:
        """Whether we've switched to conditional mode."""
        return self._in_conditional_mode
