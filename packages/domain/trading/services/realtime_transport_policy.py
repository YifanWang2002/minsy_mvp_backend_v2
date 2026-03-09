"""Pure policy object for WS realtime transport mode switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RealtimeTransportMode = Literal["pubsub", "polling_fallback"]


@dataclass(slots=True)
class RealtimeTransportState:
    """Mutable runtime state for one WS connection transport loop."""

    mode: RealtimeTransportMode
    consecutive_pubsub_failures: int = 0
    next_pubsub_probe_at: float = 0.0
    last_reconcile_at: float = 0.0
    last_fallback_poll_at: float = 0.0
    last_error: str | None = None


class RealtimeTransportPolicy:
    """Connection-local transport mode policy.

    - `pubsub` is the preferred realtime path.
    - On pubsub failures we switch to `polling_fallback`.
    - While in fallback we periodically probe pubsub and switch back on success.
    """

    def __init__(
        self,
        *,
        fallback_poll_seconds: float,
        reconcile_seconds: float,
        pubsub_probe_base_seconds: float,
        pubsub_probe_max_seconds: float,
    ) -> None:
        self._fallback_poll_seconds = max(0.1, float(fallback_poll_seconds))
        self._reconcile_seconds = max(0.2, float(reconcile_seconds))
        self._probe_base_seconds = max(0.2, float(pubsub_probe_base_seconds))
        self._probe_max_seconds = max(self._probe_base_seconds, float(pubsub_probe_max_seconds))

    def initial_state(self, *, pubsub_available: bool, now: float) -> RealtimeTransportState:
        if pubsub_available:
            return RealtimeTransportState(
                mode="pubsub",
                next_pubsub_probe_at=now,
                last_reconcile_at=now,
                last_fallback_poll_at=now,
            )
        return RealtimeTransportState(
            mode="polling_fallback",
            consecutive_pubsub_failures=1,
            next_pubsub_probe_at=now + self._probe_max_seconds,
            last_reconcile_at=now,
            last_fallback_poll_at=now,
            last_error="pubsub_unavailable",
        )

    def should_poll_fallback(self, state: RealtimeTransportState, *, now: float) -> bool:
        return state.mode == "polling_fallback" and (
            now - state.last_fallback_poll_at >= self._fallback_poll_seconds
        )

    def mark_fallback_poll(self, state: RealtimeTransportState, *, now: float) -> None:
        state.last_fallback_poll_at = now

    def should_reconcile(self, state: RealtimeTransportState, *, now: float) -> bool:
        return now - state.last_reconcile_at >= self._reconcile_seconds

    def mark_reconcile(self, state: RealtimeTransportState, *, now: float) -> None:
        state.last_reconcile_at = now

    def should_probe_pubsub(self, state: RealtimeTransportState, *, now: float) -> bool:
        return state.mode == "polling_fallback" and now >= state.next_pubsub_probe_at

    def record_pubsub_success(self, state: RealtimeTransportState, *, now: float) -> bool:
        changed = state.mode != "pubsub"
        state.mode = "pubsub"
        state.consecutive_pubsub_failures = 0
        state.next_pubsub_probe_at = now + self._probe_base_seconds
        state.last_error = None
        return changed

    def record_pubsub_failure(
        self,
        state: RealtimeTransportState,
        *,
        now: float,
        error: str,
    ) -> bool:
        changed = state.mode != "polling_fallback"
        state.mode = "polling_fallback"
        state.consecutive_pubsub_failures = max(1, state.consecutive_pubsub_failures + 1)
        backoff = min(
            self._probe_base_seconds * (2 ** (state.consecutive_pubsub_failures - 1)),
            self._probe_max_seconds,
        )
        state.next_pubsub_probe_at = now + backoff
        state.last_error = error
        return changed
