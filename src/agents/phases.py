"""Conversation phase enums and transition rules."""

from __future__ import annotations

from enum import StrEnum


class Phase(StrEnum):
    """Supported orchestrator phases."""

    KYC = "kyc"
    PRE_STRATEGY = "pre_strategy"
    STRATEGY = "strategy"
    STRESS_TEST = "stress_test"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    ERROR = "error"


class SessionStatus(StrEnum):
    """Workflow session status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


# Current product boundary:
#   - Performance-driven iteration stays inside STRATEGY.
#   - STRATEGY does not transition into STRESS_TEST yet (reserved for future
#     scenario stress tools such as crisis-window tests / Monte Carlo / etc.).
# Backward transitions still support:
#   - PRE_STRATEGY -> KYC  (user explicitly requests KYC redo)
#   - STRATEGY -> PRE_STRATEGY  (user wants to change market)
#   - STRESS_TEST -> STRATEGY  (legacy sessions only)
VALID_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.KYC: {Phase.PRE_STRATEGY, Phase.ERROR},
    Phase.PRE_STRATEGY: {Phase.STRATEGY, Phase.KYC, Phase.ERROR},
    Phase.STRATEGY: {Phase.PRE_STRATEGY, Phase.ERROR},
    Phase.STRESS_TEST: {Phase.STRATEGY, Phase.ERROR},
    Phase.DEPLOYMENT: {Phase.COMPLETED, Phase.ERROR},
    Phase.ERROR: {Phase.KYC, Phase.PRE_STRATEGY, Phase.STRATEGY},
    Phase.COMPLETED: set(),
}


def can_transition(from_phase: str, to_phase: str) -> bool:
    """Return True if a phase transition is valid."""
    try:
        source = Phase(from_phase)
        target = Phase(to_phase)
    except ValueError:
        return False
    return target in VALID_TRANSITIONS[source]
