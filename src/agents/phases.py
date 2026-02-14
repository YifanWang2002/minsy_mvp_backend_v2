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


# Transition rules now support backward transitions:
#   - PRE_STRATEGY -> KYC  (user explicitly requests KYC redo)
#   - STRATEGY -> PRE_STRATEGY  (user wants to change market)
#   - STRESS_TEST -> STRATEGY  (AI discovers issues during testing)
VALID_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.KYC: {Phase.PRE_STRATEGY, Phase.ERROR},
    Phase.PRE_STRATEGY: {Phase.STRATEGY, Phase.KYC, Phase.ERROR},
    Phase.STRATEGY: {Phase.STRESS_TEST, Phase.PRE_STRATEGY, Phase.ERROR},
    Phase.STRESS_TEST: {Phase.DEPLOYMENT, Phase.STRATEGY, Phase.ERROR},
    Phase.DEPLOYMENT: {Phase.COMPLETED, Phase.ERROR},
    Phase.ERROR: {Phase.KYC, Phase.PRE_STRATEGY, Phase.STRATEGY, Phase.STRESS_TEST},
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
