"""Orchestrator test harness — comprehensive testing infrastructure.

This module provides tools for testing the ChatOrchestrator with full observability:
- TurnObservation / ConversationObservation: Capture all AI interaction data
- ObservableChatOrchestrator: Instrumented orchestrator wrapper
- ScriptedUser / ConditionalUser: Simulate user replies
- OrchestratorTestRunner: Coordinate multi-turn conversations
- TestReporter: Generate readable test reports
"""

# Core types (no dependencies)
from .observation_types import (
    ConversationObservation,
    TurnObservation,
)

# Scripted user (depends on types for TYPE_CHECKING only)
from .scripted_user import ConditionalUser, ScriptedReply, ScriptedUser

# Observer (depends on types)
from .observer import TurnObserver

# Reporter (depends on types)
from .reporter import ObservationReporter

# Alias for backwards compatibility
TestReporter = ObservationReporter

# Lazy imports for components with heavy dependencies
# These are imported on first use to avoid import errors when
# the full backend environment is not available

def __getattr__(name: str):
    """Lazy import for heavy components."""
    if name == "ObservableChatOrchestrator":
        from .observable_orchestrator import ObservableChatOrchestrator
        return ObservableChatOrchestrator
    if name == "OrchestratorTestRunner":
        from .test_runner import OrchestratorTestRunner
        return OrchestratorTestRunner
    if name == "QuickTestRunner":
        from .test_runner import QuickTestRunner
        return QuickTestRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConversationObservation",
    "TurnObservation",
    "TurnObserver",
    "ScriptedReply",
    "ScriptedUser",
    "ConditionalUser",
    "ObservableChatOrchestrator",
    "OrchestratorTestRunner",
    "QuickTestRunner",
    "ObservationReporter",
    "TestReporter",  # Alias for backwards compatibility
]
