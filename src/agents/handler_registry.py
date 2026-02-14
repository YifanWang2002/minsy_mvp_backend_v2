"""Phase handler registry – central mapping from Phase -> PhaseHandler.

Import this module to get ``HANDLER_REGISTRY`` and ``get_handler()``.
New phases only need to be registered here; the orchestrator is untouched.
"""

from __future__ import annotations

from typing import Any

from src.agents.handler_protocol import PhaseHandler
from src.agents.handlers.deployment_handler import DeploymentHandler
from src.agents.handlers.kyc_handler import KYCHandler
from src.agents.handlers.pre_strategy_handler import PreStrategyHandler
from src.agents.handlers.strategy_handler import StrategyHandler
from src.agents.handlers.stress_test_handler import StressTestHandler
from src.agents.phases import Phase

# -- ordered list of phases that participate in the workflow session --------
# (COMPLETED and ERROR are terminal states, not handler phases)
WORKFLOW_PHASES: list[str] = [
    Phase.KYC.value,
    Phase.PRE_STRATEGY.value,
    Phase.STRATEGY.value,
    Phase.STRESS_TEST.value,
    Phase.DEPLOYMENT.value,
]

# -- singleton handler instances -------------------------------------------
_kyc_handler = KYCHandler()
_pre_strategy_handler = PreStrategyHandler()
_strategy_handler = StrategyHandler()
_stress_test_handler = StressTestHandler()
_deployment_handler = DeploymentHandler()

# -- registry dict ---------------------------------------------------------
HANDLER_REGISTRY: dict[str, PhaseHandler] = {
    Phase.KYC.value: _kyc_handler,
    Phase.PRE_STRATEGY.value: _pre_strategy_handler,
    Phase.STRATEGY.value: _strategy_handler,
    Phase.STRESS_TEST.value: _stress_test_handler,
    Phase.DEPLOYMENT.value: _deployment_handler,
}


def get_handler(phase: str) -> PhaseHandler | None:
    """Look up the handler for *phase*, or ``None`` if not registered."""
    return HANDLER_REGISTRY.get(phase)


def init_all_artifacts() -> dict[str, Any]:
    """Build a fresh artifacts dict with every phase's initial block.

    Returns a dict keyed by phase name, e.g.::

        {
            "kyc": {"profile": {}, "missing_fields": [...]},
            "pre_strategy": {"profile": {}, "missing_fields": [...]},
            ...
        }
    """
    return {
        phase: handler.init_artifacts()
        for phase, handler in HANDLER_REGISTRY.items()
    }
