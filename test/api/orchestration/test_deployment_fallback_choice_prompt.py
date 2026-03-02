from __future__ import annotations

from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.deployment_handler import DeploymentHandler
from apps.api.orchestration import ChatOrchestrator
from apps.api.agents.phases import Phase


def test_orchestrator_injects_deployment_confirmation_choice_prompt_when_missing() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    handler = DeploymentHandler()
    artifacts = {
        Phase.DEPLOYMENT.value: {
            "profile": {
                "strategy_name": "DOGEUSD 4h EMA Trend + RSI Filter (Swing)",
                "strategy_market": "crypto",
                "strategy_primary_symbol": "DOGEUSD",
                "strategy_timeframe": "4h",
                "broker_readiness_status": "ready",
                "selected_broker_account_id": "35a1bbd4-e53b-422b-a2cf-04923b69dbc4",
                "selected_broker_label": "Built-in Sandbox (paper)",
                "deployment_status": "ready",
                "deployment_confirmation_status": "pending",
            },
            "runtime": {},
            "missing_fields": ["deployment_confirmation_status"],
        }
    }
    ctx = PhaseContext(
        user_id=uuid4(),
        session_id=uuid4(),
        session_artifacts=artifacts,
        runtime_policy=RuntimePolicy(phase_stage="deployment_review_pending"),
    )

    payloads = orchestrator._ensure_required_choice_prompt_payload(
        handler=handler,
        ctx=ctx,
        missing_fields=["deployment_confirmation_status"],
        genui_payloads=[],
    )

    assert len(payloads) == 1
    prompt = payloads[0]
    assert prompt["type"] == "choice_prompt"
    assert prompt["choice_id"] == "deployment_confirmation_status"
    assert [option["id"] for option in prompt["options"]] == [
        "confirmed",
        "needs_changes",
    ]
