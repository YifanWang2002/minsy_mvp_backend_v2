from __future__ import annotations

from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, RuntimePolicy
from apps.api.agents.handlers.deployment_handler import DeploymentHandler
from apps.api.orchestration import ChatOrchestrator
from apps.api.agents.skills.deployment_skills import build_deployment_static_instructions


def test_deployment_instructions_describe_choice_selection_contract() -> None:
    instructions = build_deployment_static_instructions(language="en")

    assert "<CHOICE_SELECTION>" in instructions
    assert "choice_id=deployment_confirmation_status" in instructions
    assert "selected_option_id" in instructions


def test_orchestrator_stores_choice_selection_as_display_label() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    message = (
        '<CHOICE_SELECTION>'
        '{"choice_id":"deployment_confirmation_status",'
        '"selected_option_id":"confirmed",'
        '"selected_option_label":"Confirm deployment"}'
        "</CHOICE_SELECTION>"
    )

    stored = orchestrator._resolve_user_message_for_storage(message)

    assert stored == "Confirm deployment"


def test_deployment_prompt_allows_semantic_free_text_confirmation() -> None:
    handler = DeploymentHandler()
    ctx = PhaseContext(
        user_id=uuid4(),
        session_artifacts={
            "deployment": {
                "profile": {
                    "strategy_name": "BTC EMA Trend + ATR Risk (Swing)",
                    "strategy_market": "crypto",
                    "strategy_primary_symbol": "BTCUSD",
                    "strategy_timeframe": "4h",
                    "broker_readiness_status": "ready",
                    "selected_broker_account_id": "2305d002-3d86-4635-b8e9-380b4ff9afb2",
                    "deployment_status": "ready",
                    "deployment_confirmation_status": "pending",
                },
                "runtime": {},
                "missing_fields": ["deployment_confirmation_status"],
            }
        },
        runtime_policy=RuntimePolicy(phase_stage="deployment_review_pending"),
        turn_context={},
    )

    prompt = handler.build_prompt(ctx, "Confirm and deploy now")

    assert "structured_choice_selection_present: false" in prompt.enriched_input
    assert "semantically approves proceeding" in prompt.enriched_input
    assert "button click is one valid confirmation path, but it is not required" in prompt.enriched_input
