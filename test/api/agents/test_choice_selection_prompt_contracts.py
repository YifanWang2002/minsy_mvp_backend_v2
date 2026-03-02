from __future__ import annotations

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
