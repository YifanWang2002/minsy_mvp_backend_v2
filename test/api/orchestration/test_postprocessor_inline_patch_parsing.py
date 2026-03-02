from __future__ import annotations

from apps.api.orchestration import ChatOrchestrator


def test_extract_wrapped_payloads_parses_inline_agent_state_patch_assignment() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    raw = (
        "Deployment is ready.\n"
        'AGENT_STATE_PATCH {"deployment_status":"ready","deployment_confirmation_status":"confirmed"}\n'
    )

    cleaned, genui, patches = orchestrator._extract_wrapped_payloads(raw)

    assert genui == []
    assert patches == [
        {
            "deployment_status": "ready",
            "deployment_confirmation_status": "confirmed",
        }
    ]
    assert "AGENT_STATE_PATCH" not in cleaned
    assert "Deployment is ready." in cleaned


def test_extract_wrapped_payloads_parses_inline_agent_state_patch_equals_form() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    raw = (
        "Confirmed.\n"
        'AGENT_STATE_PATCH={"deployment_status":"blocked","broker_readiness_status":"ready","deployment_confirmation_status":"confirmed"}'
    )

    cleaned, _, patches = orchestrator._extract_wrapped_payloads(raw)

    assert patches == [
        {
            "deployment_status": "blocked",
            "broker_readiness_status": "ready",
            "deployment_confirmation_status": "confirmed",
        }
    ]
    assert cleaned.strip() == "Confirmed."
