from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from apps.api.agents.phases import Phase
from apps.api.orchestration import ChatOrchestrator


def _extract_trading_allowed_tools(policy: Any) -> set[str]:
    assert policy.allowed_tools is not None
    for tool in policy.allowed_tools:
        if tool.get("server_label") == "trading":
            return set(tool.get("allowed_tools", []))
    raise AssertionError("trading tool definition missing")


def test_deployment_preflight_runtime_policy_exposes_full_toolset() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    policy = orchestrator._build_phase_runtime_policy(
        phase=Phase.DEPLOYMENT.value,
        artifacts={
            Phase.DEPLOYMENT.value: {
                "profile": {
                    "deployment_status": "blocked",
                    "broker_readiness_status": "needs_choice",
                    "deployment_confirmation_status": "pending",
                },
            },
        },
    )

    allowed = _extract_trading_allowed_tools(policy)
    assert policy.phase_stage == "deployment_needs_broker_choice"
    assert "trading_check_deployment_readiness" in allowed
    assert "trading_create_builtin_sandbox_broker_account" in allowed
    assert "trading_create_paper_deployment" in allowed
    assert "trading_start_deployment" in allowed


def test_deployment_execute_runtime_policy_still_exposes_full_toolset_after_confirmation() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    policy = orchestrator._build_phase_runtime_policy(
        phase=Phase.DEPLOYMENT.value,
        artifacts={
            Phase.DEPLOYMENT.value: {
                "profile": {
                    "deployment_status": "ready",
                    "broker_readiness_status": "ready",
                    "selected_broker_account_id": "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8",
                    "deployment_confirmation_status": "confirmed",
                },
            },
        },
    )

    allowed = _extract_trading_allowed_tools(policy)
    assert policy.phase_stage == "deployment_execute_ready"
    assert "trading_check_deployment_readiness" in allowed
    assert "trading_create_paper_deployment" in allowed
    assert "trading_start_deployment" in allowed


async def test_deployment_auto_execute_flag_is_consumed_before_followup() -> None:
    class _FakeDb:
        def __init__(self) -> None:
            self.commit_calls = 0
            self.refresh_calls = 0

        async def commit(self) -> None:
            self.commit_calls += 1

        async def refresh(self, session: Any) -> None:
            del session
            self.refresh_calls += 1

    fake_db = _FakeDb()
    orchestrator = ChatOrchestrator(fake_db)  # type: ignore[arg-type]
    session = SimpleNamespace(
        current_phase=Phase.DEPLOYMENT.value,
        previous_response_id="resp_123",
        artifacts={
            Phase.DEPLOYMENT.value: {
                "runtime": {
                    "auto_execute_pending": True,
                }
            }
        },
    )

    consumed = await orchestrator._consume_deployment_auto_execute_flag(session=session)

    assert consumed is True
    assert (
        session.artifacts[Phase.DEPLOYMENT.value]["runtime"]["auto_execute_pending"]
        is False
    )
    assert session.previous_response_id is None
    assert fake_db.commit_calls == 1
    assert fake_db.refresh_calls == 1
