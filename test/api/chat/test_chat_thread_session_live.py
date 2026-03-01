from __future__ import annotations

from uuid import UUID

from fastapi.testclient import TestClient

from apps.api.agents.phases import Phase
from apps.api.orchestration.prompt_builder import PromptBuilderMixin


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-live"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def test_000_accessibility_new_thread_create(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)
    assert session_id


def test_010_session_list_and_detail_include_new_thread(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)

    listed = api_test_client.get(
        "/api/v1/sessions",
        headers=auth_headers,
        params={"limit": 100, "archived": "false"},
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()
    ids = {str(item["session_id"]) for item in items}
    assert session_id in ids

    detail = api_test_client.get(
        f"/api/v1/sessions/{session_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert str(payload["session_id"]) == session_id
    assert payload["current_phase"] in {
        "kyc",
        "pre_strategy",
        "strategy",
        "stress_test",
        "deployment",
        "completed",
        "error",
    }


class _DummyPromptBuilder(PromptBuilderMixin):
    @staticmethod
    def _coerce_uuid_text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return str(UUID(text))
        except ValueError:
            return None


def _snapshot_policy_tools(builder: _DummyPromptBuilder, *, phase: str, artifacts: dict) -> dict[str, tuple[str, ...]]:
    policy = builder._build_phase_runtime_policy(phase=phase, artifacts=artifacts)
    tool_defs = policy.allowed_tools or []
    return {
        str(item.get("server_label")): tuple(item.get("allowed_tools", []))
        for item in tool_defs
        if isinstance(item, dict)
    }


def test_020_phase_tool_exposure_snapshots() -> None:
    builder = _DummyPromptBuilder()

    pre_strategy_snapshot = _snapshot_policy_tools(
        builder,
        phase=Phase.PRE_STRATEGY.value,
        artifacts={},
    )
    assert "market_data" in pre_strategy_snapshot
    assert "check_symbol_available" in pre_strategy_snapshot["market_data"]
    assert "market_data_fetch_missing_ranges" in pre_strategy_snapshot["market_data"]

    strategy_snapshot = _snapshot_policy_tools(
        builder,
        phase=Phase.STRATEGY.value,
        artifacts={
            Phase.STRATEGY.value: {
                "profile": {
                    "strategy_id": "00000000-0000-4000-8000-000000000002",
                }
            }
        },
    )
    assert "strategy" in strategy_snapshot
    assert "market_data" in strategy_snapshot
    assert "backtest" in strategy_snapshot
    assert "strategy_patch_dsl" in strategy_snapshot["strategy"]
    assert "market_data_detect_missing_ranges" in strategy_snapshot["market_data"]
    assert "backtest_create_job" in strategy_snapshot["backtest"]

    deployment_snapshot = _snapshot_policy_tools(
        builder,
        phase=Phase.DEPLOYMENT.value,
        artifacts={
            Phase.DEPLOYMENT.value: {
                "profile": {
                    "deployment_status": "ready",
                }
            }
        },
    )
    assert "trading" in deployment_snapshot
    assert "trading_create_paper_deployment" in deployment_snapshot["trading"]
    assert "trading_list_deployments" in deployment_snapshot["trading"]
