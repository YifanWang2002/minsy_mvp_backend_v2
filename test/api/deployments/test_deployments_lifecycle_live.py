from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_EXAMPLE_STRATEGY_PATH = (
    _BACKEND_DIR / "packages" / "domain" / "strategy" / "assets" / "example_strategy.json"
)


def _load_example_dsl() -> dict[str, object]:
    return json.loads(_EXAMPLE_STRATEGY_PATH.read_text(encoding="utf-8"))


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-deployment-live"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _create_strategy(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    session_id = _create_thread(api_test_client, auth_headers)
    dsl = _load_example_dsl()
    dsl["strategy"]["name"] = f"Pytest Deploy {uuid4().hex[:8]}"  # type: ignore[index]

    response = api_test_client.post(
        "/api/v1/strategies/confirm",
        headers=auth_headers,
        json={
            "session_id": session_id,
            "dsl_json": dsl,
            "auto_start_backtest": False,
            "language": "en",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["strategy_id"])


def _create_broker_account(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    tag = uuid4().hex[:10]
    response = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        params={"validate": "false"},
        json={
            "provider": "alpaca",
            "mode": "paper",
            "credentials": {
                "api_key": f"pytest-deploy-key-{tag}",
                "api_secret": f"pytest-deploy-secret-{tag}",
            },
            "metadata": {"source": "pytest-deployment-live", "tag": tag},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["broker_account_id"])


def _create_deployment(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> str:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_broker_account(api_test_client, auth_headers)

    response = api_test_client.post(
        "/api/v1/deployments",
        headers=auth_headers,
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "mode": "paper",
            "capital_allocated": 0,
            "risk_limits": {"order_qty": 1},
            "runtime_state": {"source": "pytest-live"},
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["deployment_id"]
    return str(payload["deployment_id"])


def test_000_accessibility_deployment_create(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    deployment_id = _create_deployment(api_test_client, auth_headers)

    detail = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert str(payload["deployment_id"]) == deployment_id
    assert payload["mode"] == "paper"
    assert payload["status"] in {"pending", "active", "paused", "stopped", "error"}


def test_010_deployment_start_pause_and_manual_stop_alias(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    deployment_id = _create_deployment(api_test_client, auth_headers)

    started = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert started.status_code == 200, started.text
    started_payload = started.json()
    assert started_payload["deployment"]["status"] == "active"

    paused = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/pause",
        headers=auth_headers,
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["deployment"]["status"] == "paused"

    resumed = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["deployment"]["status"] == "active"

    manual_stop = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/manual-actions",
        headers=auth_headers,
        json={"action": "stop", "payload": {"reason": "pytest-live"}},
    )
    assert manual_stop.status_code == 200, manual_stop.text
    manual_payload = manual_stop.json()
    assert str(manual_payload["deployment_id"]) == deployment_id
    assert manual_payload["action"] == "stop"
    assert manual_payload["status"] in {"completed", "pending", "rejected"}

    detail = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["status"] == "stopped"
