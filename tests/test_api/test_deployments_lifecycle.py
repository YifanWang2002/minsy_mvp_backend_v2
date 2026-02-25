from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.engine.execution.alpaca_account_probe import AlpacaAccountProbeResult
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"deploy_lifecycle_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Deploy Tester"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _create_strategy_via_api(client: TestClient, headers: dict[str, str]) -> UUID:
    new_thread = client.post(
        "/api/v1/chat/new-thread",
        headers=headers,
        json={"metadata": {}},
    )
    assert new_thread.status_code == 201
    session_id = new_thread.json()["session_id"]

    strategy_payload = load_strategy_payload(EXAMPLE_PATH)
    confirm = client.post(
        "/api/v1/strategies/confirm",
        headers=headers,
        json={
            "session_id": session_id,
            "dsl_json": strategy_payload,
            "auto_start_backtest": False,
        },
    )
    assert confirm.status_code == 200
    return UUID(confirm.json()["strategy_id"])


@pytest.mark.parametrize("capital_allocated", [Decimal("0"), Decimal("1000")])
def test_deployments_lifecycle_and_permissions(
    monkeypatch: pytest.MonkeyPatch,
    capital_allocated: Decimal,
) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)

    with TestClient(app) as client:
        access_token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {access_token}"}

        broker_resp = client.post(
            "/api/v1/broker-accounts?validate=false",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "demo_key", "api_secret": "demo_secret"},
                "metadata": {"label": "primary"},
            },
        )
        assert broker_resp.status_code == 201
        broker_body = broker_resp.json()
        broker_account_id = UUID(broker_body["broker_account_id"])
        assert "credentials" not in broker_body

        strategy_id = _create_strategy_via_api(client, headers)

        create_resp = client.post(
            "/api/v1/deployments",
            headers=headers,
            json={
                "strategy_id": str(strategy_id),
                "broker_account_id": str(broker_account_id),
                "mode": "paper",
                "capital_allocated": str(capital_allocated),
                "risk_limits": {"max_position_pct": 0.2},
                "runtime_state": {"boot": True},
            },
        )
        assert create_resp.status_code == 201
        deployment_body = create_resp.json()
        deployment_id = UUID(deployment_body["deployment_id"])
        assert deployment_body["status"] == "pending"
        assert deployment_body["run"]["status"] == "stopped"

        start_resp = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
        assert start_resp.status_code == 200
        start_body = start_resp.json()
        assert start_body["deployment"]["status"] == "active"
        assert start_body["deployment"]["run"]["status"] == "starting"
        assert start_body["deployment"]["run"]["timeframe_seconds"] is not None
        assert start_body["deployment"]["run"]["timeframe_seconds"] > 0
        assert start_body["deployment"]["run"]["last_trigger_bucket"] is None
        assert start_body["deployment"]["run"]["last_enqueued_at"] is None

        pause_resp = client.post(f"/api/v1/deployments/{deployment_id}/pause", headers=headers)
        assert pause_resp.status_code == 200
        assert pause_resp.json()["deployment"]["status"] == "paused"
        assert pause_resp.json()["deployment"]["run"]["status"] == "paused"

        resume_resp = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
        assert resume_resp.status_code == 200
        assert resume_resp.json()["deployment"]["status"] == "active"

        stop_resp = client.post(f"/api/v1/deployments/{deployment_id}/stop", headers=headers)
        assert stop_resp.status_code == 200
        assert stop_resp.json()["deployment"]["status"] == "stopped"
        assert stop_resp.json()["deployment"]["run"]["status"] == "stopped"

        detail_resp = client.get(f"/api/v1/deployments/{deployment_id}", headers=headers)
        assert detail_resp.status_code == 200
        assert detail_resp.json()["deployment_id"] == str(deployment_id)
        assert detail_resp.json()["run"]["timeframe_seconds"] is not None

        list_resp = client.get("/api/v1/deployments", headers=headers)
        assert list_resp.status_code == 200
        assert any(item["deployment_id"] == str(deployment_id) for item in list_resp.json())

        orders_resp = client.get(f"/api/v1/deployments/{deployment_id}/orders", headers=headers)
        positions_resp = client.get(f"/api/v1/deployments/{deployment_id}/positions", headers=headers)
        pnl_resp = client.get(f"/api/v1/deployments/{deployment_id}/pnl", headers=headers)
        assert orders_resp.status_code == 200 and orders_resp.json() == []
        assert positions_resp.status_code == 200 and positions_resp.json() == []
        assert pnl_resp.status_code == 200 and pnl_resp.json() == []

        manual_resp = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-action",
            headers=headers,
            json={"action": "close", "payload": {"symbol": "AAPL"}},
        )
        assert manual_resp.status_code == 200
        assert manual_resp.json()["status"] == "rejected"
        assert manual_resp.json()["payload"]["_execution"]["reason"] == "deployment_stopped"

        live_mode_resp = client.post(
            "/api/v1/deployments",
            headers=headers,
            json={
                "strategy_id": str(strategy_id),
                "broker_account_id": str(broker_account_id),
                "mode": "live",
                "capital_allocated": "1000",
                "risk_limits": {},
                "runtime_state": {},
            },
        )
        assert live_mode_resp.status_code == 422

        other_token = _register_and_get_token(client)
        other_headers = {"Authorization": f"Bearer {other_token}"}
        forbidden_resp = client.get(f"/api/v1/deployments/{deployment_id}", headers=other_headers)
        assert forbidden_resp.status_code == 404


def test_create_deployment_auto_capital_from_probe_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_probe(_: dict[str, str]) -> AlpacaAccountProbeResult:
        return AlpacaAccountProbeResult(
            ok=True,
            status="paper_probe_ok",
            message="ok",
            metadata={
                "paper_http_status": 200,
                "live_http_status": 401,
                "paper_equity": "12345.67",
            },
        )

    monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _ok_probe)

    with TestClient(app) as client:
        access_token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {access_token}"}

        broker_resp = client.post(
            "/api/v1/broker-accounts",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "demo_key", "api_secret": "demo_secret"},
                "metadata": {"label": "primary"},
            },
        )
        assert broker_resp.status_code == 201
        broker_account_id = broker_resp.json()["broker_account_id"]

        strategy_id = _create_strategy_via_api(client, headers)
        create_resp = client.post(
            "/api/v1/deployments",
            headers=headers,
            json={
                "strategy_id": str(strategy_id),
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": "0",
                "risk_limits": {},
                "runtime_state": {},
            },
        )
        assert create_resp.status_code == 201
        assert create_resp.json()["capital_allocated"] == pytest.approx(12345.67)
