from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.config import settings
from src.engine.execution.adapters.base import OhlcvBar
from src.engine.execution.signal_store import signal_store
from src.engine.market_data.runtime import market_data_runtime
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"trade_approval_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Trade Approval User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _strategy_payload() -> dict:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["timeframe"] = "1m"
    payload["universe"] = {"market": "stocks", "tickers": ["AAPL"]}
    payload["trade"]["long"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}
    }
    payload["trade"]["short"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}
    }
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "exit_long",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
        }
    ]
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "exit_short",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
        }
    ]
    return payload


def _seed_bars() -> None:
    market_data_runtime.reset()
    for minute in range(2):
        market_data_runtime.ingest_1m_bar(
            market="stocks",
            symbol="AAPL",
            bar=OhlcvBar(
                timestamp=datetime(2026, 1, 8, 10, minute, tzinfo=UTC),
                open=Decimal(100 + minute),
                high=Decimal(101 + minute),
                low=Decimal(99 + minute),
                close=Decimal(100.5 + minute),
                volume=Decimal("10"),
            ),
        )


def _bootstrap_deployment(client: TestClient, *, headers: dict[str, str]) -> UUID:
    broker = client.post(
        "/api/v1/broker-accounts?validate=false",
        headers=headers,
        json={
            "provider": "alpaca",
            "mode": "paper",
            "credentials": {"api_key": "demo", "api_secret": "demo"},
            "metadata": {},
        },
    )
    assert broker.status_code == 201
    broker_id = broker.json()["broker_account_id"]

    new_thread = client.post("/api/v1/chat/new-thread", headers=headers, json={"metadata": {}})
    assert new_thread.status_code == 201
    session_id = new_thread.json()["session_id"]

    confirm = client.post(
        "/api/v1/strategies/confirm",
        headers=headers,
        json={"session_id": session_id, "dsl_json": _strategy_payload(), "auto_start_backtest": False},
    )
    assert confirm.status_code == 200
    strategy_id = confirm.json()["strategy_id"]

    created = client.post(
        "/api/v1/deployments",
        headers=headers,
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_id,
            "mode": "paper",
            "capital_allocated": "10000",
            "risk_limits": {"order_qty": 1, "max_position_notional": 5000},
            "runtime_state": {},
        },
    )
    assert created.status_code == 201
    deployment_id = UUID(created.json()["deployment_id"])

    start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
    assert start.status_code == 200
    return deployment_id


def _enable_approval_mode(client: TestClient, *, headers: dict[str, str]) -> None:
    response = client.put(
        "/api/v1/trading/preferences",
        headers=headers,
        json={
            "execution_mode": "approval_required",
            "approval_channel": "telegram",
            "approval_timeout_seconds": 180,
            "approval_scope": "open_only",
        },
    )
    assert response.status_code == 200


def _create_pending_approval(client: TestClient, *, headers: dict[str, str], deployment_id: UUID) -> str:
    process = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
    assert process.status_code == 200
    payload = process.json()
    assert payload["signal"] == "NOOP"
    assert payload["reason"] == "approval_pending"
    approval_id = payload["metadata"]["approval_request_id"]
    assert isinstance(approval_id, str)
    return approval_id


def test_process_now_creates_pending_trade_approval(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "trading_approval_enabled", True)
    signal_store.clear()
    _seed_bars()

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        deployment_id = _bootstrap_deployment(client, headers=headers)
        _enable_approval_mode(client, headers=headers)
        approval_id = _create_pending_approval(client, headers=headers, deployment_id=deployment_id)

        approvals = client.get("/api/v1/trade-approvals?status=pending", headers=headers)
        assert approvals.status_code == 200
        rows = approvals.json()
        assert len(rows) == 1
        assert rows[0]["trade_approval_request_id"] == approval_id

        orders = client.get(f"/api/v1/deployments/{deployment_id}/orders", headers=headers)
        assert orders.status_code == 200
        assert orders.json() == []


def test_api_can_approve_trade_approval_and_enqueue_execution(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "trading_approval_enabled", True)
    monkeypatch.setattr(
        "src.api.routers.trade_approvals.enqueue_execute_approved_open",
        lambda *_: "task-approval-1",
    )
    signal_store.clear()
    _seed_bars()

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        deployment_id = _bootstrap_deployment(client, headers=headers)
        _enable_approval_mode(client, headers=headers)
        approval_id = _create_pending_approval(client, headers=headers, deployment_id=deployment_id)

        approved = client.post(
            f"/api/v1/trade-approvals/{approval_id}/approve",
            headers=headers,
            json={"note": "looks good"},
        )
        assert approved.status_code == 200
        payload = approved.json()
        assert payload["status"] == "approved"
        assert payload["approved_via"] == "api"
        assert payload["metadata"]["execution_task_id"] == "task-approval-1"


def test_api_can_reject_trade_approval(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "trading_approval_enabled", True)
    signal_store.clear()
    _seed_bars()

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        deployment_id = _bootstrap_deployment(client, headers=headers)
        _enable_approval_mode(client, headers=headers)
        approval_id = _create_pending_approval(client, headers=headers, deployment_id=deployment_id)

        rejected = client.post(
            f"/api/v1/trade-approvals/{approval_id}/reject",
            headers=headers,
            json={"note": "skip this one"},
        )
        assert rejected.status_code == 200
        payload = rejected.json()
        assert payload["status"] == "rejected"
        assert payload["approved_via"] == "api"
