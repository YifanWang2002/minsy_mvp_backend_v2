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


def _register_and_get_token(client: TestClient, prefix: str) -> str:
    email = f"{prefix}_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Manual Action User"},
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
    for minute in range(2):
        market_data_runtime.ingest_1m_bar(
            market="stocks",
            symbol="AAPL",
            bar=OhlcvBar(
                timestamp=datetime(2026, 1, 5, 10, minute, tzinfo=UTC),
                open=Decimal(100 + minute),
                high=Decimal(101 + minute),
                low=Decimal(99 + minute),
                close=Decimal(100.5 + minute),
                volume=Decimal("10"),
            ),
        )


def _setup_deployment(client: TestClient, headers: dict[str, str]) -> UUID:
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

    create_deployment = client.post(
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
    assert create_deployment.status_code == 201
    deployment_id = UUID(create_deployment.json()["deployment_id"])

    start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
    assert start.status_code == 200

    process_once = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
    assert process_once.status_code == 200
    assert process_once.json()["signal"] == "OPEN_LONG"
    assert process_once.json()["execution_event_id"] is not None
    return deployment_id


def test_manual_trade_actions_execute_through_runtime(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    signal_store.clear()
    market_data_runtime.reset()
    _seed_bars()

    with TestClient(app) as client:
        token = _register_and_get_token(client, "manual_actions")
        headers = {"Authorization": f"Bearer {token}"}
        deployment_id = _setup_deployment(client, headers)

        rejected_reduce = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=headers,
            json={"action": "reduce", "payload": {"symbol": "AAPL"}},
        )
        assert rejected_reduce.status_code == 200
        assert rejected_reduce.json()["status"] == "rejected"
        assert rejected_reduce.json()["payload"]["_execution"]["reason"] == "qty_required_for_reduce"

        reduce_resp = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=headers,
            json={"action": "reduce", "payload": {"symbol": "AAPL", "qty": 0.4, "mark_price": 101}},
        )
        assert reduce_resp.status_code == 200
        reduce_body = reduce_resp.json()
        assert reduce_body["status"] == "completed"
        assert reduce_body["payload"]["_execution"]["signal"] == "CLOSE"

        positions_after_reduce = client.get(
            f"/api/v1/deployments/{deployment_id}/positions",
            headers=headers,
        )
        assert positions_after_reduce.status_code == 200
        assert positions_after_reduce.json()[0]["qty"] == 0.6

        close_resp = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=headers,
            json={"action": "close", "payload": {"symbol": "AAPL", "qty": 0.6, "mark_price": 99}},
        )
        assert close_resp.status_code == 200
        assert close_resp.json()["status"] == "completed"

        positions_after_close = client.get(
            f"/api/v1/deployments/{deployment_id}/positions",
            headers=headers,
        )
        assert positions_after_close.status_code == 200
        assert positions_after_close.json()[0]["qty"] == 0.0
        assert positions_after_close.json()[0]["side"] == "flat"


def test_manual_trade_action_permissions_and_validation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    signal_store.clear()
    market_data_runtime.reset()
    _seed_bars()

    with TestClient(app) as client:
        owner_token = _register_and_get_token(client, "manual_owner")
        owner_headers = {"Authorization": f"Bearer {owner_token}"}
        deployment_id = _setup_deployment(client, owner_headers)

        other_token = _register_and_get_token(client, "manual_other")
        other_headers = {"Authorization": f"Bearer {other_token}"}
        forbidden = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=other_headers,
            json={"action": "close", "payload": {"symbol": "AAPL"}},
        )
        assert forbidden.status_code == 404

        invalid_action = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=owner_headers,
            json={"action": "liquidate", "payload": {"symbol": "AAPL"}},
        )
        assert invalid_action.status_code == 422
