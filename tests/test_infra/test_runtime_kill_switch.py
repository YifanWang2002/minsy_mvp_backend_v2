from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.config import settings
from src.engine.execution.adapters.base import OhlcvBar
from src.engine.execution.kill_switch import RuntimeKillSwitch
from src.engine.market_data.runtime import market_data_runtime
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"kill_switch_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Kill Switch User"},
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


def test_runtime_kill_switch_scopes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_users_csv", "user-a,user-b")
    monkeypatch.setattr(
        settings,
        "paper_trading_kill_switch_deployments_csv",
        "dep-a,dep-b",
    )

    kill_switch = RuntimeKillSwitch()
    assert kill_switch.evaluate(user_id="user-a", deployment_id="dep-x").allowed is False
    assert kill_switch.evaluate(user_id="user-x", deployment_id="dep-b").allowed is False
    assert kill_switch.evaluate(user_id="user-x", deployment_id="dep-x").allowed is True


def test_process_now_honors_global_kill_switch(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", True)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    market_data_runtime.reset()
    _seed_bars()

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

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
        body = process_once.json()
        assert body["signal"] == "NOOP"
        assert body["reason"] == "kill_switch_global"
