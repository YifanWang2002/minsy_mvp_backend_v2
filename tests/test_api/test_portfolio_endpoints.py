from __future__ import annotations

import re
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
    email = f"portfolio_endpoints_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Portfolio User"},
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


def test_portfolio_and_stream_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    signal_store.clear()
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

        portfolio = client.get(f"/api/v1/deployments/{deployment_id}/portfolio", headers=headers)
        assert portfolio.status_code == 200
        portfolio_body = portfolio.json()
        assert portfolio_body["deployment_id"] == str(deployment_id)
        assert "equity" in portfolio_body
        assert portfolio_body["positions"]
        assert portfolio_body["positions"][0]["symbol"] == "AAPL"

        fills = client.get(f"/api/v1/deployments/{deployment_id}/fills", headers=headers)
        assert fills.status_code == 200
        fills_body = fills.json()
        assert len(fills_body) == 1
        assert fills_body[0]["fill_qty"] == 1.0

        stream = client.get(
            f"/api/v1/stream/deployments/{deployment_id}",
            headers=headers,
            params={"max_events": 8, "poll_seconds": 0.2, "heartbeat_seconds": 0.2},
        )
        assert stream.status_code == 200
        assert "event: deployment_status" in stream.text
        assert "event: order_update" in stream.text
        assert "event: fill_update" in stream.text
        assert "event: position_update" in stream.text
        assert "event: pnl_update" in stream.text
        assert '"orders"' in stream.text
        assert '"fills"' in stream.text
        assert "event: stream_end" in stream.text

        event_ids = [int(match) for match in re.findall(r"id: (\d+)", stream.text)]
        assert event_ids
        latest_cursor = max(event_ids)

        resumed = client.get(
            f"/api/v1/stream/deployments/{deployment_id}",
            headers=headers,
            params={
                "cursor": latest_cursor,
                "max_events": 2,
                "poll_seconds": 0.2,
                "heartbeat_seconds": 0.2,
            },
        )
        assert resumed.status_code == 200
        resumed_ids = [int(match) for match in re.findall(r"id: (\d+)", resumed.text)]
        assert resumed_ids
        assert min(resumed_ids) > latest_cursor


def test_stream_emits_trade_approval_updates(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "trading_approval_enabled", True)
    signal_store.clear()
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

        pref = client.put(
            "/api/v1/trading/preferences",
            headers=headers,
            json={"execution_mode": "approval_required", "approval_timeout_seconds": 180},
        )
        assert pref.status_code == 200

        process_once = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
        assert process_once.status_code == 200
        process_body = process_once.json()
        assert process_body["reason"] == "approval_pending"

        stream = client.get(
            f"/api/v1/stream/deployments/{deployment_id}",
            headers=headers,
            params={"max_events": 10, "poll_seconds": 0.2, "heartbeat_seconds": 0.2},
        )
        assert stream.status_code == 200
        assert "event: trade_approval_update" in stream.text
        assert '"status": "pending"' in stream.text
