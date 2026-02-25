from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.engine.execution.adapters.base import OhlcvBar, QuoteSnapshot
from src.engine.execution.deployment_lock import deployment_runtime_lock
from src.engine.execution.signal_store import signal_store
from src.engine.market_data.runtime import market_data_runtime
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"signal_order_flow_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Signal Flow User"},
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


def test_deployment_signal_to_order_flow(monkeypatch) -> None:
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
        first = process_once.json()
        assert first["execution_event_id"] is not None
        assert first["signal"] == "OPEN_LONG"
        assert first["order_id"] is not None
        assert first["idempotent_hit"] is False

        process_twice = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
        assert process_twice.status_code == 200
        second = process_twice.json()
        assert second["execution_event_id"] is not None
        assert second["signal"] == "NOOP"
        assert second["reason"] == "hold_long"
        assert second["order_id"] is None

        orders = client.get(f"/api/v1/deployments/{deployment_id}/orders", headers=headers)
        assert orders.status_code == 200
        assert len(orders.json()) == 1
        first_order = orders.json()[0]
        assert first_order["provider_status"] == "filled"
        assert first_order["reject_reason"] is None
        assert first_order["last_sync_at"] is not None
        order_metadata = first_order["metadata"]
        assert isinstance(order_metadata.get("state_transitions"), list)
        assert len(order_metadata["state_transitions"]) >= 2

        signals = client.get(
            f"/api/v1/deployments/{deployment_id}/signals",
            headers=headers,
            params={"limit": 1},
        )
        assert signals.status_code == 200
        first_page = signals.json()
        assert len(first_page) == 1
        cursor = first_page[0]["signal_event_id"]
        assert cursor is not None

        next_page = client.get(
            f"/api/v1/deployments/{deployment_id}/signals",
            headers=headers,
            params={"cursor": cursor, "limit": 5},
        )
        assert next_page.status_code == 200
        assert len(next_page.json()) >= 1


def test_process_now_records_event_when_deployment_locked(monkeypatch) -> None:
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

        lease = asyncio.run(deployment_runtime_lock.acquire(deployment_id))
        assert lease is not None
        try:
            process_once = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
            assert process_once.status_code == 200
            body = process_once.json()
            assert body["signal"] == "NOOP"
            assert body["reason"] == "deployment_locked"
            assert body["execution_event_id"] is not None
        finally:
            asyncio.run(deployment_runtime_lock.release(lease))


def test_process_now_hydrates_market_data_from_provider_when_runtime_cache_empty(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "market_data_memory_cache_enabled", True)
    monkeypatch.setattr(settings, "market_data_redis_read_enabled", False)
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", False)
    signal_store.clear()
    market_data_runtime.reset()

    async def fake_fetch_ohlcv(
        self,  # noqa: ANN001
        symbol: str,
        *,
        since=None,  # noqa: ANN001
        limit: int = 500,
    ) -> list[OhlcvBar]:
        _ = self, since, limit
        assert symbol == "AAPL"
        return [
            OhlcvBar(
                timestamp=datetime(2026, 1, 6, 9, 30, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100.5"),
                volume=Decimal("10"),
            ),
            OhlcvBar(
                timestamp=datetime(2026, 1, 6, 9, 31, tzinfo=UTC),
                open=Decimal("100.5"),
                high=Decimal("102"),
                low=Decimal("100"),
                close=Decimal("101.5"),
                volume=Decimal("10"),
            ),
        ]

    async def fake_fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:  # noqa: ANN001
        _ = self
        return QuoteSnapshot(
            symbol=symbol,
            bid=Decimal("101.4"),
            ask=Decimal("101.6"),
            last=Decimal("101.5"),
            timestamp=datetime(2026, 1, 6, 9, 31, tzinfo=UTC),
        )

    async def fake_fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:  # noqa: ANN001
        _ = self
        assert symbol == "AAPL"
        return OhlcvBar(
            timestamp=datetime(2026, 1, 6, 9, 32, tzinfo=UTC),
            open=Decimal("101.5"),
            high=Decimal("103"),
            low=Decimal("101"),
            close=Decimal("102.5"),
            volume=Decimal("8"),
        )

    monkeypatch.setattr(
        "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_ohlcv_1m",
        fake_fetch_ohlcv,
    )
    monkeypatch.setattr(
        "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_latest_quote",
        fake_fetch_latest_quote,
    )
    monkeypatch.setattr(
        "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_latest_1m_bar",
        fake_fetch_latest_1m_bar,
    )

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
        assert body["reason"] != "no_market_data"
        assert body["order_id"] is not None
        metadata = body.get("metadata") or {}
        assert metadata.get("market_data_fallback") == "hydrated"
        assert metadata.get("market_data_fallback_latest_1m_merged") == "yes"

        bars = market_data_runtime.get_recent_bars(
            market="stocks",
            symbol="AAPL",
            timeframe="1m",
            limit=10,
        )
        assert len(bars) >= 3
        assert bars[-1].timestamp == datetime(2026, 1, 6, 9, 32, tzinfo=UTC)


def test_process_now_fail_fast_when_redis_market_data_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "market_data_redis_read_enabled", True)
    monkeypatch.setattr(settings, "market_data_runtime_fail_fast_on_redis_error", True)
    signal_store.clear()
    market_data_runtime.reset()

    monkeypatch.setattr(
        market_data_runtime,
        "get_recent_bars",
        lambda **_: [],
    )
    monkeypatch.setattr(
        market_data_runtime,
        "redis_read_error_recent",
        lambda *_, **__: True,
    )
    monkeypatch.setattr(
        market_data_runtime,
        "redis_data_plane_status",
        lambda: {
            "enabled": True,
            "available": False,
            "market_data_store_ok": False,
            "subscription_store_ok": False,
            "last_error": {"operation": "get_recent_bars", "error_type": "ConnectionError"},
        },
    )

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
        assert body["reason"] == "market_data_redis_unavailable"


def test_noop_cycle_updates_position_mark_price_and_unrealized_pnl(monkeypatch) -> None:
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "market_data_memory_cache_enabled", True)
    monkeypatch.setattr(settings, "market_data_redis_read_enabled", False)
    monkeypatch.setattr(settings, "market_data_redis_write_enabled", False)
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

        first_cycle = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
        assert first_cycle.status_code == 200
        assert first_cycle.json()["signal"] == "OPEN_LONG"

        market_data_runtime.reset()

        async def fake_fetch_ohlcv(
            self,  # noqa: ANN001
            symbol: str,
            *,
            since=None,  # noqa: ANN001
            limit: int = 500,
        ) -> list[OhlcvBar]:
            _ = self, since, limit
            assert symbol == "AAPL"
            return [
                OhlcvBar(
                    timestamp=datetime(2026, 1, 6, 10, 0, tzinfo=UTC),
                    open=Decimal("92"),
                    high=Decimal("93"),
                    low=Decimal("90"),
                    close=Decimal("92"),
                    volume=Decimal("10"),
                ),
                OhlcvBar(
                    timestamp=datetime(2026, 1, 6, 10, 1, tzinfo=UTC),
                    open=Decimal("92"),
                    high=Decimal("92"),
                    low=Decimal("91"),
                    close=Decimal("91"),
                    volume=Decimal("10"),
                ),
            ]

        async def fake_fetch_latest_quote(self, symbol: str) -> QuoteSnapshot | None:  # noqa: ANN001
            _ = self
            return QuoteSnapshot(
                symbol=symbol,
                bid=Decimal("90.9"),
                ask=Decimal("91.1"),
                last=Decimal("91"),
                timestamp=datetime(2026, 1, 6, 10, 1, tzinfo=UTC),
            )

        async def fake_fetch_latest_1m_bar(self, symbol: str) -> OhlcvBar | None:  # noqa: ANN001
            _ = self
            assert symbol == "AAPL"
            return OhlcvBar(
                timestamp=datetime(2026, 1, 6, 10, 1, tzinfo=UTC),
                open=Decimal("92"),
                high=Decimal("92"),
                low=Decimal("91"),
                close=Decimal("91"),
                volume=Decimal("10"),
            )

        monkeypatch.setattr(
            "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_ohlcv_1m",
            fake_fetch_ohlcv,
        )
        monkeypatch.setattr(
            "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_latest_quote",
            fake_fetch_latest_quote,
        )
        monkeypatch.setattr(
            "src.engine.execution.adapters.alpaca_trading.AlpacaTradingAdapter.fetch_latest_1m_bar",
            fake_fetch_latest_1m_bar,
        )

        second_cycle = client.post(f"/api/v1/deployments/{deployment_id}/process-now", headers=headers)
        assert second_cycle.status_code == 200
        second_body = second_cycle.json()
        assert second_body["signal"] == "NOOP"
        assert second_body["reason"] == "hold_long"

        positions = client.get(f"/api/v1/deployments/{deployment_id}/positions", headers=headers)
        assert positions.status_code == 200
        rows = positions.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["mark_price"] == pytest.approx(91.0, rel=1e-6)
        assert row["unrealized_pnl"] == pytest.approx(-10.5, rel=1e-6)
