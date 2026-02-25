from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import health as health_router_module
from src.config import settings
from src.engine.execution.circuit_breaker import CircuitBreakerSnapshot


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router_module.router, prefix="/api/v1")
    return app


class _FakeRuntime:
    def __init__(self, checkpoints: dict[str, int]) -> None:
        self._checkpoints = checkpoints

    def checkpoints(self) -> dict[str, int]:
        return dict(self._checkpoints)


class _FakeBreaker:
    def __init__(self, state: str) -> None:
        self._state = state

    async def snapshot(self) -> CircuitBreakerSnapshot:
        return CircuitBreakerSnapshot(
            name="broker_request",
            state=self._state,  # type: ignore[arg-type]
            consecutive_failures=0,
            opened_until=None,
        )


def _patch_non_live_dependencies(monkeypatch) -> None:
    async def fake_chat_status() -> dict[str, str]:
        return {"status": "healthy", "details": "ok"}

    async def fake_backtest_status() -> dict[str, str]:
        return {"status": "healthy", "details": "ok"}

    async def fake_flower_status() -> dict[str, str]:
        return {"status": "healthy", "details": "ok"}

    async def fake_system_metrics() -> dict[str, float | str]:
        return {
            "platform": "test-os",
            "cpu_usage_percent": 10.0,
            "memory_usage_percent": 20.0,
            "memory_available_gb": 8.0,
        }

    monkeypatch.setattr(health_router_module, "_build_chat_service_status", fake_chat_status)
    monkeypatch.setattr(
        health_router_module,
        "_build_backtest_service_status",
        fake_backtest_status,
    )
    monkeypatch.setattr(health_router_module, "_build_flower_service_status", fake_flower_status)
    monkeypatch.setattr(health_router_module, "_build_system_metrics", fake_system_metrics)


def test_status_live_trading_probe_healthy(monkeypatch) -> None:
    _patch_non_live_dependencies(monkeypatch)
    monkeypatch.setattr(settings, "paper_trading_enabled", True)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", True)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_users_csv", "")
    monkeypatch.setattr(settings, "paper_trading_kill_switch_deployments_csv", "")
    monkeypatch.setattr(settings, "paper_trading_runtime_health_stale_seconds", 120)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    monkeypatch.setattr(
        health_router_module,
        "market_data_runtime",
        _FakeRuntime({"stocks:AAPL:1m": now_ms}),
    )
    monkeypatch.setattr(
        health_router_module,
        "_inspect_celery_workers",
        lambda: {"worker@paper": {"ok": "pong"}},
    )
    monkeypatch.setattr(
        health_router_module,
        "get_broker_request_circuit_breaker",
        lambda: _FakeBreaker("closed"),
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["services"]["live_trading"]["status"] == "healthy"
    assert "order_execution=broker" in payload["services"]["live_trading"]["details"]
    assert payload["status"] == "healthy"


def test_status_live_trading_probe_down_by_global_kill_switch(monkeypatch) -> None:
    _patch_non_live_dependencies(monkeypatch)
    monkeypatch.setattr(settings, "paper_trading_enabled", True)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", True)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", True)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_users_csv", "")
    monkeypatch.setattr(settings, "paper_trading_kill_switch_deployments_csv", "")

    monkeypatch.setattr(
        health_router_module,
        "market_data_runtime",
        _FakeRuntime({}),
    )
    monkeypatch.setattr(
        health_router_module,
        "_inspect_celery_workers",
        lambda: {"worker@paper": {"ok": "pong"}},
    )
    monkeypatch.setattr(
        health_router_module,
        "get_broker_request_circuit_breaker",
        lambda: _FakeBreaker("closed"),
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["services"]["live_trading"]["status"] == "down"
    assert "kill switch" in payload["services"]["live_trading"]["details"].lower()
    assert payload["status"] == "down"


def test_status_live_trading_probe_down_by_stale_data_and_no_worker(monkeypatch) -> None:
    _patch_non_live_dependencies(monkeypatch)
    monkeypatch.setattr(settings, "paper_trading_enabled", True)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", True)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_users_csv", "")
    monkeypatch.setattr(settings, "paper_trading_kill_switch_deployments_csv", "")
    monkeypatch.setattr(settings, "paper_trading_runtime_health_stale_seconds", 60)

    old_ms = int((datetime.now(UTC) - timedelta(hours=1)).timestamp() * 1000)
    monkeypatch.setattr(
        health_router_module,
        "market_data_runtime",
        _FakeRuntime({"stocks:AAPL:1m": old_ms}),
    )
    monkeypatch.setattr(health_router_module, "_inspect_celery_workers", lambda: None)
    monkeypatch.setattr(
        health_router_module,
        "get_broker_request_circuit_breaker",
        lambda: _FakeBreaker("closed"),
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["services"]["live_trading"]["status"] == "down"
    assert payload["status"] == "down"


def test_status_live_trading_probe_degraded_when_order_execution_simulated(monkeypatch) -> None:
    _patch_non_live_dependencies(monkeypatch)
    monkeypatch.setattr(settings, "paper_trading_enabled", True)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_users_csv", "")
    monkeypatch.setattr(settings, "paper_trading_kill_switch_deployments_csv", "")
    monkeypatch.setattr(settings, "paper_trading_runtime_health_stale_seconds", 120)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    monkeypatch.setattr(
        health_router_module,
        "market_data_runtime",
        _FakeRuntime({"stocks:AAPL:1m": now_ms}),
    )
    monkeypatch.setattr(
        health_router_module,
        "_inspect_celery_workers",
        lambda: {"worker@paper": {"ok": "pong"}},
    )
    monkeypatch.setattr(
        health_router_module,
        "get_broker_request_circuit_breaker",
        lambda: _FakeBreaker("closed"),
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["services"]["live_trading"]["status"] == "degraded"
    assert "order_execution=simulated" in payload["services"]["live_trading"]["details"]
    assert payload["status"] == "degraded"
