from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import health as health_router_module


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router_module.router, prefix="/api/v1")
    return app


def test_status_returns_frontend_contract(monkeypatch) -> None:
    async def fake_chat_status() -> dict[str, str]:
        return {
            "status": "healthy",
            "details": "OpenAI models endpoint is reachable.",
        }

    async def fake_backtest_status() -> dict[str, str]:
        return {
            "status": "healthy",
            "details": "Redis ping ok. Celery worker ping ok (1 worker(s)).",
        }

    async def fake_flower_status() -> dict[str, str]:
        return {
            "status": "healthy",
            "details": "Flower endpoint responded with HTTP 401.",
        }

    async def fake_system_metrics() -> dict[str, float | str]:
        return {
            "platform": "TestOS-1.0",
            "cpu_usage_percent": 23.4,
            "memory_usage_percent": 56.7,
            "memory_available_gb": 3.21,
        }

    async def fake_live_trading_status() -> dict[str, str]:
        return {
            "status": "down",
            "details": "Paper trading is disabled by PAPER_TRADING_ENABLED=false.",
        }

    monkeypatch.setattr(health_router_module, "_build_chat_service_status", fake_chat_status)
    monkeypatch.setattr(
        health_router_module,
        "_build_backtest_service_status",
        fake_backtest_status,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_flower_service_status",
        fake_flower_status,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_system_metrics",
        fake_system_metrics,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_live_trading_service_status",
        fake_live_trading_status,
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "down"
    assert payload["services"]["api_endpoint"]["status"] == "healthy"
    assert payload["services"]["chat"]["status"] == "healthy"
    assert payload["services"]["backtest"]["status"] == "healthy"
    assert payload["services"]["flower"]["status"] == "healthy"
    assert payload["services"]["live_trading"]["status"] == "down"
    assert isinstance(payload["response_time_ms"], float)
    assert isinstance(payload["timestamp"], str)
    assert payload["system"]["platform"] == "TestOS-1.0"
    assert payload["system"]["cpu_usage_percent"] == 23.4
    assert payload["system"]["memory_usage_percent"] == 56.7
    assert payload["system"]["memory_available_gb"] == 3.21


def test_status_marks_overall_down_when_core_service_down(monkeypatch) -> None:
    async def fake_chat_status() -> dict[str, str]:
        return {
            "status": "down",
            "details": "OpenAI rejected API key (HTTP 401).",
        }

    async def fake_backtest_status() -> dict[str, str]:
        return {
            "status": "degraded",
            "details": "Redis ping ok. No Celery worker responded to ping.",
        }

    async def fake_flower_status() -> dict[str, str]:
        return {
            "status": "down",
            "details": "Flower endpoint is unreachable.",
        }

    async def fake_system_metrics() -> dict[str, float | str]:
        return {
            "platform": "TestOS-2.0",
            "cpu_usage_percent": 77.7,
            "memory_usage_percent": 88.8,
            "memory_available_gb": 1.11,
        }

    async def fake_live_trading_status() -> dict[str, str]:
        return {
            "status": "degraded",
            "details": "market_data_fresh=no; paper_worker_up=yes; broker_circuit=closed.",
        }

    monkeypatch.setattr(health_router_module, "_build_chat_service_status", fake_chat_status)
    monkeypatch.setattr(
        health_router_module,
        "_build_backtest_service_status",
        fake_backtest_status,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_flower_service_status",
        fake_flower_status,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_system_metrics",
        fake_system_metrics,
    )
    monkeypatch.setattr(
        health_router_module,
        "_build_live_trading_service_status",
        fake_live_trading_status,
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "down"
    assert payload["services"]["chat"]["status"] == "down"
    assert payload["services"]["backtest"]["status"] == "degraded"
    assert payload["services"]["flower"]["status"] == "down"
    assert payload["system"]["platform"] == "TestOS-2.0"
