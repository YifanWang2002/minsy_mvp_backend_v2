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

    monkeypatch.setattr(health_router_module, "_build_chat_service_status", fake_chat_status)
    monkeypatch.setattr(
        health_router_module,
        "_build_backtest_service_status",
        fake_backtest_status,
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["services"]["api_endpoint"]["status"] == "healthy"
    assert payload["services"]["chat"]["status"] == "healthy"
    assert payload["services"]["backtest"]["status"] == "healthy"
    assert payload["services"]["live_trading"]["status"] == "down"
    assert isinstance(payload["response_time_ms"], float)
    assert isinstance(payload["timestamp"], str)
    assert payload["system"]["platform"]


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

    monkeypatch.setattr(health_router_module, "_build_chat_service_status", fake_chat_status)
    monkeypatch.setattr(
        health_router_module,
        "_build_backtest_service_status",
        fake_backtest_status,
    )

    app = _build_test_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "down"
    assert payload["services"]["chat"]["status"] == "down"
    assert payload["services"]["backtest"]["status"] == "degraded"
