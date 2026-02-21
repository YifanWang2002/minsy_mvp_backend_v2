from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api.middleware.sentry_http_status import (
    SentryHTTPStatusMiddleware,
    should_capture_http_status_event,
)
from src.config import settings


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SentryHTTPStatusMiddleware)

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/forbidden")
    async def forbidden() -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": "forbidden"})

    return app


def test_should_capture_http_status_event_respects_bounds_and_exclusions(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sentry_http_status_capture_enabled", True, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_min_code", 400, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_max_code", 599, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_exclude_paths", ["/api/v1/health"], raising=False)

    assert should_capture_http_status_event(status_code=404, path="/missing", method="GET") is True
    assert (
        should_capture_http_status_event(
            status_code=503,
            path="/api/v1/health",
            method="GET",
        )
        is False
    )
    assert should_capture_http_status_event(status_code=200, path="/ok", method="GET") is False
    assert should_capture_http_status_event(status_code=404, path="/missing", method="OPTIONS") is False


def test_middleware_captures_403_with_request_context(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sentry_http_status_capture_enabled", True, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_min_code", 400, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_max_code", 599, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_exclude_paths", [], raising=False)

    calls: list[dict[str, Any]] = []

    def _capture(message: str, **kwargs: Any) -> None:
        calls.append({"message": message, **kwargs})

    monkeypatch.setattr(
        "src.api.middleware.sentry_http_status.capture_message_with_context",
        _capture,
    )

    client = TestClient(_build_test_app())
    response = client.get(
        "/forbidden?session_id=session_123",
        headers={
            "user-agent": "pytest-agent",
            "x-request-id": "req_abc",
        },
    )

    assert response.status_code == 403
    assert len(calls) == 1

    event = calls[0]
    assert event["message"] == "FastAPI returned HTTP 403"
    assert event["level"] == "warning"
    assert event["tags"]["source"] == "fastapi_http_status"
    assert event["tags"]["status_code"] == 403
    assert event["tags"]["status_family"] == "4xx"
    assert event["tags"]["route"] == "/forbidden"
    assert event["extras"]["path"] == "/forbidden"
    assert event["extras"]["query_params"]["session_id"] == "session_123"
    assert event["extras"]["request_id"] == "req_abc"


def test_middleware_does_not_capture_200(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sentry_http_status_capture_enabled", True, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_min_code", 400, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_max_code", 599, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_exclude_paths", [], raising=False)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "src.api.middleware.sentry_http_status.capture_message_with_context",
        lambda message, **kwargs: calls.append({"message": message, **kwargs}),
    )

    client = TestClient(_build_test_app())
    response = client.get("/ok")

    assert response.status_code == 200
    assert calls == []


def test_middleware_captures_404_for_missing_route(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sentry_http_status_capture_enabled", True, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_min_code", 400, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_max_code", 599, raising=False)
    monkeypatch.setattr(settings, "sentry_http_status_exclude_paths", [], raising=False)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "src.api.middleware.sentry_http_status.capture_message_with_context",
        lambda message, **kwargs: calls.append({"message": message, **kwargs}),
    )

    client = TestClient(_build_test_app())
    response = client.get("/not-found")

    assert response.status_code == 404
    assert len(calls) == 1
    assert calls[0]["tags"]["status_code"] == 404
    assert calls[0]["extras"]["path"] == "/not-found"
