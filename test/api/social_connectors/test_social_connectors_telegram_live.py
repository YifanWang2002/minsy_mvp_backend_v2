from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_telegram_test_chart_endpoint(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get(
        "/api/v1/social/connectors/telegram/test-webapp/chart",
        params={
            "symbol": "NASDAQ:AAPL",
            "interval": "1d",
            "locale": "en",
            "theme": "light",
            "signal_id": "pytest-live",
        },
    )
    assert response.status_code == 200, response.text
    assert "text/html" in response.headers.get("content-type", "")
    assert "<html" in response.text.lower()


def test_010_telegram_test_target_status(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/social/connectors/telegram/test-target",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "configured_email" in payload
    assert isinstance(payload.get("resolved_user_exists"), bool)
    assert isinstance(payload.get("resolved_binding_connected"), bool)
