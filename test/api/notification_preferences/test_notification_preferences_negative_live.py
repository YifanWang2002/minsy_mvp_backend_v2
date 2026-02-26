from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_notification_preferences_requires_auth(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/api/v1/notifications/preferences")
    assert response.status_code == 401


def test_010_invalid_notification_preferences_payload_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.put(
        "/api/v1/notifications/preferences",
        headers=auth_headers,
        json={"telegram_enabled": "not_bool"},
    )
    assert response.status_code == 422, response.text
