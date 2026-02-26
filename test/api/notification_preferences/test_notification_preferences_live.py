from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _get_preferences(api_test_client: TestClient, auth_headers: dict[str, str]) -> dict[str, Any]:
    response = api_test_client.get(
        "/api/v1/notifications/preferences",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_000_accessibility_get_notification_preferences(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    payload = _get_preferences(api_test_client, auth_headers)
    assert isinstance(payload.get("telegram_enabled"), bool)


def test_010_update_notification_preferences_roundtrip(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    before = _get_preferences(api_test_client, auth_headers)

    update_payload = {
        "telegram_enabled": not bool(before["telegram_enabled"]),
        "position_opened_enabled": not bool(before["position_opened_enabled"]),
        "position_closed_enabled": not bool(before["position_closed_enabled"]),
    }
    updated = api_test_client.put(
        "/api/v1/notifications/preferences",
        headers=auth_headers,
        json=update_payload,
    )
    assert updated.status_code == 200, updated.text
    updated_payload = updated.json()
    for key, value in update_payload.items():
        assert updated_payload[key] == value

    restored = api_test_client.put(
        "/api/v1/notifications/preferences",
        headers=auth_headers,
        json={
            "telegram_enabled": before["telegram_enabled"],
            "position_opened_enabled": before["position_opened_enabled"],
            "position_closed_enabled": before["position_closed_enabled"],
        },
    )
    assert restored.status_code == 200, restored.text
