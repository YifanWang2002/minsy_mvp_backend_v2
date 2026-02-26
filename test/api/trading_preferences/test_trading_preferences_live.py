from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _get_preferences(api_test_client: TestClient, auth_headers: dict[str, str]) -> dict[str, Any]:
    response = api_test_client.get(
        "/api/v1/trading/preferences",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_000_accessibility_get_trading_preferences(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    payload = _get_preferences(api_test_client, auth_headers)
    assert payload["execution_mode"] in {"auto_execute", "approval_required"}


def test_010_update_trading_preferences_roundtrip(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    before = _get_preferences(api_test_client, auth_headers)

    update_payload = {
        "execution_mode": "approval_required",
        "approval_channel": "telegram",
        "approval_timeout_seconds": 120,
        "approval_scope": "open_only",
    }
    updated = api_test_client.put(
        "/api/v1/trading/preferences",
        headers=auth_headers,
        json=update_payload,
    )
    assert updated.status_code == 200, updated.text
    updated_payload = updated.json()
    for key, value in update_payload.items():
        assert updated_payload[key] == value

    restored = api_test_client.put(
        "/api/v1/trading/preferences",
        headers=auth_headers,
        json={
            "execution_mode": before["execution_mode"],
            "approval_channel": before["approval_channel"],
            "approval_timeout_seconds": before["approval_timeout_seconds"],
            "approval_scope": before["approval_scope"],
        },
    )
    assert restored.status_code == 200, restored.text
