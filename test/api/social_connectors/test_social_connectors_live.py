from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_social_connectors(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/social/connectors",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_010_social_connectors_telegram_activities(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/social/connectors/telegram/activities",
        headers=auth_headers,
        params={"limit": 20},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["provider"] == "telegram"
    assert isinstance(payload.get("items"), list)
