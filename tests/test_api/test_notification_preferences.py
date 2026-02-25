from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_access_token(client: TestClient) -> str:
    email = f"notify_pref_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Notify Pref User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def test_notification_preferences_get_and_update() -> None:
    with TestClient(app) as client:
        token = _register_and_get_access_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        get_default = client.get("/api/v1/notifications/preferences", headers=headers)
        assert get_default.status_code == 200
        body = get_default.json()
        assert body["telegram_enabled"] is True
        assert body["position_opened_enabled"] is True

        update = client.put(
            "/api/v1/notifications/preferences",
            headers=headers,
            json={
                "telegram_enabled": False,
                "position_opened_enabled": False,
                "execution_anomaly_enabled": False,
            },
        )
        assert update.status_code == 200
        updated = update.json()
        assert updated["telegram_enabled"] is False
        assert updated["position_opened_enabled"] is False
        assert updated["execution_anomaly_enabled"] is False

        get_after = client.get("/api/v1/notifications/preferences", headers=headers)
        assert get_after.status_code == 200
        persisted = get_after.json()
        assert persisted["telegram_enabled"] is False
        assert persisted["position_opened_enabled"] is False
