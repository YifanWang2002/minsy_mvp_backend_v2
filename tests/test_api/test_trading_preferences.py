from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"trading_pref_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Trading Preference User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def test_get_and_update_trading_preferences() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        get_default = client.get("/api/v1/trading/preferences", headers=headers)
        assert get_default.status_code == 200
        default_payload = get_default.json()
        assert default_payload["execution_mode"] == "auto_execute"
        assert default_payload["approval_channel"] == "telegram"
        assert default_payload["approval_timeout_seconds"] == 120
        assert default_payload["approval_scope"] == "open_only"

        update = client.put(
            "/api/v1/trading/preferences",
            headers=headers,
            json={
                "execution_mode": "approval_required",
                "approval_channel": "telegram",
                "approval_timeout_seconds": 180,
                "approval_scope": "open_only",
            },
        )
        assert update.status_code == 200
        updated_payload = update.json()
        assert updated_payload["execution_mode"] == "approval_required"
        assert updated_payload["approval_timeout_seconds"] == 180

        get_updated = client.get("/api/v1/trading/preferences", headers=headers)
        assert get_updated.status_code == 200
        assert get_updated.json()["execution_mode"] == "approval_required"
