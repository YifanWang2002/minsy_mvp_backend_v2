from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_get_preferences_returns_defaults_when_not_persisted() -> None:
    email = f"prefs_default_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Prefs Default"},
        )
        access_token = register.json()["access_token"]

        response = client.get(
            "/api/v1/auth/preferences",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["theme_mode"] == "system"
    assert body["locale"] == "en"
    assert body["font_scale"] == "default"
    assert body["has_persisted"] is False
    assert body["updated_at"] is None


def test_put_preferences_upserts_and_get_returns_saved_values() -> None:
    email = f"prefs_put_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Prefs Put"},
        )
        access_token = register.json()["access_token"]

        put_response = client.put(
            "/api/v1/auth/preferences",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"theme_mode": "dark", "locale": "zh", "font_scale": "large"},
        )
        get_response = client.get(
            "/api/v1/auth/preferences",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    assert put_response.status_code == 200
    put_body = put_response.json()
    assert put_body["theme_mode"] == "dark"
    assert put_body["locale"] == "zh"
    assert put_body["font_scale"] == "large"
    assert put_body["has_persisted"] is True
    assert isinstance(put_body["updated_at"], str)
    assert put_body["updated_at"]

    assert get_response.status_code == 200
    get_body = get_response.json()
    assert get_body["theme_mode"] == "dark"
    assert get_body["locale"] == "zh"
    assert get_body["font_scale"] == "large"
    assert get_body["has_persisted"] is True
    assert isinstance(get_body["updated_at"], str)
    assert get_body["updated_at"]


def test_put_preferences_with_invalid_value_returns_422() -> None:
    email = f"prefs_invalid_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Prefs Invalid"},
        )
        access_token = register.json()["access_token"]

        response = client.put(
            "/api/v1/auth/preferences",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"theme_mode": "blue", "locale": "zh", "font_scale": "large"},
        )

    assert response.status_code == 422


def test_get_preferences_without_token_returns_401() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/auth/preferences")

    assert response.status_code == 401
