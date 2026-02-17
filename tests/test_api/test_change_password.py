from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_change_password_success_and_new_password_login_works() -> None:
    email = f"change_password_ok_{uuid4().hex}@test.com"
    old_password = "pass1234"
    new_password = "newpass123"

    with TestClient(app) as client:
        register_resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": old_password, "name": "Password User"},
        )
        access_token = register_resp.json()["access_token"]

        change_resp = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": old_password, "new_password": new_password},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        old_login_resp = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": old_password},
        )
        new_login_resp = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": new_password},
        )

    assert register_resp.status_code == 201
    assert change_resp.status_code == 200
    assert change_resp.json()["detail"] == "Password updated successfully."
    assert old_login_resp.status_code == 401
    assert new_login_resp.status_code == 200


def test_change_password_with_wrong_current_password_returns_401() -> None:
    email = f"change_password_wrong_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register_resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Password User"},
        )
        access_token = register_resp.json()["access_token"]

        change_resp = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": "wrong1234", "new_password": "newpass123"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

    assert register_resp.status_code == 201
    assert change_resp.status_code == 401
    assert change_resp.json()["detail"] == "Current password is incorrect."


def test_change_password_without_token_returns_401() -> None:
    with TestClient(app) as client:
        change_resp = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": "pass1234", "new_password": "newpass123"},
        )

    assert change_resp.status_code == 401


def test_change_password_with_same_password_returns_400() -> None:
    email = f"change_password_same_{uuid4().hex}@test.com"
    password = "pass1234"

    with TestClient(app) as client:
        register_resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "name": "Password User"},
        )
        access_token = register_resp.json()["access_token"]

        change_resp = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": password, "new_password": password},
            headers={"Authorization": f"Bearer {access_token}"},
        )

    assert register_resp.status_code == 201
    assert change_resp.status_code == 400
    assert (
        change_resp.json()["detail"]
        == "New password must be different from current password."
    )
