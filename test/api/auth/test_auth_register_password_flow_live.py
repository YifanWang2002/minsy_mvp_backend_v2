from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def _register_user(
    api_test_client: TestClient,
    *,
    password: str,
) -> tuple[str, dict[str, object]]:
    email = f"pytest-auth-{uuid4().hex[:16]}@example.com"
    response = api_test_client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "name": f"pytest-auth-{uuid4().hex[:8]}",
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["user_id"]
    assert payload["access_token"]
    assert payload["refresh_token"]
    return email, payload


def _login(
    api_test_client: TestClient,
    *,
    email: str,
    password: str,
) -> dict[str, object]:
    response = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["access_token"]
    assert payload["refresh_token"]
    return payload


def test_000_accessibility_auth_register_and_login(
    api_test_client: TestClient,
) -> None:
    password = "pytest-pass-001"
    email, registered = _register_user(api_test_client, password=password)

    login_payload = _login(
        api_test_client,
        email=email,
        password=password,
    )
    assert str(login_payload["user_id"]) == str(registered["user_id"])

    me = api_test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login_payload['access_token']}"},
    )
    assert me.status_code == 200, me.text
    assert me.json()["email"] == email


def test_010_change_password_for_registered_user(
    api_test_client: TestClient,
) -> None:
    old_password = "pytest-pass-old"
    new_password = "pytest-pass-new"
    email, _ = _register_user(api_test_client, password=old_password)
    login_payload = _login(api_test_client, email=email, password=old_password)
    headers = {"Authorization": f"Bearer {login_payload['access_token']}"}

    changed = api_test_client.post(
        "/api/v1/auth/change-password",
        headers=headers,
        json={
            "current_password": old_password,
            "new_password": new_password,
        },
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["detail"] == "Password updated successfully."

    old_login = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": old_password},
    )
    assert old_login.status_code == 401, old_login.text

    new_login_payload = _login(
        api_test_client,
        email=email,
        password=new_password,
    )
    assert new_login_payload["access_token"]


def test_020_register_duplicate_email_conflict(
    api_test_client: TestClient,
) -> None:
    password = "pytest-pass-dup"
    email, _ = _register_user(api_test_client, password=password)

    duplicated = api_test_client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "name": "pytest-dup-user",
        },
    )
    assert duplicated.status_code == 409, duplicated.text
