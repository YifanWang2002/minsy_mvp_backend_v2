from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_auth_login(
    api_test_client: TestClient,
    seeded_user_credentials: tuple[str, str],
) -> None:
    email, password = seeded_user_credentials
    response = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["user_id"]
    assert payload["access_token"]
    assert payload["refresh_token"]


def test_010_auth_me_uses_real_seed_user(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    seeded_user_credentials: tuple[str, str],
) -> None:
    expected_email, _ = seeded_user_credentials
    response = api_test_client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["email"] == expected_email
    assert payload["user_id"]


def test_020_auth_refresh_returns_new_access_token(
    api_test_client: TestClient,
    seeded_user_credentials: tuple[str, str],
) -> None:
    email, password = seeded_user_credentials
    login = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200, login.text
    login_payload = login.json()

    refreshed = api_test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": login_payload["refresh_token"]},
    )
    assert refreshed.status_code == 200, refreshed.text
    refreshed_payload = refreshed.json()
    assert refreshed_payload["access_token"]
    assert refreshed_payload["refresh_token"]
    assert refreshed_payload["access_token"] != login_payload["access_token"]
