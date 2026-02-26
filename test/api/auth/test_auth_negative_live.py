from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_invalid_password_rejected(
    api_test_client: TestClient,
    seeded_user_credentials: tuple[str, str],
) -> None:
    email, _ = seeded_user_credentials
    response = api_test_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "wrong-password"},
    )
    assert response.status_code == 401, response.text


def test_010_protected_endpoint_requires_authorization(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.get("/api/v1/auth/me")
    assert response.status_code == 401, response.text


def test_020_refresh_rejects_invalid_token(
    api_test_client: TestClient,
) -> None:
    response = api_test_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "not-a-valid-token"},
    )
    assert response.status_code == 401, response.text
