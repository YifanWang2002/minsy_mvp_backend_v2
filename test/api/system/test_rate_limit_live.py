from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from packages.shared_settings.schema.settings import settings


def _register_user_and_get_headers(api_test_client: TestClient) -> dict[str, str]:
    password = "pytest-rate-limit-pass"
    email = f"pytest-rate-limit-{uuid4().hex[:16]}@example.com"
    response = api_test_client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "name": f"pytest-rate-limit-{uuid4().hex[:8]}",
        },
    )
    assert response.status_code == 201, response.text
    access_token = response.json()["access_token"]
    return {"Authorization": f"Bearer {access_token}"}


def test_000_accessibility_auth_me_rate_limit_hits_429(
    api_test_client: TestClient,
) -> None:
    auth_headers = _register_user_and_get_headers(api_test_client)
    limit = max(1, int(settings.auth_rate_limit))
    attempts = limit + 3

    status_codes: list[int] = []
    first_429_index: int | None = None
    for idx in range(attempts):
        response = api_test_client.get("/api/v1/auth/me", headers=auth_headers)
        status_codes.append(response.status_code)
        if response.status_code == 429:
            first_429_index = idx
            break

    assert first_429_index is not None, status_codes
    assert any(code == 200 for code in status_codes[: first_429_index + 1]), status_codes
