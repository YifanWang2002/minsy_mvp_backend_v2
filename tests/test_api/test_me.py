from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_me_with_valid_token_returns_user_info() -> None:
    email = f"me_ok_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Me User"},
        )
        access_token = register.json()["access_token"]

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == email
    assert body["kyc_status"] == "incomplete"


def test_me_without_token_returns_401() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/auth/me")

    assert response.status_code == 401
