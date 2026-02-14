from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_register_success_returns_201_and_tokens() -> None:
    email = f"register_{uuid4().hex}@test.com"
    payload = {"email": email, "password": "pass1234", "name": "Register User"}

    with TestClient(app) as client:
        response = client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["user_id"]
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] > 0


def test_register_duplicate_email_returns_409() -> None:
    email = f"duplicate_{uuid4().hex}@test.com"
    payload = {"email": email, "password": "pass1234", "name": "Duplicate User"}

    with TestClient(app) as client:
        first = client.post("/api/v1/auth/register", json=payload)
        second = client.post("/api/v1/auth/register", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409
