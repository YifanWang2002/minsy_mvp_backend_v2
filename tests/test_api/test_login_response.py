from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_login_response_contains_kyc_status() -> None:
    email = f"login_kyc_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "KYC User"},
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "pass1234"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "user" in body
    assert body["user"]["kyc_status"] == "incomplete"
