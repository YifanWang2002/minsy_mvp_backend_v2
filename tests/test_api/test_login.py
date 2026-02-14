from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_login_correct_password_returns_200_and_tokens() -> None:
    email = f"login_ok_{uuid4().hex}@test.com"
    register_payload = {"email": email, "password": "pass1234", "name": "Login OK"}
    login_payload = {"email": email, "password": "pass1234"}

    with TestClient(app) as client:
        register_resp = client.post("/api/v1/auth/register", json=register_payload)
        login_resp = client.post("/api/v1/auth/login", json=login_payload)

    assert register_resp.status_code == 201
    assert login_resp.status_code == 200
    body = login_resp.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] > 0


def test_login_wrong_password_returns_401() -> None:
    email = f"login_fail_{uuid4().hex}@test.com"
    register_payload = {"email": email, "password": "pass1234", "name": "Login Fail"}
    wrong_login_payload = {"email": email, "password": "wrong-password"}

    with TestClient(app) as client:
        register_resp = client.post("/api/v1/auth/register", json=register_payload)
        login_resp = client.post("/api/v1/auth/login", json=wrong_login_payload)

    assert register_resp.status_code == 201
    assert login_resp.status_code == 401
