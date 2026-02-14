from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _assert_status(response, expected: int, step: str) -> None:
    if response.status_code != expected:
        raise AssertionError(
            f"{step}: expected {expected}, got {response.status_code}, body={response.text}"
        )


def test_auth_flow_script_equivalent() -> None:
    email = f"smoke_{uuid4().hex[:10]}@example.com"
    password = "123456"
    name = "Smoke User"

    with TestClient(app) as client:
        register_resp = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "name": name},
        )
        _assert_status(register_resp, 201, "register")
        register_json = register_resp.json()
        register_user_id = str(register_json.get("user_id") or "")

        login_resp = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        _assert_status(login_resp, 200, "login")
        login_json = login_resp.json()
        login_user_id = str(login_json.get("user_id") or "")
        access_token = str(login_json.get("access_token") or "")
        refresh_token = str(login_json.get("refresh_token") or "")

        assert register_user_id
        assert login_user_id == register_user_id
        assert access_token
        assert refresh_token

        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        _assert_status(me_resp, 200, "me")
        me_json = me_resp.json()
        assert str(me_json.get("user_id")) == login_user_id
        assert me_json.get("kyc_status") == "incomplete"

        refresh_resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        _assert_status(refresh_resp, 200, "refresh")
        refresh_json = refresh_resp.json()
        next_access = str(refresh_json.get("access_token") or "")
        next_refresh = str(refresh_json.get("refresh_token") or "")
        assert next_access
        assert next_refresh
        assert next_access != access_token
