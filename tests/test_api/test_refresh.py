from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app


def test_valid_refresh_token_returns_new_pair() -> None:
    email = f"refresh_ok_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Refresh User"},
        )
        refresh_token = register.json()["refresh_token"]

        refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})

    assert register.status_code == 201
    assert refreshed.status_code == 200
    body = refreshed.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] > 0


def test_expired_refresh_token_returns_401() -> None:
    email = f"refresh_exp_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Refresh Expired"},
        )
        user_id = register.json()["user_id"]

        expired_refresh = jwt.encode(
            {
                "sub": user_id,
                "type": "refresh",
                "iat": int((datetime.now(UTC) - timedelta(days=10)).timestamp()),
                "exp": int((datetime.now(UTC) - timedelta(days=8)).timestamp()),
            },
            settings.secret_key,
            algorithm=settings.jwt_algorithm,
        )

        response = client.post("/api/v1/auth/refresh", json={"refresh_token": expired_refresh})

    assert response.status_code == 401
