from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app


def test_token_format_is_jwt_like() -> None:
    email = f"jwt_format_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "JWT User"},
        )

    assert register.status_code == 201
    token = register.json()["access_token"]
    assert token.count(".") == 2


def test_expired_token_returns_401() -> None:
    email = f"jwt_expired_{uuid4().hex}@test.com"

    with TestClient(app) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "pass1234", "name": "Expired User"},
        )
        user_id = register.json()["user_id"]

        expired_payload = {
            "sub": user_id,
            "type": "access",
            "iat": int((datetime.now(UTC) - timedelta(days=2)).timestamp()),
            "exp": int((datetime.now(UTC) - timedelta(days=1)).timestamp()),
        }
        expired_token = jwt.encode(
            expired_payload,
            settings.secret_key,
            algorithm=settings.jwt_algorithm,
        )

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

    assert response.status_code == 401


def test_forged_token_returns_401() -> None:
    forged = jwt.encode(
        {
            "sub": str(uuid4()),
            "type": "access",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        },
        "wrong-secret-that-is-long-enough-for-hs256-testing",
        algorithm=settings.jwt_algorithm,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {forged}"},
        )

    assert response.status_code == 401
