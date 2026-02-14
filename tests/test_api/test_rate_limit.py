import time
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api.routers.auth import me_rate_limiter
from src.main import app


def test_rate_limit_hits_429_on_31st_request_and_recovers_after_window() -> None:
    email = f"ratelimit_{uuid4().hex}@test.com"

    old_limit = me_rate_limiter.limit
    old_window = me_rate_limiter.window
    me_rate_limiter.limit = 30
    me_rate_limiter.window = 1

    try:
        with TestClient(app) as client:
            register = client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "pass1234", "name": "Rate User"},
            )
            access_token = register.json()["access_token"]
            headers = {"Authorization": f"Bearer {access_token}"}

            for _ in range(30):
                resp = client.get("/api/v1/auth/me", headers=headers)
                assert resp.status_code == 200

            blocked = client.get("/api/v1/auth/me", headers=headers)
            assert blocked.status_code == 429

            time.sleep(1.2)
            recovered = client.get("/api/v1/auth/me", headers=headers)
            assert recovered.status_code == 200
    finally:
        me_rate_limiter.limit = old_limit
        me_rate_limiter.window = old_window
