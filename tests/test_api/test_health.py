from fastapi.testclient import TestClient

from src.main import app


def test_health_returns_ok_with_db_and_redis_true() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["db"] is True
    assert payload["redis"] is True
