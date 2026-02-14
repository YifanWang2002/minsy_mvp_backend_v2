from fastapi.testclient import TestClient

from src.main import create_app
from src.models import database as db_module
from src.models import redis as redis_module


def test_app_lifespan_startup_shutdown_no_leak() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert db_module.engine is not None
        assert db_module.session_factory is not None
        assert redis_module.redis_client is not None
        assert redis_module.redis_pool is not None

    assert db_module.engine is None
    assert db_module.session_factory is None
    assert redis_module.redis_client is None
    assert redis_module.redis_pool is None
