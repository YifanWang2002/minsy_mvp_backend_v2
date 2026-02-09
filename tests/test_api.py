"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Minsy API"
    assert data["status"] == "running"


def test_health(client):
    """Test health endpoint with system metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cpu" in data
    assert "memory" in data
    assert "gpu" in data


def test_api_v1_root(client):
    """Test API v1 root endpoint."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data


def test_api_v1_health(client):
    """Test API v1 health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_strategies_list(client):
    """Test strategies list endpoint."""
    response = client.get("/api/v1/strategies/")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data


def test_chat_list(client):
    """Test chat conversations list endpoint."""
    response = client.get("/api/v1/chat/")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
