from __future__ import annotations

from fastapi.testclient import TestClient


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-live-sessions"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def test_000_accessibility_sessions_list(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        "/api/v1/sessions",
        headers=auth_headers,
        params={"limit": 20, "archived": "false"},
    )
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)


def test_010_sessions_list_contains_newly_created_thread(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)
    response = api_test_client.get(
        "/api/v1/sessions",
        headers=auth_headers,
        params={"limit": 100, "archived": "false"},
    )
    assert response.status_code == 200, response.text
    ids = {str(item["session_id"]) for item in response.json()}
    assert session_id in ids
