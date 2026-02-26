from __future__ import annotations

from fastapi.testclient import TestClient


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-archive-flow"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def test_000_accessibility_archive_unarchive_flow(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)

    archive_response = api_test_client.post(
        f"/api/v1/sessions/{session_id}/archive",
        headers=auth_headers,
    )
    assert archive_response.status_code == 204, archive_response.text

    archived_list = api_test_client.get(
        "/api/v1/sessions",
        headers=auth_headers,
        params={"limit": 100, "archived": "true"},
    )
    assert archived_list.status_code == 200, archived_list.text
    archived_ids = {str(item["session_id"]) for item in archived_list.json()}
    assert session_id in archived_ids

    unarchive_response = api_test_client.post(
        f"/api/v1/sessions/{session_id}/unarchive",
        headers=auth_headers,
    )
    assert unarchive_response.status_code == 204, unarchive_response.text


def test_010_session_delete_flow(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)

    delete_response = api_test_client.delete(
        f"/api/v1/sessions/{session_id}",
        headers=auth_headers,
    )
    assert delete_response.status_code == 204, delete_response.text

    detail = api_test_client.get(
        f"/api/v1/sessions/{session_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 404, detail.text
