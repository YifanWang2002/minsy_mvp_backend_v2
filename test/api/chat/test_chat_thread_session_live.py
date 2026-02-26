from __future__ import annotations

from fastapi.testclient import TestClient


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-live"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def test_000_accessibility_new_thread_create(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)
    assert session_id


def test_010_session_list_and_detail_include_new_thread(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)

    listed = api_test_client.get(
        "/api/v1/sessions",
        headers=auth_headers,
        params={"limit": 100, "archived": "false"},
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()
    ids = {str(item["session_id"]) for item in items}
    assert session_id in ids

    detail = api_test_client.get(
        f"/api/v1/sessions/{session_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert str(payload["session_id"]) == session_id
    assert payload["current_phase"] in {
        "kyc",
        "pre_strategy",
        "strategy",
        "stress_test",
        "deployment",
        "completed",
        "error",
    }
