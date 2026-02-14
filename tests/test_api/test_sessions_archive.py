from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"sessions_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Session User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _create_session(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/chat/new-thread",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_archive_unarchive_and_delete_session() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        session_id = _create_session(client, token)
        auth_headers = {"Authorization": f"Bearer {token}"}

        active_before = client.get("/api/v1/sessions", headers=auth_headers)
        assert active_before.status_code == 200
        active_rows = active_before.json()
        assert len(active_rows) == 1
        assert active_rows[0]["session_id"] == session_id
        assert active_rows[0]["archived_at"] is None

        archive = client.post(f"/api/v1/sessions/{session_id}/archive", headers=auth_headers)
        assert archive.status_code == 204

        active_after_archive = client.get("/api/v1/sessions", headers=auth_headers)
        assert active_after_archive.status_code == 200
        assert active_after_archive.json() == []

        archived_list = client.get(
            "/api/v1/sessions",
            params={"archived": "true"},
            headers=auth_headers,
        )
        assert archived_list.status_code == 200
        archived_rows = archived_list.json()
        assert len(archived_rows) == 1
        assert archived_rows[0]["session_id"] == session_id
        assert archived_rows[0]["archived_at"] is not None

        unarchive = client.post(
            f"/api/v1/sessions/{session_id}/unarchive",
            headers=auth_headers,
        )
        assert unarchive.status_code == 204

        active_after_unarchive = client.get("/api/v1/sessions", headers=auth_headers)
        assert active_after_unarchive.status_code == 200
        active_rows_after_unarchive = active_after_unarchive.json()
        assert len(active_rows_after_unarchive) == 1
        assert active_rows_after_unarchive[0]["session_id"] == session_id
        assert active_rows_after_unarchive[0]["archived_at"] is None

        delete_response = client.delete(
            f"/api/v1/sessions/{session_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 204

        active_after_delete = client.get("/api/v1/sessions", headers=auth_headers)
        archived_after_delete = client.get(
            "/api/v1/sessions",
            params={"archived": "true"},
            headers=auth_headers,
        )
        assert active_after_delete.status_code == 200
        assert archived_after_delete.status_code == 200
        assert active_after_delete.json() == []
        assert archived_after_delete.json() == []
