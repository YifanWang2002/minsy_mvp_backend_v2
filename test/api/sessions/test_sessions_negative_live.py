from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_session_detail_missing_returns_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/sessions/{uuid4()}",
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_010_session_archive_and_delete_missing_returns_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    missing_id = uuid4()
    archive = api_test_client.post(
        f"/api/v1/sessions/{missing_id}/archive",
        headers=auth_headers,
    )
    assert archive.status_code == 404

    unarchive = api_test_client.post(
        f"/api/v1/sessions/{missing_id}/unarchive",
        headers=auth_headers,
    )
    assert unarchive.status_code == 404

    delete = api_test_client.delete(
        f"/api/v1/sessions/{missing_id}",
        headers=auth_headers,
    )
    assert delete.status_code == 404
