from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_000_accessibility_unsupported_backtest_analysis_rejected(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/backtests/jobs/{uuid4()}/analysis/not_supported",
        headers=auth_headers,
    )
    assert response.status_code == 422, response.text


def test_010_missing_backtest_job_returns_404(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get(
        f"/api/v1/backtests/jobs/{uuid4()}/analysis/overview",
        headers=auth_headers,
    )
    assert response.status_code == 404, response.text
