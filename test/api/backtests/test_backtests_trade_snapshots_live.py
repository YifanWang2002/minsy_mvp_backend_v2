from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_backtests_trade_snapshots_requires_existing_job(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.post(
        "/api/v1/backtests/jobs/00000000-0000-0000-0000-000000000001/trade-snapshots",
        headers=auth_headers,
        json={
            "selection_mode": "latest",
            "selection_count": 1,
            "lookback_bars": 30,
            "lookforward_bars": 30,
            "render_images": False,
        },
    )
    assert response.status_code == 404, response.text


def test_010_backtests_trade_snapshots_contract_with_demo_job(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    demo_response = api_test_client.post(
        "/api/v1/backtests/jobs/demo/ensure",
        headers=auth_headers,
    )
    if demo_response.status_code != 200:
        assert demo_response.status_code in {404, 409}, demo_response.text
        return

    demo_payload = demo_response.json()
    job_id = demo_payload["job_id"]
    response = api_test_client.post(
        f"/api/v1/backtests/jobs/{job_id}/trade-snapshots",
        headers=auth_headers,
        json={
            "selection_mode": "latest",
            "selection_count": 2,
            "lookback_bars": 40,
            "lookforward_bars": 20,
            "render_images": False,
        },
    )

    if response.status_code != 200:
        assert response.status_code in {409, 422}, response.text
        detail = response.json().get("detail")
        assert isinstance(detail, dict)
        assert detail.get("code") in {
            "BACKTEST_JOB_NOT_READY",
            "BACKTEST_TRADE_SNAPSHOT_INVALID_INPUT",
        }
        return

    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["strategy_id"]
    assert payload["status"] == "done"
    assert payload["strategy_payload_source"] in {
        "strategy_revision",
        "current_strategy_fallback",
    }
    assert payload["selection"]["mode"] == "latest"
    assert payload["selection"]["selected_count"] <= 2
    assert payload["window"]["lookback_bars"] == 40
    assert payload["window"]["lookforward_bars"] == 20
    assert isinstance(payload["snapshots"], list)
    assert isinstance(payload["warnings"], list)

    if payload["snapshots"]:
        first = payload["snapshots"][0]
        assert "trade" in first
        assert "slice" in first
        assert "candles" in first["slice"]
        assert "indicators" in first["slice"]
