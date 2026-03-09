from __future__ import annotations

from fastapi.testclient import TestClient

from test._support.deployment_runtime_harness import DeploymentRuntimeHarness


def test_030_portfolio_realtime_resilience_live(
    api_test_client: TestClient,
    seeded_user_credentials: tuple[str, str],
) -> None:
    email, password = seeded_user_credentials
    harness = DeploymentRuntimeHarness(api_test_client)
    report = harness.run_v2_portfolio_realtime_resilience_scenario(
        email=email,
        password=password,
    )

    assert report.context.deployment_id
    assert report.db_before is not None
    assert report.db_after is not None
    assert report.db_after.deployment_status == "stopped"
    assert report.db_after.table_counts["manual_trade_actions"] >= (
        report.db_before.table_counts["manual_trade_actions"] + 2
    )
    assert report.artifacts.get("ws_fallback_mode_seen") is True
    assert report.artifacts.get("ws_recovered_mode_seen") is True
    assert report.artifacts.get("ws_observed_event_count", 0) > 0
    assert report.report_json_path.exists()
    assert report.report_markdown_path.exists()
