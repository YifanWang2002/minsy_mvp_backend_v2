from __future__ import annotations

from fastapi.testclient import TestClient

from test._support.deployment_runtime_harness import DeploymentRuntimeHarness


def test_000_deployment_runtime_harness_v1_live(
    api_test_client: TestClient,
) -> None:
    harness = DeploymentRuntimeHarness(api_test_client)
    report = harness.run_v1_rest_scenario()

    assert report.context.deployment_id
    assert report.db_before is not None
    assert report.db_after is not None
    assert report.db_after.deployment_status == "stopped"
    assert report.db_after.table_counts["manual_trade_actions"] >= 1
    assert report.db_after.table_counts["trading_event_outbox"] >= (
        report.db_before.table_counts["trading_event_outbox"]
    )
    assert report.report_json_path.exists()
    assert report.report_markdown_path.exists()


def test_010_deployment_runtime_harness_v1_1_live(
    api_test_client: TestClient,
) -> None:
    harness = DeploymentRuntimeHarness(api_test_client)
    report = harness.run_v1_1_chat_mcp_scenario()

    assert report.context.deployment_id
    assert report.db_after is not None
    assert report.db_after.deployment_status == "stopped"
    assert any(step.driver == "chat" for step in report.steps)
    assert any(step.driver == "mcp" for step in report.steps)
    assert report.report_json_path.exists()
    assert report.report_markdown_path.exists()


def test_020_deployment_runtime_harness_v1_2_live(
    api_test_client: TestClient,
) -> None:
    harness = DeploymentRuntimeHarness(api_test_client)
    report = harness.run_v1_2_approval_notification_scenario()

    assert report.context.deployment_id
    assert report.db_before is not None
    assert report.db_after is not None
    assert report.db_after.table_counts["trade_approval_requests"] >= 1
    assert report.db_after.table_counts["notification_outbox"] >= 1
    assert report.db_after.table_counts["social_connector_bindings"] >= 1
    assert any(step.name == "seed_trade_approval_request" for step in report.steps)
    assert any(step.name == "dispatch_pending_notifications" for step in report.steps)
    assert "approval_decision_path" in report.artifacts
    assert report.report_json_path.exists()
    assert report.report_markdown_path.exists()
