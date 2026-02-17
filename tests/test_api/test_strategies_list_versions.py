"""API tests for strategy list/version/diff endpoints."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.main import app
from src.models import database as db_module
from src.models.backtest import BacktestJob


def _register_and_get_token(client: TestClient) -> str:
    email = f"strategy_list_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Strategy List User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _create_thread(client: TestClient, *, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/chat/new-thread",
        headers=headers,
        json={"metadata": {}},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def _get_user_id(client: TestClient, *, headers: dict[str, str]) -> str:
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200
    user_id = response.json().get("user_id")
    assert isinstance(user_id, str) and user_id
    return user_id


def _confirm_strategy(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
    dsl_json: dict,
    strategy_id: str | None = None,
) -> dict:
    payload: dict[str, object] = {
        "session_id": session_id,
        "dsl_json": dsl_json,
        "auto_start_backtest": False,
        "language": "en",
    }
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id
    response = client.post("/api/v1/strategies/confirm", headers=headers, json=payload)
    assert response.status_code == 200
    return response.json()


def _insert_completed_backtest_job(
    client: TestClient,
    *,
    strategy_id: str,
    user_id: str,
    session_id: str,
    strategy_version: int | None,
    total_return_pct: float,
    max_drawdown_pct: float,
    sharpe: float,
) -> str:
    async def _insert() -> str:
        assert db_module.AsyncSessionLocal is not None
        now = datetime.now(UTC)
        async with db_module.AsyncSessionLocal() as db:
            job = BacktestJob(
                strategy_id=UUID(strategy_id),
                user_id=UUID(user_id),
                session_id=UUID(session_id),
                status="completed",
                progress=100,
                current_step="done",
                config=(
                    {"strategy_version": strategy_version}
                    if strategy_version is not None
                    else {}
                ),
                results={
                    "summary": {
                        "total_return_pct": total_return_pct,
                        "max_drawdown_pct": max_drawdown_pct,
                    },
                    "performance": {"metrics": {"sharpe": sharpe}},
                    "equity_curve": [
                        {"timestamp": "2024-01-01T00:00:00+00:00", "equity": 100000.0},
                        {"timestamp": "2024-01-02T00:00:00+00:00", "equity": 100500.0},
                        {"timestamp": "2024-01-03T00:00:00+00:00", "equity": 101000.0},
                    ],
                },
                submitted_at=now,
                completed_at=now,
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
            return str(job.id)

    return client.portal.call(_insert)


def test_list_strategies_returns_current_user_rows_with_latest_backtest() -> None:
    with TestClient(app) as client:
        token_user_a = _register_and_get_token(client)
        token_user_b = _register_and_get_token(client)
        headers_a = {"Authorization": f"Bearer {token_user_a}"}
        headers_b = {"Authorization": f"Bearer {token_user_b}"}

        session_a = _create_thread(client, headers=headers_a)
        session_b = _create_thread(client, headers=headers_b)
        user_a = _get_user_id(client, headers=headers_a)
        user_b = _get_user_id(client, headers=headers_b)

        payload_a = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload_a["strategy"]["description"] = "user-a-strategy"
        confirm_a = _confirm_strategy(
            client,
            headers=headers_a,
            session_id=session_a,
            dsl_json=payload_a,
        )

        payload_b = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload_b["strategy"]["description"] = "user-b-strategy"
        confirm_b = _confirm_strategy(
            client,
            headers=headers_b,
            session_id=session_b,
            dsl_json=payload_b,
        )

        _insert_completed_backtest_job(
            client,
            strategy_id=confirm_a["strategy_id"],
            user_id=user_a,
            session_id=session_a,
            strategy_version=1,
            total_return_pct=12.34,
            max_drawdown_pct=-4.56,
            sharpe=1.23,
        )
        _insert_completed_backtest_job(
            client,
            strategy_id=confirm_b["strategy_id"],
            user_id=user_b,
            session_id=session_b,
            strategy_version=1,
            total_return_pct=99.9,
            max_drawdown_pct=-1.0,
            sharpe=9.9,
        )

        listed = client.get("/api/v1/strategies", headers=headers_a)
        assert listed.status_code == 200
        rows = listed.json()
        assert isinstance(rows, list)
        assert len(rows) == 1
        item = rows[0]
        assert item["strategy_id"] == confirm_a["strategy_id"]
        assert item["metadata"]["strategy_name"]
        assert item["latest_backtest"]["status"] == "done"
        assert item["latest_backtest"]["strategy_version"] == 1
        assert item["latest_backtest"]["total_return_pct"] == 12.34
        assert item["latest_backtest"]["max_drawdown_pct"] == -4.56
        assert item["latest_backtest"]["sharpe_ratio"] == 1.23
        assert len(item["latest_backtest"]["equity_curve"]) == 3


def test_strategy_versions_include_version_specific_backtest_summary() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, headers=headers)
        user_id = _get_user_id(client, headers=headers)

        payload_v1 = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload_v1["strategy"]["description"] = "desc-v1"
        confirm_v1 = _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v1,
        )
        strategy_id = confirm_v1["strategy_id"]

        payload_v2 = deepcopy(payload_v1)
        payload_v2["strategy"]["description"] = "desc-v2"
        payload_v2["timeframe"] = "4h"
        _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v2,
            strategy_id=strategy_id,
        )

        _insert_completed_backtest_job(
            client,
            strategy_id=strategy_id,
            user_id=user_id,
            session_id=session_id,
            strategy_version=1,
            total_return_pct=10.0,
            max_drawdown_pct=-3.0,
            sharpe=1.0,
        )
        _insert_completed_backtest_job(
            client,
            strategy_id=strategy_id,
            user_id=user_id,
            session_id=session_id,
            strategy_version=2,
            total_return_pct=20.0,
            max_drawdown_pct=-2.0,
            sharpe=2.0,
        )

        response = client.get(
            f"/api/v1/strategies/{strategy_id}/versions",
            headers=headers,
        )
        assert response.status_code == 200
        versions = response.json()
        assert isinstance(versions, list)
        assert len(versions) >= 2

        by_version = {item["version"]: item for item in versions}
        assert 1 in by_version
        assert 2 in by_version
        assert by_version[1]["dsl_json"]["strategy"]["description"] == "desc-v1"
        assert by_version[2]["dsl_json"]["strategy"]["description"] == "desc-v2"
        assert by_version[1]["backtest"]["total_return_pct"] == 10.0
        assert by_version[1]["backtest"]["sharpe_ratio"] == 1.0
        assert by_version[2]["backtest"]["total_return_pct"] == 20.0
        assert by_version[2]["backtest"]["sharpe_ratio"] == 2.0


def test_strategy_diff_returns_display_items_with_old_and_new_values() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, headers=headers)

        payload_v1 = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload_v1["strategy"]["description"] = "initial-description"
        confirm_v1 = _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v1,
        )
        strategy_id = confirm_v1["strategy_id"]

        payload_v2 = deepcopy(payload_v1)
        payload_v2["strategy"]["description"] = "next-description"
        payload_v2["timeframe"] = "4h"
        _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v2,
            strategy_id=strategy_id,
        )

        diff = client.get(
            f"/api/v1/strategies/{strategy_id}/diff",
            headers=headers,
            params={"from_version": 1, "to_version": 2},
        )
        assert diff.status_code == 200
        body = diff.json()
        assert body["strategy_id"] == strategy_id
        assert body["from_version"] == 1
        assert body["to_version"] == 2
        assert body["patch_op_count"] >= 1

        diff_by_path = {item["path"]: item for item in body["diff_items"]}
        assert "/strategy/description" in diff_by_path
        description_item = diff_by_path["/strategy/description"]
        assert description_item["old_value"] == "initial-description"
        assert description_item["new_value"] == "next-description"


def test_strategy_versions_and_diff_are_hidden_from_other_users() -> None:
    with TestClient(app) as client:
        token_owner = _register_and_get_token(client)
        token_other = _register_and_get_token(client)
        headers_owner = {"Authorization": f"Bearer {token_owner}"}
        headers_other = {"Authorization": f"Bearer {token_other}"}

        session_id = _create_thread(client, headers=headers_owner)
        payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        confirm = _confirm_strategy(
            client,
            headers=headers_owner,
            session_id=session_id,
            dsl_json=payload,
        )
        strategy_id = confirm["strategy_id"]

        versions = client.get(
            f"/api/v1/strategies/{strategy_id}/versions",
            headers=headers_other,
        )
        assert versions.status_code == 404

        diff = client.get(
            f"/api/v1/strategies/{strategy_id}/diff",
            headers=headers_other,
            params={"from_version": 1, "to_version": 1},
        )
        assert diff.status_code == 404


def test_list_strategies_fallbacks_to_latest_job_without_version_snapshot() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, headers=headers)
        user_id = _get_user_id(client, headers=headers)

        payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload["strategy"]["description"] = "fallback-no-version"
        confirm = _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload,
        )

        _insert_completed_backtest_job(
            client,
            strategy_id=confirm["strategy_id"],
            user_id=user_id,
            session_id=session_id,
            strategy_version=None,
            total_return_pct=8.5,
            max_drawdown_pct=-2.2,
            sharpe=1.4,
        )

        listed = client.get("/api/v1/strategies", headers=headers)
        assert listed.status_code == 200
        rows = listed.json()
        assert isinstance(rows, list)
        assert len(rows) == 1
        backtest = rows[0]["latest_backtest"]
        assert backtest["status"] == "done"
        assert backtest["strategy_version"] is None
        assert backtest["total_return_pct"] == 8.5


def test_strategy_versions_only_attach_backtest_to_matching_version() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, headers=headers)
        user_id = _get_user_id(client, headers=headers)

        payload_v1 = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        payload_v1["strategy"]["description"] = "v1-no-backtest"
        confirm_v1 = _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v1,
        )
        strategy_id = confirm_v1["strategy_id"]

        payload_v2 = deepcopy(payload_v1)
        payload_v2["strategy"]["description"] = "v2-with-backtest"
        _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload_v2,
            strategy_id=strategy_id,
        )

        _insert_completed_backtest_job(
            client,
            strategy_id=strategy_id,
            user_id=user_id,
            session_id=session_id,
            strategy_version=2,
            total_return_pct=14.0,
            max_drawdown_pct=-3.5,
            sharpe=1.9,
        )

        response = client.get(
            f"/api/v1/strategies/{strategy_id}/versions",
            headers=headers,
        )
        assert response.status_code == 200
        versions = response.json()
        by_version = {item["version"]: item for item in versions}
        assert by_version[1]["backtest"] is None
        assert by_version[2]["backtest"]["total_return_pct"] == 14.0


def test_strategy_diff_returns_404_for_missing_revision() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        session_id = _create_thread(client, headers=headers)
        payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
        confirm = _confirm_strategy(
            client,
            headers=headers,
            session_id=session_id,
            dsl_json=payload,
        )
        strategy_id = confirm["strategy_id"]

        response = client.get(
            f"/api/v1/strategies/{strategy_id}/diff",
            headers=headers,
            params={"from_version": 99, "to_version": 1},
        )
        assert response.status_code == 404
        detail = response.json().get("detail", {})
        assert detail.get("code") == "STRATEGY_REVISION_NOT_FOUND"
