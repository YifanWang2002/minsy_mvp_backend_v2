from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from apps.api.routes import deployments as deployments_route
from apps.api.routes import portfolio as portfolio_route
from packages.infra.db import session as db_session_module
from packages.infra.db.models.deployment import Deployment
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.db.models.manual_trade_action import ManualTradeAction
from packages.infra.db.models.order import Order
from packages.infra.db.models.pnl_snapshot import PnlSnapshot
from packages.infra.db.models.trading_event_outbox import TradingEventOutbox
from packages.domain.trading.runtime import runtime_service as runtime_service_module


def _build_deployable_dsl() -> dict[str, object]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": f"Pytest Deploy Guardrail {uuid4().hex[:8]}",
            "description": "Live API deployment guardrail test",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTC/USD"],
        },
        "timeframe": "1m",
        "factors": {
            "ema_9": {
                "type": "ema",
                "params": {"period": 9, "source": "close"},
            },
            "ema_21": {
                "type": "ema",
                "params": {"period": 21, "source": "close"},
            },
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_9"},
                            "op": "cross_above",
                            "b": {"ref": "ema_21"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_cross_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_9"},
                                "op": "cross_below",
                                "b": {"ref": "ema_21"},
                            }
                        },
                    },
                    {
                        "type": "bracket_rr",
                        "name": "protective_bracket",
                        "stop": {"kind": "pct", "value": 0.01},
                        "risk_reward": 2.0,
                    },
                ],
            }
        },
    }


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-deployment-guardrails-live"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _create_strategy(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/strategies/confirm",
        headers=auth_headers,
        json={
            "session_id": _create_thread(api_test_client, auth_headers),
            "dsl_json": _build_deployable_dsl(),
            "auto_start_backtest": False,
            "language": "en",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["strategy_id"])


def _create_sandbox_broker_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> str:
    response = api_test_client.post(
        "/api/v1/broker-accounts",
        headers=auth_headers,
        params={"validate": "false"},
        json={
            "provider": "sandbox",
            "mode": "paper",
            "metadata": {"source": f"pytest-deployment-{uuid4().hex[:8]}"},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["broker_account_id"])


def _create_deployment(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    *,
    strategy_id: str,
    broker_account_id: str,
) -> str:
    response = api_test_client.post(
        "/api/v1/deployments",
        headers=auth_headers,
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "mode": "paper",
            "capital_allocated": 10000,
            "risk_limits": {},
            "runtime_state": {},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["deployment_id"])


def _deactivate_broker_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    *,
    broker_account_id: str,
) -> None:
    response = api_test_client.post(
        f"/api/v1/broker-accounts/{broker_account_id}/deactivate",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text


def _outbox_stats(api_test_client: TestClient, deployment_id: str) -> tuple[int, int]:
    deployment_uuid = UUID(deployment_id)

    async def _load() -> tuple[int, int]:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            count = await db.scalar(
                select(func.count())
                .select_from(TradingEventOutbox)
                .where(TradingEventOutbox.deployment_id == deployment_uuid)
            )
            latest_seq = await db.scalar(
                select(TradingEventOutbox.event_seq)
                .where(TradingEventOutbox.deployment_id == deployment_uuid)
                .order_by(TradingEventOutbox.event_seq.desc())
                .limit(1)
            )
            return int(count or 0), int(latest_seq or 0)

    assert api_test_client.portal is not None
    return api_test_client.portal.call(_load)


def test_000_create_deployment_rejects_inactive_broker_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    _deactivate_broker_account(
        api_test_client,
        auth_headers,
        broker_account_id=broker_account_id,
    )

    response = api_test_client.post(
        "/api/v1/deployments",
        headers=auth_headers,
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "mode": "paper",
            "capital_allocated": 10000,
            "risk_limits": {},
            "runtime_state": {},
        },
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "BROKER_ACCOUNT_INACTIVE"


def test_010_start_deployment_rejects_inactive_broker_account(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    _deactivate_broker_account(
        api_test_client,
        auth_headers,
        broker_account_id=broker_account_id,
    )

    response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "BROKER_ACCOUNT_INACTIVE"


def test_020_stream_heartbeat_does_not_append_outbox_rows(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    before_count, before_latest_seq = _outbox_stats(api_test_client, deployment_id)
    assert before_count > 0

    response = api_test_client.get(
        f"/api/v1/stream/deployments/{deployment_id}",
        headers=auth_headers,
        params={
            "cursor": before_latest_seq,
            "poll_seconds": 0.2,
            "heartbeat_seconds": 0.2,
            "max_events": 1,
        },
    )

    assert response.status_code == 200, response.text
    assert "event: heartbeat" in response.text

    after_count, after_latest_seq = _outbox_stats(api_test_client, deployment_id)
    assert after_count == before_count
    assert after_latest_seq == before_latest_seq


def test_030_manual_action_is_queued_before_runtime_execution(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )

    monkeypatch.setattr(
        deployments_route,
        "enqueue_paper_trading_runtime",
        lambda *args, **kwargs: "paper-task-1",
    )
    start_response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert start_response.status_code == 200, start_response.text

    enqueued_action_ids: list[str] = []

    def _fake_enqueue(action_id) -> str:
        enqueued_action_ids.append(str(action_id))
        return "manual-task-1"

    async def _unexpected_sync_execute(*args, **kwargs):
        raise AssertionError(
            "manual action should not execute synchronously when queueing succeeds"
        )

    monkeypatch.setattr(
        deployments_route,
        "enqueue_execute_manual_trade_action",
        _fake_enqueue,
    )
    monkeypatch.setattr(
        deployments_route,
        "execute_manual_trade_action",
        _unexpected_sync_execute,
    )

    response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/manual-actions",
        headers=auth_headers,
        json={
            "action": "open",
            "payload": {
                "symbol": "BTC/USD",
                "side": "long",
                "qty": 0.25,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "executing"
    assert body["payload"]["_execution"]["reason"] == "queued"
    assert body["payload"]["_execution"]["task_id"] == "manual-task-1"
    assert len(enqueued_action_ids) == 1

    action_id = UUID(body["manual_trade_action_id"])

    async def _load_status() -> tuple[str, str | None]:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            action = await db.scalar(
                select(ManualTradeAction).where(ManualTradeAction.id == action_id)
            )
            assert action is not None
            execution = (
                action.payload.get("_execution")
                if isinstance(action.payload, dict)
                and isinstance(action.payload.get("_execution"), dict)
                else {}
            )
            reason = execution.get("reason") if isinstance(execution, dict) else None
            return action.status, reason if isinstance(reason, str) else None

    assert api_test_client.portal is not None
    stored_status, stored_reason = api_test_client.portal.call(_load_status)
    assert stored_status == "executing"
    assert stored_reason == "queued"


def test_040_manual_action_lock_conflict_schedules_retry(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )

    monkeypatch.setattr(
        deployments_route,
        "enqueue_paper_trading_runtime",
        lambda *args, **kwargs: "paper-task-lock-test",
    )
    start_response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert start_response.status_code == 200, start_response.text

    enqueue_calls: list[tuple[str, int | None]] = []

    def _fake_enqueue(action_id, *, countdown_seconds=None) -> str | None:
        enqueue_calls.append((str(action_id), countdown_seconds))
        if len(enqueue_calls) == 1:
            return None
        return "manual-retry-task-1"

    async def _always_locked(deployment_id_arg) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(
        deployments_route,
        "enqueue_execute_manual_trade_action",
        _fake_enqueue,
    )
    monkeypatch.setattr(
        runtime_service_module.deployment_runtime_lock,
        "acquire",
        _always_locked,
    )

    response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/manual-actions",
        headers=auth_headers,
        json={
            "action": "open",
            "payload": {
                "symbol": "BTC/USD",
                "side": "long",
                "qty": 0.25,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "executing"
    execution = body["payload"]["_execution"]
    assert execution["reason"] == "waiting_for_runtime_lock"
    assert execution["task_id"] == "manual-retry-task-1"
    assert execution["lock_retry_count"] == 1
    assert len(enqueue_calls) == 2
    assert enqueue_calls[0][1] is None
    assert enqueue_calls[1][1] == 1


def test_050_non_active_portfolio_uses_cached_snapshot_without_refresh(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    deployment_uuid = UUID(deployment_id)
    snapshot_time = datetime.now(UTC)

    async def _seed_snapshot() -> None:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            db.add(
                PnlSnapshot(
                    deployment_id=deployment_uuid,
                    equity=Decimal("10025"),
                    cash=Decimal("10010"),
                    margin_used=Decimal("15"),
                    unrealized_pnl=Decimal("9"),
                    realized_pnl=Decimal("6"),
                    snapshot_time=snapshot_time,
                )
            )
            await db.commit()

    async def _unexpected_refresh(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(
            "portfolio refresh should not run for cached non-active deployments"
        )

    assert api_test_client.portal is not None
    api_test_client.portal.call(_seed_snapshot)
    monkeypatch.setattr(
        portfolio_route,
        "refresh_portfolio_snapshot_for_poll",
        _unexpected_refresh,
    )

    response = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/portfolio",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["equity"] == 10025.0
    assert payload["cash"] == 10010.0
    assert payload["margin_used"] == 15.0


def test_060_active_cached_portfolio_still_syncs_pending_orders(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    deployment_uuid = UUID(deployment_id)
    snapshot_time = datetime.now(UTC)

    monkeypatch.setattr(
        deployments_route,
        "enqueue_paper_trading_runtime",
        lambda *args, **kwargs: "paper-task-portfolio-sync",
    )
    start_response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert start_response.status_code == 200, start_response.text

    async def _seed_snapshot() -> None:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            db.add(
                PnlSnapshot(
                    deployment_id=deployment_uuid,
                    equity=Decimal("10050"),
                    cash=Decimal("10000"),
                    margin_used=Decimal("25"),
                    unrealized_pnl=Decimal("10"),
                    realized_pnl=Decimal("15"),
                    snapshot_time=snapshot_time,
                )
            )
            await db.commit()

    sync_calls: list[str] = []

    async def _fake_sync_pending_orders(*args, **kwargs):  # noqa: ANN002, ANN003
        sync_calls.append("called")
        return {"pending_order_fill_updates": 0}

    async def _unexpected_refresh(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(
            "full portfolio refresh should not run when cached active snapshot is fresh"
        )

    assert api_test_client.portal is not None
    api_test_client.portal.call(_seed_snapshot)
    monkeypatch.setattr(
        portfolio_route,
        "sync_pending_orders_for_poll",
        _fake_sync_pending_orders,
    )
    monkeypatch.setattr(
        portfolio_route,
        "refresh_portfolio_snapshot_for_poll",
        _unexpected_refresh,
    )

    response = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/portfolio",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    assert sync_calls == ["called"]


def test_070_latest_manual_action_returns_most_recent_row(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )

    create_response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/manual-actions",
        headers=auth_headers,
        json={
            "action": "open",
            "payload": {
                "symbol": "BTC/USD",
                "side": "long",
                "qty": 0.25,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    created = create_response.json()

    latest_response = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/manual-actions/latest",
        headers=auth_headers,
    )

    assert latest_response.status_code == 200, latest_response.text
    latest = latest_response.json()
    assert latest["manual_trade_action_id"] == created["manual_trade_action_id"]
    assert latest["status"] == created["status"]


def test_080_portfolio_metrics_ignore_account_level_broker_totals(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    deployment_uuid = UUID(deployment_id)
    snapshot_time = datetime.now(UTC)

    monkeypatch.setattr(
        deployments_route,
        "enqueue_paper_trading_runtime",
        lambda *args, **kwargs: "paper-task-broker-metrics",
    )
    start_response = api_test_client.post(
        f"/api/v1/deployments/{deployment_id}/start",
        headers=auth_headers,
    )
    assert start_response.status_code == 200, start_response.text

    fake_broker_equity = 987654.32
    fake_broker_cash = 123456.78

    async def _seed_snapshot_and_runtime_state() -> None:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            db.add(
                PnlSnapshot(
                    deployment_id=deployment_uuid,
                    equity=Decimal("10050"),
                    cash=Decimal("10000"),
                    margin_used=Decimal("25"),
                    unrealized_pnl=Decimal("10"),
                    realized_pnl=Decimal("15"),
                    snapshot_time=snapshot_time,
                )
            )
            run = await db.scalar(
                select(DeploymentRun)
                .where(DeploymentRun.deployment_id == deployment_uuid)
                .order_by(DeploymentRun.created_at.desc(), DeploymentRun.id.desc())
                .limit(1)
            )
            assert run is not None
            run.runtime_state = {
                **(run.runtime_state if isinstance(run.runtime_state, dict) else {}),
                "broker_account": {
                    "provider": "sandbox",
                    "source": "broker_reported",
                    "sync_status": "ok",
                    "equity": fake_broker_equity,
                    "cash": fake_broker_cash,
                    "margin_used": 333.0,
                    "unrealized_pnl": 444.0,
                    "realized_pnl": 555.0,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            }
            await db.commit()

    async def _fake_sync_pending_orders(*args, **kwargs):  # noqa: ANN002, ANN003
        return {"pending_order_fill_updates": 0}

    assert api_test_client.portal is not None
    api_test_client.portal.call(_seed_snapshot_and_runtime_state)
    monkeypatch.setattr(
        portfolio_route,
        "sync_pending_orders_for_poll",
        _fake_sync_pending_orders,
    )

    response = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/portfolio",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["metrics_source"] == "platform_estimate"
    assert payload["equity"] == 10050.0
    assert payload["cash"] == 10000.0
    assert payload["margin_used"] == 25.0
    assert payload["unrealized_pnl"] == 10.0
    assert payload["realized_pnl"] == 15.0
    assert payload["broker_account"]["equity"] == fake_broker_equity
    assert payload["broker_account"]["cash"] == fake_broker_cash


def test_090_latest_manual_action_reconciles_terminal_order_state(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    strategy_id = _create_strategy(api_test_client, auth_headers)
    broker_account_id = _create_sandbox_broker_account(api_test_client, auth_headers)
    deployment_id = _create_deployment(
        api_test_client,
        auth_headers,
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
    )
    deployment_uuid = UUID(deployment_id)
    order_id = uuid4()

    async def _seed_rows() -> None:
        assert db_session_module.AsyncSessionLocal is not None
        async with db_session_module.AsyncSessionLocal() as db:
            deployment = await db.scalar(
                select(Deployment).where(Deployment.id == deployment_uuid)
            )
            assert deployment is not None
            order = Order(
                id=order_id,
                deployment_id=deployment_uuid,
                provider_order_id=f"provider-{order_id.hex[:8]}",
                client_order_id=f"client-{order_id.hex[:8]}",
                symbol="BTC/USD",
                side="buy",
                type="market",
                qty=Decimal("0.25"),
                price=Decimal("42000"),
                status="filled",
                provider_updated_at=datetime.now(UTC),
                last_sync_at=datetime.now(UTC),
                submitted_at=datetime.now(UTC),
                metadata_={"provider_status": "filled"},
            )
            action = ManualTradeAction(
                user_id=deployment.user_id,
                deployment_id=deployment_uuid,
                action="open",
                status="accepted",
                payload={
                    "symbol": "BTC/USD",
                    "qty": 0.25,
                    "_execution": {
                        "status": "accepted",
                        "reason": "order_pending_sync",
                        "order_id": str(order_id),
                    },
                },
            )
            db.add(order)
            db.add(action)
            await db.commit()

    assert api_test_client.portal is not None
    api_test_client.portal.call(_seed_rows)

    response = api_test_client.get(
        f"/api/v1/deployments/{deployment_id}/manual-actions/latest",
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["payload"]["_execution"]["status"] == "completed"
    assert payload["payload"]["_execution"]["reason"] == "order_filled"
