"""Deployments route tests with mocked service boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from apps.api.routes import deployments as deployments_routes
from apps.api.schemas.events import DeploymentResponse
from apps.api.schemas.requests import DeploymentCreateRequest, ManualTradeActionRequest
from packages.domain.exceptions import DomainError


class _FakeDbSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commit_count = 0

    def add(self, obj: object) -> None:
        if getattr(obj, "id", None) is None:
            setattr(obj, "id", uuid4())
        self.added.append(obj)

    async def commit(self) -> None:
        self.commit_count += 1

    async def refresh(self, obj: object) -> None:
        now = datetime.now(UTC)
        if getattr(obj, "created_at", None) is None:
            setattr(obj, "created_at", now)
        setattr(obj, "updated_at", now)


class _FakeScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4())


def _build_deployment_response(
    *,
    deployment_id: UUID,
    strategy_id: UUID,
    user_id: UUID,
) -> DeploymentResponse:
    now = datetime.now(UTC)
    return DeploymentResponse(
        deployment_id=deployment_id,
        strategy_id=strategy_id,
        strategy_name="demo",
        user_id=user_id,
        broker_provider="alpaca",
        mode="paper",
        status="pending",
        market="us",
        symbols=["AAPL"],
        timeframe="1m",
        capital_allocated=100.0,
        risk_limits={},
        deployed_at=None,
        stopped_at=None,
        created_at=now,
        updated_at=now,
        run=None,
    )


@pytest.mark.asyncio
async def test_create_deployment_route_uses_mocked_domain_service(monkeypatch) -> None:
    user = _fake_user()
    db = _FakeDbSession()
    strategy_id = uuid4()
    broker_account_id = uuid4()
    deployment_id = uuid4()
    payload = DeploymentCreateRequest(
        strategy_id=strategy_id,
        broker_account_id=broker_account_id,
        mode="paper",
        capital_allocated=Decimal("100"),
        risk_limits={"max_drawdown": 0.2},
        runtime_state={"foo": "bar"},
    )
    captured: dict[str, object] = {}

    async def _fake_create_domain(db_session, **kwargs):
        captured["db_session"] = db_session
        captured["kwargs"] = kwargs
        return SimpleNamespace(id=deployment_id)

    async def _fake_append_snapshot(db_session, *, deployment_id):
        captured["append_snapshot"] = (db_session, deployment_id)

    def _fake_serialize(_deployment):
        return _build_deployment_response(
            deployment_id=deployment_id,
            strategy_id=strategy_id,
            user_id=user.id,
        )

    monkeypatch.setattr(
        deployments_routes.deployment_ops_domain,
        "create_deployment",
        _fake_create_domain,
    )
    monkeypatch.setattr(
        deployments_routes,
        "append_trading_event_snapshot",
        _fake_append_snapshot,
    )
    monkeypatch.setattr(deployments_routes, "_serialize_deployment", _fake_serialize)

    response = await deployments_routes.create_deployment(
        payload=payload,
        user=user,
        db=db,
    )

    assert response.deployment_id == deployment_id
    assert captured["db_session"] is db
    assert captured["kwargs"] == {
        "strategy_id": strategy_id,
        "broker_account_id": broker_account_id,
        "user_id": user.id,
        "mode": "paper",
        "capital_allocated": Decimal("100"),
        "risk_limits": {"max_drawdown": 0.2},
        "runtime_state": {"foo": "bar"},
    }
    assert captured["append_snapshot"] == (db, deployment_id)


@pytest.mark.asyncio
async def test_create_deployment_route_propagates_domain_error(monkeypatch) -> None:
    user = _fake_user()
    db = _FakeDbSession()
    payload = DeploymentCreateRequest(
        strategy_id=uuid4(),
        broker_account_id=uuid4(),
        mode="paper",
        capital_allocated=Decimal("9999"),
        risk_limits={},
        runtime_state={},
    )

    async def _fake_create_domain(_db_session, **_kwargs):
        raise DomainError(
            status_code=422,
            code="DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET",
            message="Requested capital exceeds remaining broker capital budget.",
        )

    monkeypatch.setattr(
        deployments_routes.deployment_ops_domain,
        "create_deployment",
        _fake_create_domain,
    )

    with pytest.raises(deployments_routes.HTTPException) as exc_info:
        await deployments_routes.create_deployment(
            payload=payload,
            user=user,
            db=db,
        )

    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail
    assert detail["code"] == "DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET"


@pytest.mark.asyncio
async def test_create_manual_action_route_uses_mocked_queue_service(monkeypatch) -> None:
    user = _fake_user()
    db = _FakeDbSession()
    deployment_id = uuid4()
    payload = ManualTradeActionRequest(action="open", payload={"symbol": "AAPL"})
    queued: dict[str, object] = {}
    snapshot_calls: list[tuple[object, object]] = []

    async def _fake_load_owned_deployment(_db, *, deployment_id, user_id):
        assert deployment_id is not None
        assert user_id == user.id
        return SimpleNamespace(status="active")

    def _fake_enqueue(action_id, *, countdown_seconds=None):
        queued["action_id"] = action_id
        queued["countdown_seconds"] = countdown_seconds
        return "task_mocked_123"

    async def _unexpected_execute_manual_trade_action(*_args, **_kwargs):
        raise AssertionError("manual action route should not call runtime fallback when task is queued")

    async def _fake_append_snapshot(db_session, *, deployment_id):
        snapshot_calls.append((db_session, deployment_id))

    monkeypatch.setattr(
        deployments_routes,
        "_load_owned_deployment",
        _fake_load_owned_deployment,
    )
    monkeypatch.setattr(
        deployments_routes,
        "enqueue_execute_manual_trade_action",
        _fake_enqueue,
    )
    monkeypatch.setattr(
        deployments_routes,
        "execute_manual_trade_action",
        _unexpected_execute_manual_trade_action,
    )
    monkeypatch.setattr(
        deployments_routes,
        "append_trading_event_snapshot",
        _fake_append_snapshot,
    )

    response = await deployments_routes.create_manual_action(
        deployment_id=deployment_id,
        payload=payload,
        user=user,
        db=db,
    )

    assert response.status == "executing"
    assert response.payload.get("_execution", {}).get("task_id") == "task_mocked_123"
    assert queued["action_id"] == response.manual_trade_action_id
    assert queued["countdown_seconds"] is None
    assert db.commit_count >= 2
    assert len(snapshot_calls) >= 2
    assert all(call[0] is db for call in snapshot_calls)
    assert all(call[1] == deployment_id for call in snapshot_calls)


@pytest.mark.asyncio
async def test_list_deployment_orders_history_returns_paginated_payload(monkeypatch) -> None:
    user = _fake_user()
    deployment_id = uuid4()
    now = datetime.now(UTC)
    order_one = SimpleNamespace(
        id=uuid4(),
        deployment_id=deployment_id,
        provider_order_id="provider-1",
        client_order_id="client-1",
        symbol="AAPL",
        side="buy",
        type="market",
        qty=Decimal("1"),
        price=Decimal("101"),
        status="filled",
        reject_reason=None,
        last_sync_at=now,
        submitted_at=now - timedelta(minutes=1),
        metadata_={"provider_status": "filled"},
        created_at=now,
        updated_at=now,
    )
    order_two = SimpleNamespace(
        id=uuid4(),
        deployment_id=deployment_id,
        provider_order_id="provider-2",
        client_order_id="client-2",
        symbol="AAPL",
        side="sell",
        type="market",
        qty=Decimal("1"),
        price=Decimal("102"),
        status="filled",
        reject_reason=None,
        last_sync_at=now,
        submitted_at=now - timedelta(minutes=2),
        metadata_={"provider_status": "filled"},
        created_at=now,
        updated_at=now,
    )
    db = SimpleNamespace(
        scalar=AsyncMock(return_value=6),
        scalars=AsyncMock(return_value=_FakeScalarRows([order_one, order_two])),
    )

    async def _fake_load_owned_deployment(_db, *, deployment_id, user_id):
        assert deployment_id is not None
        assert user_id == user.id
        return SimpleNamespace(id=deployment_id)

    monkeypatch.setattr(
        deployments_routes,
        "_load_owned_deployment",
        _fake_load_owned_deployment,
    )

    response = await deployments_routes.list_deployment_orders_history(
        deployment_id=deployment_id,
        page=2,
        page_size=2,
        user=user,
        db=db,
    )

    assert response.page == 2
    assert response.page_size == 2
    assert response.total == 6
    assert response.total_pages == 3
    assert len(response.items) == 2
    assert [item.order_id for item in response.items] == [order_one.id, order_two.id]
