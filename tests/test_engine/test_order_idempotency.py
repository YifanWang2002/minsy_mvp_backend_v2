from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.execution.adapters.base import AdapterError, OrderIntent, OrderState
from src.engine.execution.order_manager import OrderManager
from src.models.deployment import Deployment
from src.models.order import Order
from src.models.order_state_transition import OrderStateTransition
from src.models.session import Session
from src.models.strategy import Strategy
from src.models.user import User


async def _create_deployment(db_session: AsyncSession) -> Deployment:
    user = User(email="order_idempotency@test.com", password_hash="pw", name="Order Tester")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="deployment",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    strategy = Strategy(
        user_id=user.id,
        session_id=session.id,
        name="Idempotency Strategy",
        description="",
        strategy_type="trend",
        symbols=["AAPL"],
        timeframe="1m",
        parameters={},
        entry_rules={},
        exit_rules={},
        risk_management={},
        dsl_payload={
            "dsl_version": "1.0",
            "strategy": {"name": "Idempotency Strategy", "description": ""},
            "universe": {"market": "stocks", "tickers": ["AAPL"]},
            "timeframe": "1m",
            "factors": {},
            "trade": {},
        },
        status="validated",
        version=1,
    )
    db_session.add(strategy)
    await db_session.flush()

    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        mode="paper",
        status="active",
        risk_limits={},
        capital_allocated=Decimal("10000"),
        deployed_at=datetime.now(UTC),
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)
    return deployment


@pytest.mark.asyncio
async def test_order_manager_idempotency(db_session: AsyncSession) -> None:
    deployment = await _create_deployment(db_session)
    manager = OrderManager()
    intent = OrderIntent(
        client_order_id="same-order-id",
        symbol="AAPL",
        side="buy",
        qty=Decimal("1"),
    )

    first = await manager.submit_order_intent(
        db=db_session,
        deployment_id=str(deployment.id),
        intent=intent,
        adapter=None,
    )
    second = await manager.submit_order_intent(
        db=db_session,
        deployment_id=str(deployment.id),
        intent=intent,
        adapter=None,
    )

    assert first.idempotent_hit is False
    assert second.idempotent_hit is True
    assert first.order.id == second.order.id

    count = await db_session.scalar(select(func.count(Order.id)))
    assert count == 1
    transitions = await db_session.scalar(select(func.count(OrderStateTransition.id)))
    assert transitions == 1


@pytest.mark.asyncio
async def test_order_manager_adapter_failure_does_not_persist_stale_order(
    db_session: AsyncSession,
) -> None:
    deployment = await _create_deployment(db_session)
    manager = OrderManager()
    intent = OrderIntent(
        client_order_id="adapter-fail-order-id",
        symbol="ETHUSD",
        side="buy",
        qty=Decimal("0.003"),
    )

    class FailingAdapter:
        async def submit_order(self, _: OrderIntent) -> OrderState:
            raise AdapterError("mock 403")

    with pytest.raises(AdapterError):
        await manager.submit_order_intent(
            db=db_session,
            deployment_id=str(deployment.id),
            intent=intent,
            adapter=FailingAdapter(),
        )

    count = await db_session.scalar(select(func.count(Order.id)))
    assert count == 0


@pytest.mark.asyncio
async def test_order_manager_adapter_success_persists_provider_state(
    db_session: AsyncSession,
) -> None:
    deployment = await _create_deployment(db_session)
    manager = OrderManager()
    intent = OrderIntent(
        client_order_id="adapter-success-order-id",
        symbol="ETHUSD",
        side="buy",
        qty=Decimal("0.006"),
    )

    class SuccessAdapter:
        async def submit_order(self, order_intent: OrderIntent) -> OrderState:
            return OrderState(
                provider_order_id="provider-order-1",
                client_order_id=order_intent.client_order_id,
                symbol=order_intent.symbol,
                side=order_intent.side,
                order_type=order_intent.order_type,
                qty=order_intent.qty,
                filled_qty=Decimal("0"),
                status="accepted",
                submitted_at=datetime.now(UTC),
                avg_fill_price=None,
                raw={},
            )

    first = await manager.submit_order_intent(
        db=db_session,
        deployment_id=str(deployment.id),
        intent=intent,
        adapter=SuccessAdapter(),
    )
    second = await manager.submit_order_intent(
        db=db_session,
        deployment_id=str(deployment.id),
        intent=intent,
        adapter=SuccessAdapter(),
    )

    assert first.idempotent_hit is False
    assert second.idempotent_hit is True
    assert first.order.id == second.order.id
    assert first.order.provider_order_id == "provider-order-1"
    assert first.order.status == "accepted"
    assert first.order.last_sync_at is not None

    count = await db_session.scalar(select(func.count(Order.id)))
    assert count == 1
    transitions = await db_session.scalar(select(func.count(OrderStateTransition.id)))
    assert transitions == 1
