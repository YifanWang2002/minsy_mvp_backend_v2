from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.execution.adapters.base import OrderState
from src.engine.execution.runtime_service import sync_order_status_from_adapter
from src.models.deployment import Deployment
from src.models.order import Order
from src.models.order_state_transition import OrderStateTransition
from src.models.session import Session
from src.models.strategy import Strategy
from src.models.user import User


async def _create_order(db_session: AsyncSession) -> Order:
    user = User(email="alpaca_order_sync@test.com", password_hash="pw", name="Order Sync User")
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
        name="Order Sync Strategy",
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
            "strategy": {"name": "Order Sync Strategy", "description": ""},
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
    await db_session.flush()

    order = Order(
        deployment_id=deployment.id,
        provider_order_id="provider-order-sync-1",
        client_order_id="client-order-sync-1",
        symbol="AAPL",
        side="buy",
        type="market",
        qty=Decimal("1"),
        status="accepted",
        metadata_={"provider_status": "accepted"},
    )
    db_session.add(order)
    await db_session.commit()
    await db_session.refresh(order)
    return order


class _FilledAdapter:
    async def fetch_order(self, _order_id: str) -> OrderState:
        return OrderState(
            provider_order_id="provider-order-sync-1",
            client_order_id="client-order-sync-1",
            symbol="AAPL",
            side="buy",
            order_type="market",
            qty=Decimal("1"),
            filled_qty=Decimal("1"),
            status="filled",
            submitted_at=datetime.now(UTC),
            avg_fill_price=Decimal("101.25"),
            reject_reason=None,
            provider_updated_at=datetime.now(UTC),
            raw={},
        )


class _MissingAdapter:
    async def fetch_order(self, _order_id: str) -> None:
        return None


@pytest.mark.asyncio
async def test_sync_order_status_from_adapter_updates_transition(db_session: AsyncSession) -> None:
    order = await _create_order(db_session)
    synced = await sync_order_status_from_adapter(
        db_session,
        order=order,
        adapter=_FilledAdapter(),
    )

    assert synced.status == "filled"
    assert synced.last_sync_at is not None
    assert synced.provider_updated_at is not None
    assert float(synced.price or Decimal("0")) == 101.25
    assert synced.metadata_["provider_status"] == "filled"

    transitions = (
        await db_session.scalars(
            select(OrderStateTransition).where(OrderStateTransition.order_id == synced.id)
        )
    ).all()
    assert len(transitions) == 1
    assert transitions[0].from_status == "accepted"
    assert transitions[0].to_status == "filled"


@pytest.mark.asyncio
async def test_sync_order_status_from_adapter_handles_missing_provider_order(
    db_session: AsyncSession,
) -> None:
    order = await _create_order(db_session)
    synced = await sync_order_status_from_adapter(
        db_session,
        order=order,
        adapter=_MissingAdapter(),
    )

    assert synced.status == "accepted"
    assert synced.last_sync_at is not None
