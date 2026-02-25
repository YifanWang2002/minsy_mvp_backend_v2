from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.execution.runtime_service import _apply_position_after_fill
from src.engine.pnl.service import PnlService
from src.models.deployment import Deployment
from src.models.position import Position
from src.models.session import Session
from src.models.strategy import Strategy
from src.models.user import User


async def _create_deployment(db_session: AsyncSession) -> Deployment:
    user = User(email="pnl_engine@test.com", password_hash="pw", name="PnL User")
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
        name="PnL Strategy",
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
            "strategy": {"name": "PnL Strategy", "description": ""},
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
async def test_pnl_service_handles_partial_close_and_fee(db_session: AsyncSession) -> None:
    deployment = await _create_deployment(db_session)
    position = Position(
        deployment_id=deployment.id,
        symbol="AAPL",
        side="long",
        qty=Decimal("2"),
        avg_entry_price=Decimal("100"),
        mark_price=Decimal("110"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )
    db_session.add(position)
    await db_session.commit()

    await _apply_position_after_fill(
        db=db_session,
        deployment_id=deployment.id,
        symbol="AAPL",
        signal="CLOSE",
        fill_qty=Decimal("1"),
        fill_price=Decimal("120"),
        current_position_side="long",
        fill_fee=Decimal("0.5"),
    )
    await db_session.flush()
    position.mark_price = Decimal("115")
    await db_session.flush()

    pnl_service = PnlService()
    snapshot = await pnl_service.build_snapshot(db_session, deployment_id=deployment.id)

    assert snapshot.realized_pnl == Decimal("19.5")
    assert snapshot.unrealized_pnl == Decimal("15")
    assert snapshot.margin_used == Decimal("115")
    assert snapshot.cash == Decimal("9904.5")
    assert snapshot.equity == Decimal("10034.5")

    row = await pnl_service.persist_snapshot(db_session, snapshot=snapshot)
    assert row.realized_pnl == Decimal("19.5")
    assert row.unrealized_pnl == Decimal("15")


@pytest.mark.asyncio
async def test_pnl_service_supports_batched_closing_sequence(db_session: AsyncSession) -> None:
    deployment = await _create_deployment(db_session)
    position = Position(
        deployment_id=deployment.id,
        symbol="AAPL",
        side="long",
        qty=Decimal("3"),
        avg_entry_price=Decimal("100"),
        mark_price=Decimal("100"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )
    db_session.add(position)
    await db_session.commit()

    await _apply_position_after_fill(
        db=db_session,
        deployment_id=deployment.id,
        symbol="AAPL",
        signal="CLOSE",
        fill_qty=Decimal("1"),
        fill_price=Decimal("110"),
        current_position_side="long",
        fill_fee=Decimal("1"),
    )
    await _apply_position_after_fill(
        db=db_session,
        deployment_id=deployment.id,
        symbol="AAPL",
        signal="CLOSE",
        fill_qty=Decimal("1"),
        fill_price=Decimal("90"),
        current_position_side="long",
        fill_fee=Decimal("0"),
    )
    await _apply_position_after_fill(
        db=db_session,
        deployment_id=deployment.id,
        symbol="AAPL",
        signal="CLOSE",
        fill_qty=Decimal("1"),
        fill_price=Decimal("100"),
        current_position_side="long",
        fill_fee=Decimal("0.5"),
    )

    await db_session.flush()
    await db_session.refresh(position)
    assert position.qty == Decimal("0")
    assert position.side == "flat"
    assert position.realized_pnl == Decimal("-1.5")

    pnl_service = PnlService()
    short_unrealized = pnl_service.compute_unrealized(
        side="short",
        qty=Decimal("2"),
        avg_entry_price=Decimal("100"),
        mark_price=Decimal("92"),
    )
    assert short_unrealized == Decimal("16")
