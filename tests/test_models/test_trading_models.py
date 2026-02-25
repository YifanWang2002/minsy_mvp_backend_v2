from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.engine.execution.credentials import CredentialCipher
from src.models.broker_account import BrokerAccount
from src.models.deployment import Deployment
from src.models.deployment_run import DeploymentRun
from src.models.fill import Fill
from src.models.manual_trade_action import ManualTradeAction
from src.models.order import Order
from src.models.pnl_snapshot import PnlSnapshot
from src.models.position import Position
from src.models.session import Session
from src.models.strategy import Strategy
from src.models.user import User


async def _create_user_and_strategy(db_session: AsyncSession) -> tuple[User, Strategy]:
    user = User(email="trading_models@test.com", password_hash="pw", name="Trader")
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
        name="Test Strategy",
        description="for trading models",
        strategy_type="trend",
        symbols=["AAPL"],
        timeframe="1m",
        parameters={},
        entry_rules={},
        exit_rules={},
        risk_management={},
        dsl_payload={
            "dsl_version": "1.0",
            "strategy": {"name": "Test Strategy", "description": "for trading models"},
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
    return user, strategy


@pytest.mark.asyncio
async def test_trading_models_relationships(db_session: AsyncSession) -> None:
    user, strategy = await _create_user_and_strategy(db_session)

    cipher = CredentialCipher("test-secret")
    account = BrokerAccount(
        user_id=user.id,
        provider="alpaca",
        mode="paper",
        encrypted_credentials=cipher.encrypt({"api_key": "k", "api_secret": "s"}),
        status="active",
    )
    db_session.add(account)
    await db_session.flush()

    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        mode="paper",
        status="pending",
        risk_limits={"max_position_pct": 0.2},
        capital_allocated=Decimal("10000"),
    )
    db_session.add(deployment)
    await db_session.flush()

    run = DeploymentRun(
        deployment_id=deployment.id,
        strategy_id=strategy.id,
        broker_account_id=account.id,
        status="running",
        runtime_state={"heartbeat": "ok"},
    )
    db_session.add(run)
    await db_session.flush()

    order = Order(
        deployment_id=deployment.id,
        provider_order_id="provider-order-1",
        client_order_id="client-order-1",
        symbol="AAPL",
        side="buy",
        type="market",
        qty=Decimal("1"),
        status="accepted",
    )
    db_session.add(order)
    await db_session.flush()

    fill = Fill(
        order_id=order.id,
        fill_price=Decimal("100"),
        fill_qty=Decimal("1"),
        fee=Decimal("0.1"),
        filled_at=datetime.now(UTC),
    )
    position = Position(
        deployment_id=deployment.id,
        symbol="AAPL",
        side="long",
        qty=Decimal("1"),
        avg_entry_price=Decimal("100"),
        mark_price=Decimal("101"),
        unrealized_pnl=Decimal("1"),
        realized_pnl=Decimal("0"),
    )
    pnl = PnlSnapshot(
        deployment_id=deployment.id,
        equity=Decimal("10001"),
        cash=Decimal("9900"),
        margin_used=Decimal("100"),
        unrealized_pnl=Decimal("1"),
        realized_pnl=Decimal("0"),
        snapshot_time=datetime.now(UTC),
    )
    manual_action = ManualTradeAction(
        user_id=user.id,
        deployment_id=deployment.id,
        action="close",
        payload={"symbol": "AAPL"},
        status="pending",
    )
    db_session.add_all([fill, position, pnl, manual_action])
    await db_session.commit()

    loaded = await db_session.scalar(
        select(Deployment)
        .options(
            selectinload(Deployment.deployment_runs),
            selectinload(Deployment.orders).selectinload(Order.fills),
            selectinload(Deployment.positions),
            selectinload(Deployment.pnl_snapshots),
            selectinload(Deployment.manual_trade_actions),
        )
        .where(Deployment.id == deployment.id)
    )
    assert loaded is not None
    assert len(loaded.deployment_runs) == 1
    assert len(loaded.orders) == 1
    assert len(loaded.orders[0].fills) == 1
    assert len(loaded.positions) == 1
    assert len(loaded.pnl_snapshots) == 1
    assert len(loaded.manual_trade_actions) == 1

    decrypted = cipher.decrypt(account.encrypted_credentials)
    assert decrypted["api_key"] == "k"
    assert account.encrypted_credentials != '{"api_key":"k","api_secret":"s"}'


@pytest.mark.asyncio
async def test_orders_client_order_id_is_unique(db_session: AsyncSession) -> None:
    user, strategy = await _create_user_and_strategy(db_session)
    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        mode="paper",
        status="pending",
        risk_limits={},
        capital_allocated=Decimal("0"),
    )
    db_session.add(deployment)
    await db_session.flush()

    first = Order(
        deployment_id=deployment.id,
        provider_order_id="provider-order-1",
        client_order_id="same-client-order-id",
        symbol="AAPL",
        side="buy",
        type="market",
        qty=Decimal("1"),
        status="accepted",
    )
    second = Order(
        deployment_id=deployment.id,
        provider_order_id="provider-order-2",
        client_order_id="same-client-order-id",
        symbol="AAPL",
        side="buy",
        type="market",
        qty=Decimal("1"),
        status="accepted",
    )

    db_session.add(first)
    await db_session.commit()

    db_session.add(second)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()
