from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.models.deployment import Deployment
from src.models.session import Session
from src.models.strategy import Strategy
from src.models.trade_approval_request import TradeApprovalRequest
from src.models.user import User


@pytest.mark.asyncio
async def test_trade_approval_request_requires_unique_approval_key(db_session) -> None:
    user = User(email=f"approval_model_{uuid4().hex}@test.com", password_hash="pw", name="Model User")
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
        name="Model Strategy",
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
            "strategy": {"name": "Model Strategy"},
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
        risk_limits={"order_qty": 1},
        capital_allocated=Decimal("10000"),
    )
    db_session.add(deployment)
    await db_session.flush()

    shared_key = f"open_approval:{deployment.id}:OPEN_LONG:AAPL:1m:1"
    request_a = TradeApprovalRequest(
        user_id=user.id,
        deployment_id=deployment.id,
        signal="OPEN_LONG",
        side="long",
        symbol="AAPL",
        qty=Decimal("1"),
        mark_price=Decimal("100"),
        reason="entry",
        timeframe="1m",
        bar_time=datetime.now(UTC),
        approval_channel="telegram",
        approval_key=shared_key,
        status="pending",
        requested_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
        intent_payload={},
    )
    db_session.add(request_a)
    await db_session.commit()

    request_b = TradeApprovalRequest(
        user_id=user.id,
        deployment_id=deployment.id,
        signal="OPEN_LONG",
        side="long",
        symbol="AAPL",
        qty=Decimal("1"),
        mark_price=Decimal("100"),
        reason="entry_again",
        timeframe="1m",
        bar_time=datetime.now(UTC),
        approval_channel="telegram",
        approval_key=shared_key,
        status="pending",
        requested_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
        intent_payload={},
    )
    db_session.add(request_b)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()
