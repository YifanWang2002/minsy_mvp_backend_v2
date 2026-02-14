from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.backtest import BacktestJob
from src.models.deployment import Deployment
from src.models.phase_transition import PhaseTransition
from src.models.session import Message
from src.models.session import Session as AgentSession
from src.models.strategy import Strategy
from src.models.user import User, UserProfile


@pytest.mark.asyncio
async def test_models_create_insert_read(db_session: AsyncSession) -> None:
    user = User(
        email="models@example.com",
        password_hash="hashed_pw",
        name="Models Tester",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    profile = UserProfile(
        user_id=user.id,
        trading_years_bucket="years_3_5",
        risk_tolerance="moderate",
        return_expectation="growth",
        kyc_status="complete",
    )
    db_session.add(profile)

    session = AgentSession(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={"hello": "world"},
        metadata_={"source": "test"},
    )
    db_session.add(session)
    await db_session.flush()

    message = Message(
        session_id=session.id,
        role="user",
        content="Build me a strategy",
        phase="strategy",
        response_id="resp_1",
    )
    db_session.add(message)

    strategy = Strategy(
        user_id=user.id,
        session_id=session.id,
        name="Test Strategy",
        description="placeholder",
        strategy_type="momentum",
        symbols=["AAPL", "NVDA"],
        timeframe="1d",
        parameters={"lookback": 20},
        entry_rules={"rule": "price_above_ma"},
        exit_rules={"rule": "price_below_ma"},
        risk_management={"stop_loss": 0.08},
        status="draft",
        version=1,
    )
    db_session.add(strategy)
    await db_session.flush()

    backtest_job = BacktestJob(
        strategy_id=strategy.id,
        user_id=user.id,
        session_id=session.id,
        status="queued",
        progress=0,
        config={"start_date": "2024-01-01", "end_date": "2024-12-31"},
    )
    db_session.add(backtest_job)
    await db_session.flush()

    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        backtest_job_id=backtest_job.id,
        mode="paper",
        status="pending",
        risk_limits={"max_position": 0.1},
        capital_allocated=Decimal("1000.00"),
    )
    db_session.add(deployment)

    transition = PhaseTransition(
        session_id=session.id,
        from_phase="kyc",
        to_phase="strategy",
        trigger="system",
        metadata_={"reason": "auto"},
    )
    db_session.add(transition)

    await db_session.commit()

    loaded_user = await db_session.scalar(select(User).where(User.email == "models@example.com"))
    loaded_profile = await db_session.scalar(
        select(UserProfile).where(UserProfile.user_id == user.id)
    )
    loaded_session = await db_session.scalar(
        select(AgentSession).where(AgentSession.user_id == user.id)
    )
    loaded_message = await db_session.scalar(
        select(Message).where(Message.session_id == session.id)
    )
    loaded_strategy = await db_session.scalar(
        select(Strategy).where(Strategy.user_id == user.id)
    )
    loaded_backtest = await db_session.scalar(
        select(BacktestJob).where(BacktestJob.user_id == user.id)
    )
    loaded_deployment = await db_session.scalar(
        select(Deployment).where(Deployment.user_id == user.id)
    )
    loaded_transition = await db_session.scalar(
        select(PhaseTransition).where(PhaseTransition.session_id == session.id)
    )

    assert loaded_user is not None
    assert loaded_profile is not None
    assert loaded_session is not None
    assert loaded_message is not None
    assert loaded_strategy is not None
    assert loaded_backtest is not None
    assert loaded_deployment is not None
    assert loaded_transition is not None
