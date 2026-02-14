from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.backtest import BacktestJob
from src.models.deployment import Deployment
from src.models.phase_transition import PhaseTransition
from src.models.session import Message, Session
from src.models.strategy import Strategy
from src.models.user import User, UserProfile

DEMO_EMAIL = "demo@minsy.ai"


async def _seed_demo_workflow(db: AsyncSession) -> bool:
    user = await db.scalar(select(User).where(User.email == DEMO_EMAIL))
    if user is not None:
        return False

    user = User(
        email=DEMO_EMAIL,
        password_hash="dev_only_hash",
        name="Demo User",
        is_active=True,
    )
    db.add(user)
    await db.flush()

    db.add(
        UserProfile(
            user_id=user.id,
            trading_years_bucket="years_3_5",
            risk_tolerance="moderate",
            return_expectation="growth",
            kyc_status="complete",
        )
    )

    session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={"kyc_summary": "placeholder"},
        metadata_={"source": "seed"},
    )
    db.add(session)
    await db.flush()

    db.add(
        Message(
            session_id=session.id,
            role="user",
            content="Please build a simple momentum strategy on AAPL.",
            phase="strategy",
        )
    )

    strategy = Strategy(
        user_id=user.id,
        session_id=session.id,
        name="Demo Momentum Strategy",
        description="Seed strategy for development only.",
        strategy_type="momentum",
        symbols=["AAPL", "NVDA"],
        timeframe="1d",
        parameters={"lookback": 20},
        entry_rules={"signal": "price_above_sma"},
        exit_rules={"signal": "price_below_sma"},
        risk_management={"stop_loss": 0.08, "take_profit": 0.2},
        status="backtested",
        version=1,
    )
    db.add(strategy)
    await db.flush()

    backtest = BacktestJob(
        strategy_id=strategy.id,
        user_id=user.id,
        session_id=session.id,
        status="completed",
        progress=100,
        current_step="analyzing",
        config={
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100000,
        },
        results={"sharpe": 1.2, "return": 0.18},
    )
    db.add(backtest)
    await db.flush()

    db.add(
        Deployment(
            strategy_id=strategy.id,
            user_id=user.id,
            backtest_job_id=backtest.id,
            mode="paper",
            status="active",
            risk_limits={"max_position": 0.15, "daily_loss_limit": 0.02},
            capital_allocated=Decimal("25000"),
        )
    )
    db.add(
        PhaseTransition(
            session_id=session.id,
            from_phase="kyc",
            to_phase="strategy",
            trigger="system",
            metadata_={"reason": "seed"},
        )
    )

    await db.commit()
    return True


@pytest.mark.asyncio
async def test_seed_data_script_equivalent_inserts_demo_workflow(
    db_session: AsyncSession,
) -> None:
    inserted = await _seed_demo_workflow(db_session)
    assert inserted is True

    seeded_user = await db_session.scalar(select(User).where(User.email == DEMO_EMAIL))
    assert seeded_user is not None

    profile = await db_session.scalar(
        select(UserProfile).where(UserProfile.user_id == seeded_user.id)
    )
    assert profile is not None
    assert profile.kyc_status == "complete"

    seeded_session = await db_session.scalar(
        select(Session).where(Session.user_id == seeded_user.id)
    )
    assert seeded_session is not None
    assert seeded_session.current_phase == "strategy"


@pytest.mark.asyncio
async def test_seed_data_script_equivalent_is_idempotent(
    db_session: AsyncSession,
) -> None:
    first = await _seed_demo_workflow(db_session)
    second = await _seed_demo_workflow(db_session)

    assert first is True
    assert second is False

    user_count = await db_session.scalar(
        select(func.count()).select_from(User).where(User.email == DEMO_EMAIL)
    )
    assert user_count == 1
