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
from src.models.user import User


@pytest.mark.asyncio
async def test_jsonb_fields_round_trip_dict_data(db_session: AsyncSession) -> None:
    user = User(email="jsonb@example.com", password_hash="hashed", name="JSONB Tester")
    db_session.add(user)
    await db_session.flush()

    session = AgentSession(
        user_id=user.id,
        artifacts={"summary": {"risk": "aggressive"}},
        metadata_={"trace_id": "abc123"},
    )
    db_session.add(session)
    await db_session.flush()

    message = Message(
        session_id=session.id,
        role="assistant",
        content="ok",
        phase="strategy",
        tool_calls=[{"name": "search", "args": {"q": "AAPL"}, "result": {"ok": True}}],
        token_usage={"prompt": 10, "completion": 20, "total": 30},
    )
    db_session.add(message)

    strategy = Strategy(
        user_id=user.id,
        session_id=session.id,
        name="jsonb strategy",
        strategy_type="dual_ma",
        symbols=["AAPL"],
        timeframe="1d",
        parameters={"fast": 10, "slow": 50},
        entry_rules={"condition": "fast_cross_up"},
        exit_rules={"condition": "fast_cross_down"},
        risk_management={"stop_loss": 0.05, "take_profit": 0.1},
    )
    db_session.add(strategy)
    await db_session.flush()

    backtest = BacktestJob(
        strategy_id=strategy.id,
        user_id=user.id,
        session_id=session.id,
        config={"start": "2024-01-01", "end": "2024-12-31", "initial_capital": 100000},
        results={"return": 0.21, "max_drawdown": 0.12},
    )
    db_session.add(backtest)
    await db_session.flush()

    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        backtest_job_id=backtest.id,
        risk_limits={"daily_loss_limit": 0.02, "max_position": 0.15},
        capital_allocated=Decimal("5000.00"),
    )
    transition = PhaseTransition(
        session_id=session.id,
        from_phase="strategy",
        to_phase="backtest",
        trigger="ai_output",
        metadata_={"confidence": 0.86},
    )
    db_session.add_all([deployment, transition])

    await db_session.commit()

    loaded_session = await db_session.scalar(select(AgentSession).where(AgentSession.id == session.id))
    loaded_message = await db_session.scalar(select(Message).where(Message.session_id == session.id))
    loaded_strategy = await db_session.scalar(select(Strategy).where(Strategy.id == strategy.id))
    loaded_backtest = await db_session.scalar(select(BacktestJob).where(BacktestJob.id == backtest.id))
    loaded_deployment = await db_session.scalar(
        select(Deployment).where(Deployment.id == deployment.id)
    )
    loaded_transition = await db_session.scalar(
        select(PhaseTransition).where(PhaseTransition.id == transition.id)
    )

    assert loaded_session is not None and loaded_session.artifacts["summary"]["risk"] == "aggressive"
    assert loaded_session is not None and loaded_session.metadata_["trace_id"] == "abc123"
    assert loaded_message is not None and loaded_message.tool_calls[0]["name"] == "search"
    assert loaded_message is not None and loaded_message.token_usage["total"] == 30
    assert loaded_strategy is not None and loaded_strategy.parameters["fast"] == 10
    assert loaded_backtest is not None and loaded_backtest.results["return"] == 0.21
    assert loaded_deployment is not None and loaded_deployment.risk_limits["max_position"] == 0.15
    assert loaded_transition is not None and loaded_transition.metadata_["confidence"] == 0.86
