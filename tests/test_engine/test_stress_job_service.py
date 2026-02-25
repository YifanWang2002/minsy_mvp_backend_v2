from __future__ import annotations

from uuid import uuid4

import pytest

from src.engine.stress.service import (
    StressInputError,
    create_stress_job,
    get_optimization_pareto_points,
    get_stress_job_view,
)
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.models.session import Session
from src.models.user import User


@pytest.mark.asyncio
async def test_create_and_get_stress_job_view(db_session) -> None:
    user = User(
        email=f"stress_service_{uuid4().hex}@example.com",
        password_hash="hash",
        name="stress-user",
    )
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="stress_test",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    payload = load_strategy_payload(EXAMPLE_PATH)
    persistence = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=payload,
        auto_commit=False,
    )
    strategy = persistence.strategy
    await db_session.commit()

    receipt = await create_stress_job(
        db_session,
        job_type="monte_carlo",
        strategy_id=strategy.id,
        backtest_job_id=None,
        config={"num_trials": 100, "horizon_bars": 30},
        user_id=user.id,
        auto_commit=True,
    )

    assert receipt.job_type == "monte_carlo"
    assert receipt.status == "pending"

    view = await get_stress_job_view(
        db_session,
        job_id=receipt.job_id,
        user_id=user.id,
    )
    assert view.job_type == "monte_carlo"
    assert view.status == "pending"
    assert view.progress == 0

    with pytest.raises(StressInputError):
        await get_optimization_pareto_points(
            db_session,
            job_id=receipt.job_id,
            x_metric="total_return_pct",
            y_metric="max_drawdown_pct",
            user_id=user.id,
        )
