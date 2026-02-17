from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.backtest import (
    BacktestJobNotFoundError,
    BacktestStrategyNotFoundError,
    BacktestJobView,
    create_backtest_job,
    execute_backtest_job,
    execute_backtest_job_with_fresh_session,
    get_backtest_job_view,
    schedule_backtest_job,
)
from src.engine.data import DataLoader
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.models.backtest import BacktestJob
from src.models.session import Session as AgentSession
from src.models.user import User


async def _create_strategy(db_session: AsyncSession, *, email: str):
    user = User(email=email, password_hash="hashed", name=email)
    db_session.add(user)
    await db_session.flush()

    session = AgentSession(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
    payload["universe"]["tickers"] = ["BTCUSDT"]
    created = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=payload,
    )
    return created.strategy


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=180, freq="4h", tz="UTC")
    close = [100 + i * 0.3 for i in range(90)] + [127 - i * 0.35 for i in range(90)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [item + 0.4 for item in close],
            "low": [item - 0.4 for item in close],
            "close": close,
            "volume": [2000.0] * len(close),
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_backtest_job_lifecycle_done(db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_done@example.com")

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        return _sample_frame()

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        return {
            "available_timerange": {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-02-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr(DataLoader, "load", _fake_load)
    monkeypatch.setattr(DataLoader, "get_symbol_metadata", _fake_metadata)

    receipt = await create_backtest_job(db_session, strategy_id=strategy.id)
    assert receipt.status == "pending"

    view = await execute_backtest_job(db_session, job_id=receipt.job_id)
    assert view.status == "done"
    assert view.progress == 100
    assert view.result is not None
    assert "summary" in view.result
    assert "performance" in view.result

    queried = await get_backtest_job_view(db_session, job_id=receipt.job_id)
    assert queried.status == "done"
    assert queried.completed_at is not None


@pytest.mark.asyncio
async def test_backtest_job_lifecycle_failed(db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_failed@example.com")

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        raise FileNotFoundError("test missing data")

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        return {
            "available_timerange": {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-02-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr(DataLoader, "load", _fake_load)
    monkeypatch.setattr(DataLoader, "get_symbol_metadata", _fake_metadata)

    receipt = await create_backtest_job(db_session, strategy_id=strategy.id)
    view = await execute_backtest_job(db_session, job_id=receipt.job_id)

    assert view.status == "failed"
    assert view.error is not None
    assert view.error["code"] == "BACKTEST_RUN_ERROR"
    assert "test missing data" in view.error["message"]


@pytest.mark.asyncio
async def test_backtest_job_view_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(BacktestJobNotFoundError):
        await get_backtest_job_view(db_session, job_id=uuid4())


@pytest.mark.asyncio
async def test_create_backtest_job_rejects_foreign_user_context(
    db_session: AsyncSession,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_owner_guard@example.com")
    with pytest.raises(BacktestStrategyNotFoundError):
        await create_backtest_job(
            db_session,
            strategy_id=strategy.id,
            user_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_get_backtest_job_view_rejects_foreign_user_context(
    db_session: AsyncSession,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_job_guard@example.com")
    receipt = await create_backtest_job(db_session, strategy_id=strategy.id)

    with pytest.raises(BacktestJobNotFoundError):
        await get_backtest_job_view(
            db_session,
            job_id=receipt.job_id,
            user_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_create_backtest_job_snapshots_strategy_version(db_session: AsyncSession) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_version_snapshot@example.com")
    receipt = await create_backtest_job(db_session, strategy_id=strategy.id)

    persisted = await db_session.scalar(
        select(BacktestJob).where(BacktestJob.id == receipt.job_id)
    )
    assert persisted is not None
    assert isinstance(persisted.config, dict)
    assert persisted.config.get("strategy_version") == int(strategy.version)


@pytest.mark.asyncio
async def test_create_backtest_job_rejects_invalid_config_values(db_session: AsyncSession) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_invalid_config@example.com")

    with pytest.raises(ValueError, match="initial_capital must be > 0"):
        await create_backtest_job(
            db_session,
            strategy_id=strategy.id,
            initial_capital=0.0,
        )

    with pytest.raises(ValueError, match="commission_rate must be >= 0"):
        await create_backtest_job(
            db_session,
            strategy_id=strategy.id,
            commission_rate=-0.001,
        )

    with pytest.raises(ValueError, match="slippage_bps must be >= 0"):
        await create_backtest_job(
            db_session,
            strategy_id=strategy.id,
            slippage_bps=-1.0,
        )


@pytest.mark.asyncio
async def test_backtest_job_fails_when_stored_strategy_payload_is_invalid(
    db_session: AsyncSession,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_invalid_payload@example.com")
    strategy.dsl_payload = {}  # intentionally broken payload
    await db_session.flush()

    receipt = await create_backtest_job(db_session, strategy_id=strategy.id)
    view = await execute_backtest_job(db_session, job_id=receipt.job_id)

    assert view.status == "failed"
    assert view.error is not None
    assert view.error["code"] == "BACKTEST_RUN_ERROR"
    assert "validation" in view.error["message"].lower()


@pytest.mark.asyncio
async def test_backtest_job_uses_explicit_date_range_over_metadata(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_timerange@example.com")
    captured: dict[str, datetime] = {}

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return _sample_frame()

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        raise AssertionError("metadata should not be fetched when explicit dates are provided")

    monkeypatch.setattr(DataLoader, "load", _fake_load)
    monkeypatch.setattr(DataLoader, "get_symbol_metadata", _fake_metadata)

    receipt = await create_backtest_job(
        db_session,
        strategy_id=strategy.id,
        start_date="2024-01-15T00:00:00+00:00",
        end_date="2024-01-20T00:00:00+00:00",
    )
    view = await execute_backtest_job(db_session, job_id=receipt.job_id)

    assert view.status == "done"
    assert captured["start_date"] == datetime(2024, 1, 15, 0, 0, tzinfo=UTC)
    assert captured["end_date"] == datetime(2024, 1, 20, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_backtest_job_uses_explicit_start_with_metadata_end(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_partial_start@example.com")
    captured: dict[str, datetime] = {}

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return _sample_frame()

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        return {
            "available_timerange": {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-03-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr(DataLoader, "load", _fake_load)
    monkeypatch.setattr(DataLoader, "get_symbol_metadata", _fake_metadata)

    receipt = await create_backtest_job(
        db_session,
        strategy_id=strategy.id,
        start_date="2024-01-15T00:00:00+00:00",
    )
    view = await execute_backtest_job(db_session, job_id=receipt.job_id)

    assert view.status == "done"
    assert captured["start_date"] == datetime(2024, 1, 15, 0, 0, tzinfo=UTC)
    assert captured["end_date"] == datetime(2024, 3, 1, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_backtest_job_uses_metadata_start_with_explicit_end(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="bt_service_partial_end@example.com")
    captured: dict[str, datetime] = {}

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return _sample_frame()

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        return {
            "available_timerange": {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-03-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr(DataLoader, "load", _fake_load)
    monkeypatch.setattr(DataLoader, "get_symbol_metadata", _fake_metadata)

    receipt = await create_backtest_job(
        db_session,
        strategy_id=strategy.id,
        end_date="2024-02-10T00:00:00+00:00",
    )
    view = await execute_backtest_job(db_session, job_id=receipt.job_id)

    assert view.status == "done"
    assert captured["start_date"] == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert captured["end_date"] == datetime(2024, 2, 10, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_schedule_backtest_job_enqueues_worker_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, UUID | str] = {}
    job_id = uuid4()

    def _fake_enqueue(target_job_id: UUID) -> str:
        captured["job_id"] = target_job_id
        return "celery-task-123"

    monkeypatch.setattr("src.engine.backtest.service._enqueue_backtest_job", _fake_enqueue)

    task_id = await schedule_backtest_job(job_id)
    assert task_id == "celery-task-123"
    assert captured["job_id"] == job_id


@pytest.mark.asyncio
async def test_execute_backtest_job_with_fresh_session_initializes_db_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    observed: dict[str, object] = {}

    class _FakeSessionContext:
        async def __aenter__(self) -> str:
            return "fake-session"

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    class _FakeSessionFactory:
        def __call__(self) -> _FakeSessionContext:
            return _FakeSessionContext()

    async def _fake_init_postgres(*, ensure_schema: bool = True) -> None:
        from src.engine.backtest import service as service_module

        observed["ensure_schema"] = ensure_schema
        service_module.db_module.AsyncSessionLocal = _FakeSessionFactory()

    async def _fake_execute_backtest_job(session, *, job_id: UUID, auto_commit: bool):  # noqa: ANN001
        observed["session"] = session
        observed["job_id"] = job_id
        observed["auto_commit"] = auto_commit
        return BacktestJobView(
            job_id=job_id,
            strategy_id=uuid4(),
            status="done",
            progress=100,
            current_step="done",
            result={},
            error=None,
            submitted_at=datetime(2024, 1, 1, tzinfo=UTC),
            completed_at=datetime(2024, 1, 2, tzinfo=UTC),
        )

    monkeypatch.setattr("src.engine.backtest.service.db_module.AsyncSessionLocal", None)
    monkeypatch.setattr("src.engine.backtest.service.db_module.init_postgres", _fake_init_postgres)
    monkeypatch.setattr("src.engine.backtest.service.execute_backtest_job", _fake_execute_backtest_job)

    view = await execute_backtest_job_with_fresh_session(job_id)
    assert view.status == "done"
    assert observed == {
        "ensure_schema": False,
        "session": "fake-session",
        "job_id": job_id,
        "auto_commit": True,
    }
