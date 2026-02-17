from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.backtest.service import execute_backtest_job
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.mcp.context_auth import McpContextClaims
from src.mcp.backtest import tools as backtest_tools
from src.models.session import Session as AgentSession
from src.models.user import User


class _SessionContext:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


async def _create_strategy(db_session: AsyncSession, email: str):
    user = User(email=email, password_hash="hash", name=email)
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
    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)
    return created.strategy


def _claims_for(*, user_id: Any, session_id: Any | None) -> McpContextClaims:
    now = datetime.now(UTC)
    return McpContextClaims(
        user_id=user_id,
        session_id=session_id,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        trace_id="test-trace",
        phase="strategy",
    )


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=160, freq="4h", tz="UTC")
    close = [100 + i * 0.4 for i in range(80)] + [132 - i * 0.35 for i in range(80)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [item + 0.5 for item in close],
            "low": [item - 0.5 for item in close],
            "close": close,
            "volume": [2000.0] * len(close),
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_backtest_tools_create_run_and_fetch_result(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="mcp_backtest_ok@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    async def _fake_schedule(job_id):  # noqa: ANN001
        await execute_backtest_job(db_session, job_id=job_id, auto_commit=True)

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

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(backtest_tools, "schedule_backtest_job", _fake_schedule)
    monkeypatch.setattr("src.engine.backtest.service.DataLoader.load", _fake_load)
    monkeypatch.setattr("src.engine.backtest.service.DataLoader.get_symbol_metadata", _fake_metadata)

    mcp = FastMCP("test-backtest-tools-ok")
    backtest_tools.register_backtest_tools(mcp)

    create_call = await mcp.call_tool(
        "backtest_create_job",
        {
            "strategy_id": str(strategy.id),
            "run_now": False,
        },
    )
    create_payload = _extract_payload(create_call)

    assert create_payload["ok"] is True
    assert create_payload["status"] == "done"
    job_id = create_payload["job_id"]
    assert job_id

    job_call = await mcp.call_tool(
        "backtest_get_job",
        {"job_id": job_id},
    )
    job_payload = _extract_payload(job_call)
    assert job_payload["ok"] is True
    assert job_payload["status"] == "done"
    assert isinstance(job_payload["result"], dict)
    assert "summary" in job_payload["result"]
    assert "performance" in job_payload["result"]
    performance = job_payload["result"]["performance"]
    assert isinstance(performance, dict)
    assert "series" not in performance
    assert "metrics" in performance
    assert "meta" in performance

    hour_call = await mcp.call_tool(
        "backtest_entry_hour_pnl_heatmap",
        {"job_id": job_id},
    )
    hour_payload = _extract_payload(hour_call)
    assert hour_payload["ok"] is True
    assert len(hour_payload["heatmap"]) == 24

    weekday_call = await mcp.call_tool(
        "backtest_entry_weekday_pnl",
        {"job_id": job_id},
    )
    weekday_payload = _extract_payload(weekday_call)
    assert weekday_payload["ok"] is True
    assert len(weekday_payload["weekday_stats"]) == 7

    monthly_call = await mcp.call_tool(
        "backtest_monthly_return_table",
        {"job_id": job_id},
    )
    monthly_payload = _extract_payload(monthly_call)
    assert monthly_payload["ok"] is True
    assert monthly_payload["month_count"] >= 1

    holding_call = await mcp.call_tool(
        "backtest_holding_period_pnl_bins",
        {"job_id": job_id},
    )
    holding_payload = _extract_payload(holding_call)
    assert holding_payload["ok"] is True
    assert len(holding_payload["holding_bins"]) >= 1

    side_call = await mcp.call_tool(
        "backtest_long_short_breakdown",
        {"job_id": job_id},
    )
    side_payload = _extract_payload(side_call)
    assert side_payload["ok"] is True
    assert len(side_payload["side_stats"]) == 2

    exit_call = await mcp.call_tool(
        "backtest_exit_reason_breakdown",
        {"job_id": job_id},
    )
    exit_payload = _extract_payload(exit_call)
    assert exit_payload["ok"] is True
    assert isinstance(exit_payload["exit_reason_stats"], list)

    underwater_call = await mcp.call_tool(
        "backtest_underwater_curve",
        {"job_id": job_id, "max_points": 32},
    )
    underwater_payload = _extract_payload(underwater_call)
    assert underwater_payload["ok"] is True
    assert underwater_payload["point_count"] <= 32
    assert underwater_payload["point_count_total"] >= underwater_payload["point_count"]

    rolling_call = await mcp.call_tool(
        "backtest_rolling_metrics",
        {"job_id": job_id, "max_points": 30},
    )
    rolling_payload = _extract_payload(rolling_call)
    assert rolling_payload["ok"] is True
    assert len(rolling_payload["sharpe_curve_points"]) <= 30
    assert len(rolling_payload["win_rate_curve_points"]) <= 30


@pytest.mark.asyncio
async def test_backtest_tools_pending_and_input_errors(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="mcp_backtest_pending@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    async def _fake_schedule(job_id):  # noqa: ANN001
        return None

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(backtest_tools, "schedule_backtest_job", _fake_schedule)

    mcp = FastMCP("test-backtest-tools-errors")
    backtest_tools.register_backtest_tools(mcp)

    pending_call = await mcp.call_tool(
        "backtest_create_job",
        {
            "strategy_id": str(strategy.id),
            "run_now": False,
        },
    )
    pending_payload = _extract_payload(pending_call)
    assert pending_payload["ok"] is True
    assert pending_payload["status"] == "pending"
    assert pending_payload["result_ready"] is False

    pending_job_call = await mcp.call_tool(
        "backtest_get_job",
        {"job_id": pending_payload["job_id"]},
    )
    pending_job_payload = _extract_payload(pending_job_call)
    assert pending_job_payload["ok"] is True
    assert pending_job_payload["status"] == "pending"
    assert pending_job_payload["result"] is None

    invalid_uuid_call = await mcp.call_tool(
        "backtest_get_job",
        {"job_id": "not-a-uuid"},
    )
    invalid_uuid_payload = _extract_payload(invalid_uuid_call)
    assert invalid_uuid_payload["ok"] is False
    assert invalid_uuid_payload["error"]["code"] == "INVALID_UUID"

    missing_strategy_call = await mcp.call_tool(
        "backtest_create_job",
        {"strategy_id": "f5e818db-4b9a-4d7f-bf67-cfb2cc4f9663"},
    )
    missing_strategy_payload = _extract_payload(missing_strategy_call)
    assert missing_strategy_payload["ok"] is False
    assert missing_strategy_payload["error"]["code"] == "STRATEGY_NOT_FOUND"


@pytest.mark.asyncio
async def test_backtest_tools_failed_run_returns_error_payload(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="mcp_backtest_failed@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    async def _fake_schedule(job_id):  # noqa: ANN001
        await execute_backtest_job(db_session, job_id=job_id, auto_commit=True)

    def _fake_load(
        self,  # noqa: ANN001
        market: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        raise RuntimeError("simulated load failure")

    def _fake_metadata(self, market: str, symbol: str) -> dict[str, object]:  # noqa: ANN001
        return {
            "available_timerange": {
                "start": "2024-01-01T00:00:00+00:00",
                "end": "2024-02-01T00:00:00+00:00",
            }
        }

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(backtest_tools, "schedule_backtest_job", _fake_schedule)
    monkeypatch.setattr("src.engine.backtest.service.DataLoader.load", _fake_load)
    monkeypatch.setattr("src.engine.backtest.service.DataLoader.get_symbol_metadata", _fake_metadata)

    mcp = FastMCP("test-backtest-tools-failed")
    backtest_tools.register_backtest_tools(mcp)

    create_call = await mcp.call_tool(
        "backtest_create_job",
        {
            "strategy_id": str(strategy.id),
            "run_now": False,
        },
    )
    create_payload = _extract_payload(create_call)
    assert create_payload["ok"] is True
    assert create_payload["status"] == "failed"
    assert isinstance(create_payload["error"], dict)
    assert create_payload["error"]["code"] == "BACKTEST_RUN_ERROR"

    job_call = await mcp.call_tool(
        "backtest_get_job",
        {"job_id": create_payload["job_id"]},
    )
    job_payload = _extract_payload(job_call)
    assert job_payload["ok"] is True
    assert job_payload["status"] == "failed"
    assert isinstance(job_payload["result"], dict)
    assert job_payload["result"]["error"]["code"] == "BACKTEST_RUN_ERROR"


@pytest.mark.asyncio
async def test_backtest_create_job_rejects_foreign_strategy_when_context_user_mismatch(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="mcp_backtest_ctx_owner@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    async def _fake_schedule(job_id):  # noqa: ANN001
        return None

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(backtest_tools, "schedule_backtest_job", _fake_schedule)
    monkeypatch.setattr(
        backtest_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(user_id=uuid4(), session_id=strategy.session_id),
    )

    mcp = FastMCP("test-backtest-tools-context-owner")
    backtest_tools.register_backtest_tools(mcp)

    create_call = await mcp.call_tool(
        "backtest_create_job",
        {"strategy_id": str(strategy.id)},
    )
    create_payload = _extract_payload(create_call)
    assert create_payload["ok"] is False
    assert create_payload["error"]["code"] == "STRATEGY_NOT_FOUND"


@pytest.mark.asyncio
async def test_backtest_get_job_rejects_foreign_user_when_context_mismatch(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = await _create_strategy(db_session, email="mcp_backtest_ctx_job@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    async def _fake_schedule(job_id):  # noqa: ANN001
        return None

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(backtest_tools, "schedule_backtest_job", _fake_schedule)
    monkeypatch.setattr(backtest_tools, "_resolve_context_claims", lambda _ctx: None)

    mcp = FastMCP("test-backtest-tools-context-job")
    backtest_tools.register_backtest_tools(mcp)

    create_call = await mcp.call_tool(
        "backtest_create_job",
        {"strategy_id": str(strategy.id)},
    )
    create_payload = _extract_payload(create_call)
    assert create_payload["ok"] is True

    monkeypatch.setattr(
        backtest_tools,
        "_resolve_context_claims",
        lambda _ctx: _claims_for(user_id=uuid4(), session_id=strategy.session_id),
    )
    get_call = await mcp.call_tool(
        "backtest_get_job",
        {"job_id": create_payload["job_id"]},
    )
    get_payload = _extract_payload(get_call)
    assert get_payload["ok"] is False
    assert get_payload["error"]["code"] == "JOB_NOT_FOUND"
