from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from mcp.server.fastmcp import FastMCP

from src.mcp.stress import tools as stress_tools


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


@dataclass
class _FakeReceipt:
    job_id: Any
    strategy_id: Any
    job_type: str
    status: str
    progress: int


@dataclass
class _FakeView:
    job_id: Any
    strategy_id: Any
    base_backtest_job_id: Any
    job_type: str
    status: str
    progress: int
    current_step: str
    config: dict[str, Any]
    summary: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    trials: tuple[dict[str, Any], ...]
    error: dict[str, Any] | None
    submitted_at: datetime
    completed_at: datetime | None


@pytest.mark.asyncio
async def test_stress_tools_create_get_and_pareto(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    job_id = uuid4()
    strategy_id = uuid4()

    async def _fake_create_stress_job(*args: Any, **kwargs: Any) -> _FakeReceipt:  # noqa: ANN401
        return _FakeReceipt(
            job_id=job_id,
            strategy_id=strategy_id,
            job_type="monte_carlo",
            status="pending",
            progress=0,
        )

    async def _fake_schedule_stress_job(_job_id: Any) -> str:  # noqa: ANN401
        return "task-123"

    async def _fake_get_stress_job_view(*args: Any, **kwargs: Any) -> _FakeView:  # noqa: ANN401
        return _FakeView(
            job_id=job_id,
            strategy_id=strategy_id,
            base_backtest_job_id=None,
            job_type="monte_carlo",
            status="done",
            progress=100,
            current_step="completed",
            config={"method": "iid_bootstrap"},
            summary={"risk_of_ruin": 0.12},
            items=(),
            trials=(),
            error=None,
            submitted_at=now,
            completed_at=now,
        )

    async def _fake_execute_stress_job(*args: Any, **kwargs: Any) -> _FakeView:  # noqa: ANN401
        return await _fake_get_stress_job_view()

    async def _fake_get_pareto(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        return [{"x": 10.0, "y": 5.0, "trial_no": 1, "params": {"p": 1}}]

    monkeypatch.setattr(stress_tools, "create_stress_job", _fake_create_stress_job)
    monkeypatch.setattr(stress_tools, "schedule_stress_job", _fake_schedule_stress_job)
    monkeypatch.setattr(stress_tools, "get_stress_job_view", _fake_get_stress_job_view)
    monkeypatch.setattr(stress_tools, "execute_stress_job", _fake_execute_stress_job)
    monkeypatch.setattr(stress_tools, "get_optimization_pareto_points", _fake_get_pareto)
    monkeypatch.setattr(
        stress_tools,
        "list_black_swan_windows",
        lambda market: [
            {
                "window_id": "covid",
                "label": "COVID",
                "market": market,
                "start": "2020-02-20T00:00:00+00:00",
                "end": "2020-04-30T00:00:00+00:00",
            }
        ],
    )

    class _SessionCtx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    async def _fake_new_db_session() -> _SessionCtx:
        return _SessionCtx()

    monkeypatch.setattr(stress_tools, "_new_db_session", _fake_new_db_session)

    mcp = FastMCP("test-stress-tools")
    stress_tools.register_stress_tools(mcp)

    capabilities = _extract_payload(await mcp.call_tool("stress_capabilities", {}))
    assert capabilities["ok"] is True
    assert capabilities["available"] is True

    listed = _extract_payload(
        await mcp.call_tool(
            "stress_black_swan_list_windows",
            {"market": "crypto"},
        )
    )
    assert listed["ok"] is True
    assert listed["count"] == 1

    created = _extract_payload(
        await mcp.call_tool(
            "stress_monte_carlo_create_job",
            {
                "strategy_id": str(strategy_id),
                "num_trials": 200,
                "horizon_bars": 50,
                "run_async": True,
            },
        )
    )
    assert created["ok"] is True
    assert created["stress_job_id"] == str(job_id)
    assert created["queued_task_id"] == "task-123"

    fetched = _extract_payload(
        await mcp.call_tool(
            "stress_monte_carlo_get_job",
            {"stress_job_id": str(job_id)},
        )
    )
    assert fetched["ok"] is True
    assert fetched["summary"]["risk_of_ruin"] == 0.12

    pareto = _extract_payload(
        await mcp.call_tool(
            "stress_optimize_get_pareto",
            {
                "stress_job_id": str(job_id),
                "x_metric": "total_return_pct",
                "y_metric": "max_drawdown_pct",
            },
        )
    )
    assert pareto["ok"] is True
    assert pareto["points"][0]["trial_no"] == 1
