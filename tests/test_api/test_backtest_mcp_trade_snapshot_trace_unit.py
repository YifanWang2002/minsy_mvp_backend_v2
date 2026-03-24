"""Unit tests for MCP backtest_trade_snapshots decision-trace passthrough."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.mcp.domains.backtest import tools as backtest_tools


class _FakeMcp:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def _decorator(func):
            self.tools[func.__name__] = func
            return func

        return _decorator


class _FakeDbSessionCtx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


@pytest.mark.anyio
async def test_backtest_trade_snapshots_mcp_passes_include_decision_trace(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_new_db_session():
        return _FakeDbSessionCtx()

    async def _fake_build_backtest_trade_snapshots(*args, **kwargs):  # noqa: ANN002, ANN003
        del args
        captured.update(kwargs)
        return {
            "job_id": str(kwargs["job_id"]),
            "strategy_id": str(uuid4()),
            "status": "done",
            "selection": {"selected_count": 1},
            "window": {"lookback_bars": kwargs.get("lookback_bars", 0)},
            "snapshots": [],
            "warnings": [],
        }

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        backtest_tools,
        "build_backtest_trade_snapshots",
        _fake_build_backtest_trade_snapshots,
    )
    monkeypatch.setattr(
        backtest_tools,
        "_resolve_context_claims",
        lambda _ctx: SimpleNamespace(user_id=uuid4()),
    )

    fake_mcp = _FakeMcp()
    backtest_tools.register_backtest_tools(fake_mcp)

    tool = fake_mcp.tools["backtest_trade_snapshots"]
    raw = await tool(
        job_id=str(uuid4()),
        selection_mode="latest",
        selection_count=1,
        include_decision_trace=True,
        ctx=object(),
    )
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["tool"] == "backtest_trade_snapshots"
    assert captured["include_decision_trace"] is True
