"""Unit tests for MCP backtest quota error payload enrichment."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from apps.mcp.domains.backtest import tools as backtest_tools
from packages.domain.billing.quota_service import QuotaExceededError


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
async def test_backtest_create_job_quota_error_contains_snapshot_fields(monkeypatch):
    async def _fake_new_db_session():
        return _FakeDbSessionCtx()

    async def _fake_create_backtest_job(*args, **kwargs):
        del args, kwargs
        raise QuotaExceededError(
            metric="cpu_tokens_monthly_total",
            tier="free",
            used=30,
            limit=30,
            remaining=0,
            reset_at=datetime(2026, 4, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(backtest_tools, "_new_db_session", _fake_new_db_session)
    monkeypatch.setattr(
        backtest_tools, "create_backtest_job", _fake_create_backtest_job
    )

    fake_mcp = _FakeMcp()
    backtest_tools.register_backtest_tools(fake_mcp)

    tool = fake_mcp.tools["backtest_create_job"]
    raw = await tool(strategy_id=str(uuid4()))
    payload = json.loads(raw)

    assert payload["ok"] is False
    assert payload["tool"] == "backtest_create_job"
    assert payload["error"]["code"] == "QUOTA_EXCEEDED"
    assert payload["error"]["status_code"] == 402
    assert payload["error"]["metric"] == "cpu_tokens_monthly_total"
    assert payload["error"]["tier"] == "free"
    assert payload["error"]["used"] == 30
    assert payload["error"]["limit"] == 30
    assert payload["error"]["remaining"] == 0
    assert payload["error"]["reset_at"] == "2026-04-01T00:00:00+00:00"
