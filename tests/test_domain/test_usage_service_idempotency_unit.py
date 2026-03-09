"""Unit tests for usage idempotency control paths."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.billing.usage_service import MonthlyUsageSnapshot, UsageService


class _FakeDb:
    def __init__(self, scalar_result=None) -> None:
        self._scalar_result = scalar_result

    async def scalar(self, _stmt):
        return self._scalar_result


async def test_record_ai_tokens_skips_increment_when_reference_already_recorded(monkeypatch) -> None:
    service = UsageService(_FakeDb())
    user_id = uuid4()

    increments: list[dict] = []

    async def _fake_insert(self, **_kwargs):
        return False

    async def _fake_increment(self, **kwargs):
        increments.append(kwargs)

    async def _fake_snapshot(self, **_kwargs):
        return MonthlyUsageSnapshot(
            window_month=date(2026, 3, 1),
            ai_input_tokens=10,
            ai_reasoning_tokens=0,
            ai_output_tokens=20,
            ai_total_tokens=30,
            cpu_tokens_total=2,
        )

    monkeypatch.setattr(UsageService, "_insert_usage_event", _fake_insert)
    monkeypatch.setattr(UsageService, "_increment_monthly_usage", _fake_increment)
    monkeypatch.setattr(UsageService, "get_monthly_usage", _fake_snapshot)

    snapshot = await service.record_ai_tokens(
        user_id=user_id,
        input_tokens=5,
        reasoning_tokens=0,
        output_tokens=5,
        source="chat_turn",
        reference_type="assistant_message",
        reference_id="msg_1",
    )

    assert snapshot.ai_total_tokens == 30
    assert increments == []


async def test_record_cpu_tokens_skips_new_insert_when_legacy_reference_exists(monkeypatch) -> None:
    service = UsageService(_FakeDb(scalar_result=SimpleNamespace(id="legacy")))
    user_id = uuid4()

    inserted = False

    async def _fake_insert(self, **_kwargs):
        nonlocal inserted
        inserted = True
        return True

    async def _fake_snapshot(self, **_kwargs):
        return MonthlyUsageSnapshot(
            window_month=date(2026, 3, 1),
            ai_input_tokens=0,
            ai_reasoning_tokens=0,
            ai_output_tokens=0,
            ai_total_tokens=0,
            cpu_tokens_total=7,
        )

    monkeypatch.setattr(UsageService, "_insert_usage_event", _fake_insert)
    monkeypatch.setattr(UsageService, "get_monthly_usage", _fake_snapshot)

    snapshot = await service.record_cpu_tokens(
        user_id=user_id,
        quantity=3,
        source="backtest_job_create",
        reference_type="backtest_job",
        reference_id="job_1",
    )

    assert snapshot.cpu_tokens_total == 7
    assert inserted is False
