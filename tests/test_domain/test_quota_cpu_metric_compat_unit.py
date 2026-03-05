"""Unit tests for CPU quota metric backward compatibility."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.billing.quota_service import QuotaService
from packages.domain.billing.usage_service import UsageMetric


class _UsageStub:
    async def get_monthly_usage(self, *, user_id):
        del user_id
        return SimpleNamespace(
            window_month=date(2026, 3, 1),
            ai_total_tokens=0,
            cpu_tokens_total=12,
        )

    async def get_capacity_snapshot(self, *, user_id):
        del user_id
        return SimpleNamespace(
            strategies_current_count=0,
            deployments_running_count=0,
        )


async def test_legacy_cpu_jobs_metric_maps_to_cpu_tokens(monkeypatch):
    quota = QuotaService(_UsageStub())

    monkeypatch.setattr(
        quota,
        "_limits_for_tier",
        lambda _tier: {"cpu_jobs_monthly_total": 30},
    )

    snapshot = await quota.resolve_metric_snapshot(
        user_id=uuid4(),
        tier="free",
        metric=UsageMetric.LEGACY_CPU_JOBS_MONTHLY_TOTAL,
    )

    assert snapshot.metric == UsageMetric.CPU_TOKENS_MONTHLY_TOTAL
    assert snapshot.used == 12
    assert snapshot.limit == 30
    assert snapshot.remaining == 18
