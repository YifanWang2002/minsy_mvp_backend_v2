"""Unit tests for deployment activation quota guard."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from packages.domain.billing.usage_service import UsageMetric
from packages.domain.trading import deployment_ops


class _FakeDb:
    def __init__(self, tier: str = "free") -> None:
        self._tier = tier

    async def scalar(self, _stmt):
        return SimpleNamespace(current_tier=self._tier)


async def test_activation_quota_guard_checks_running_deployments(monkeypatch):
    user_id = uuid4()
    calls: list[dict] = []

    class _QuotaServiceCapture:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(deployment_ops, "QuotaService", _QuotaServiceCapture)
    monkeypatch.setattr(deployment_ops, "UsageService", lambda _db: object())

    await deployment_ops._assert_running_deployment_quota_for_activation(
        _FakeDb(),
        user_id=user_id,
        current_status="paused",
    )

    assert len(calls) == 1
    assert calls[0]["metric"] == UsageMetric.DEPLOYMENTS_RUNNING_COUNT
    assert calls[0]["increment"] == 1
    assert calls[0]["user_id"] == user_id


async def test_activation_quota_guard_skips_when_already_active(monkeypatch):
    called = False

    class _QuotaServiceCapture:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **_kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(deployment_ops, "QuotaService", _QuotaServiceCapture)
    monkeypatch.setattr(deployment_ops, "UsageService", lambda _db: object())

    await deployment_ops._assert_running_deployment_quota_for_activation(
        _FakeDb(),
        user_id=uuid4(),
        current_status="active",
    )

    assert called is False
