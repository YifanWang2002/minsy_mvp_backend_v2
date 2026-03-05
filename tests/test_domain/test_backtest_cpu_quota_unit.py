"""Unit tests for bars-based CPU quota accounting on backtest submission."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.backtest import service as backtest_service
from packages.domain.billing.usage_service import UsageMetric


class _FakeDb:
    def __init__(self, scalar_results: list[object]) -> None:
        self._scalar_results = list(scalar_results)
        self.added: list[object] = []

    async def scalar(self, _stmt):
        if not self._scalar_results:
            return None
        return self._scalar_results.pop(0)

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        for obj in self.added:
            if hasattr(obj, "submitted_at") and getattr(obj, "submitted_at") is None:
                setattr(obj, "submitted_at", datetime.now(UTC))
        return None

    async def commit(self) -> None:
        return None

    async def refresh(self, _obj: object) -> None:
        return None


def _fake_validate_values(
    *,
    initial_capital: float,
    commission_rate: float,
    slippage_bps: float,
) -> tuple[float, float, float]:
    return initial_capital, commission_rate, slippage_bps


async def test_create_backtest_job_uses_cpu_tokens_for_quota_and_usage(monkeypatch):
    strategy_id = uuid4()
    user_id = uuid4()
    strategy = SimpleNamespace(
        id=strategy_id,
        user_id=user_id,
        session_id=uuid4(),
        version=3,
        dsl_payload={},
    )
    owner = SimpleNamespace(current_tier="free")
    db = _FakeDb([strategy, owner])

    quota_calls: list[dict] = []
    usage_calls: list[dict] = []

    class _QuotaServiceCapture:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **kwargs):
            quota_calls.append(kwargs)

    class _UsageServiceCapture:
        def __init__(self, _db) -> None:
            del _db

        async def record_cpu_tokens(self, **kwargs):
            usage_calls.append(kwargs)

    def _fake_enforce(*, strategy_payload: dict, config: dict) -> None:
        del strategy_payload
        config["estimated_bars"] = 250_001

    conversion_calls: list[dict] = []

    def _fake_cpu_conversion(*, estimated_bars: int | None, billing_cost_model):
        conversion_calls.append(
            {
                "estimated_bars": estimated_bars,
                "billing_cost_model": billing_cost_model,
            }
        )
        return SimpleNamespace(
            estimated_bars=int(estimated_bars or 0),
            bars_per_token=100_000,
            token_quantity=3,
        )

    monkeypatch.setattr(backtest_service, "QuotaService", _QuotaServiceCapture)
    monkeypatch.setattr(backtest_service, "UsageService", _UsageServiceCapture)
    monkeypatch.setattr(
        backtest_service,
        "_validated_backtest_config_values",
        _fake_validate_values,
    )
    monkeypatch.setattr(
        backtest_service,
        "_enforce_backtest_bar_limit_at_submission",
        _fake_enforce,
    )
    monkeypatch.setattr(
        backtest_service,
        "compute_cpu_tokens_from_bars",
        _fake_cpu_conversion,
    )

    await backtest_service.create_backtest_job(
        db,
        strategy_id=strategy_id,
        user_id=user_id,
        auto_commit=False,
    )

    assert len(conversion_calls) == 1
    assert conversion_calls[0]["estimated_bars"] == 250_001

    assert len(quota_calls) == 1
    assert quota_calls[0]["metric"] == UsageMetric.CPU_TOKENS_MONTHLY_TOTAL
    assert quota_calls[0]["increment"] == 3

    assert len(usage_calls) == 1
    assert usage_calls[0]["quantity"] == 3
    assert usage_calls[0]["metadata"]["estimated_bars"] == 250_001
    assert usage_calls[0]["metadata"]["cpu_tokens_increment"] == 3


async def test_create_backtest_job_defaults_to_one_cpu_token_when_no_bar_estimate(monkeypatch):
    strategy_id = uuid4()
    user_id = uuid4()
    strategy = SimpleNamespace(
        id=strategy_id,
        user_id=user_id,
        session_id=uuid4(),
        version=1,
        dsl_payload={},
    )
    owner = SimpleNamespace(current_tier="free")
    db = _FakeDb([strategy, owner])

    quota_calls: list[dict] = []

    class _QuotaServiceCapture:
        def __init__(self, _usage) -> None:
            del _usage

        async def assert_quota_available(self, **kwargs):
            quota_calls.append(kwargs)

    class _UsageServiceCapture:
        def __init__(self, _db) -> None:
            del _db

        async def record_cpu_tokens(self, **kwargs):
            del kwargs

    def _fake_enforce(*, strategy_payload: dict, config: dict) -> None:
        del strategy_payload, config

    def _fake_cpu_conversion(*, estimated_bars: int | None, billing_cost_model):
        del billing_cost_model
        assert estimated_bars is None
        return SimpleNamespace(
            estimated_bars=0,
            bars_per_token=100_000,
            token_quantity=1,
        )

    monkeypatch.setattr(backtest_service, "QuotaService", _QuotaServiceCapture)
    monkeypatch.setattr(backtest_service, "UsageService", _UsageServiceCapture)
    monkeypatch.setattr(
        backtest_service,
        "_validated_backtest_config_values",
        _fake_validate_values,
    )
    monkeypatch.setattr(
        backtest_service,
        "_enforce_backtest_bar_limit_at_submission",
        _fake_enforce,
    )
    monkeypatch.setattr(
        backtest_service,
        "compute_cpu_tokens_from_bars",
        _fake_cpu_conversion,
    )

    await backtest_service.create_backtest_job(
        db,
        strategy_id=strategy_id,
        user_id=user_id,
        auto_commit=False,
    )

    assert len(quota_calls) == 1
    assert quota_calls[0]["increment"] == 1
