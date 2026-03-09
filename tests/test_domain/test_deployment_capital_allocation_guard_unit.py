"""Unit tests for deployment capital allocation guardrails."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from packages.domain.exceptions import DomainError
from packages.domain.trading import deployment_ops


class _FakeDb:
    def __init__(self, *, reserved_capital: Decimal) -> None:
        self._reserved_capital = reserved_capital
        self.scalar_calls = 0

    async def scalar(self, _stmt):
        self.scalar_calls += 1
        return self._reserved_capital


def _budget(*, total: str, reserved: str, remaining: str) -> deployment_ops.BrokerCapitalBudget:
    return deployment_ops.BrokerCapitalBudget(
        total_capital=Decimal(total),
        reserved_capital=Decimal(reserved),
        remaining_capital=Decimal(remaining),
        reservation_statuses=("pending", "active", "paused", "error"),
    )


def _resolution(amount: str, source: str = "adapter_fetch") -> deployment_ops.DeploymentCapitalResolution:
    return deployment_ops.DeploymentCapitalResolution(
        amount=Decimal(amount),
        source=source,
    )


@pytest.mark.asyncio
async def test_resolve_broker_capital_budget_uses_default_statuses_and_remaining() -> None:
    db = _FakeDb(reserved_capital=Decimal("325.12"))
    account = SimpleNamespace(id=uuid4(), user_id=uuid4(), mode="paper")

    budget = await deployment_ops.resolve_broker_capital_budget(
        db,
        account=account,
        total_capital=Decimal("1000.00"),
    )

    assert budget.total_capital == Decimal("1000.00")
    assert budget.reserved_capital == Decimal("325.12")
    assert budget.remaining_capital == Decimal("674.88")
    assert budget.reservation_statuses == ("pending", "active", "paused", "error")
    assert db.scalar_calls == 1


@pytest.mark.asyncio
async def test_resolve_broker_capital_budget_caps_remaining_at_zero() -> None:
    db = _FakeDb(reserved_capital=Decimal("1600.00"))
    account = SimpleNamespace(id=uuid4(), user_id=uuid4(), mode="paper")

    budget = await deployment_ops.resolve_broker_capital_budget(
        db,
        account=account,
        total_capital=Decimal("1000.00"),
        reservation_statuses=("active", "paused"),
    )

    assert budget.total_capital == Decimal("1000.00")
    assert budget.reserved_capital == Decimal("1600.00")
    assert budget.remaining_capital == Decimal("0.00")
    assert budget.reservation_statuses == ("active", "paused")


def test_resolve_deployment_capital_with_budget_rejects_over_remaining() -> None:
    with pytest.raises(DomainError) as exc_info:
        deployment_ops.resolve_deployment_capital_with_budget(
            requested_capital=Decimal("700.00"),
            budget=_budget(total="1000.00", reserved="400.00", remaining="600.00"),
            auto_resolution=_resolution("1000.00"),
        )

    assert exc_info.value.detail["code"] == "DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET"


def test_resolve_deployment_capital_with_budget_uses_requested_amount() -> None:
    resolved = deployment_ops.resolve_deployment_capital_with_budget(
        requested_capital=Decimal("250.00"),
        budget=_budget(total="1000.00", reserved="100.00", remaining="900.00"),
        auto_resolution=_resolution("1000.00"),
    )

    assert resolved.amount == Decimal("250.00")
    assert resolved.source == "requested"


def test_resolve_deployment_capital_with_budget_caps_auto_to_remaining() -> None:
    resolved = deployment_ops.resolve_deployment_capital_with_budget(
        requested_capital=Decimal("0.00"),
        budget=_budget(total="1000.00", reserved="600.00", remaining="400.00"),
        auto_resolution=_resolution("1000.00", source="validation_metadata"),
    )

    assert resolved.amount == Decimal("400.00")
    assert resolved.source == "validation_metadata:capped_by_remaining_budget"


def test_resolve_deployment_capital_with_budget_rejects_when_remaining_empty() -> None:
    with pytest.raises(DomainError) as exc_info:
        deployment_ops.resolve_deployment_capital_with_budget(
            requested_capital=Decimal("0.00"),
            budget=_budget(total="1000.00", reserved="1000.00", remaining="0.00"),
            auto_resolution=_resolution("1000.00"),
        )

    assert exc_info.value.detail["code"] == "DEPLOYMENT_CAPITAL_EXCEEDS_REMAINING_BUDGET"
