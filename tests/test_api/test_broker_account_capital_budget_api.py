"""API route tests for broker account capital-budget endpoint."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.api.routes import broker_accounts as broker_accounts_routes
from packages.domain.trading.deployment_ops import BrokerCapitalBudget


class _FakeDbSession:
    def __init__(self, account: SimpleNamespace | None) -> None:
        self._account = account

    async def scalar(self, _stmt):
        return self._account


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
async def test_get_broker_account_capital_budget_returns_budget(monkeypatch) -> None:
    user = _fake_user()
    account = SimpleNamespace(id=uuid4(), user_id=user.id, mode="paper")
    db = _FakeDbSession(account)

    async def _fake_resolve_budget(_db, *, account):
        assert account is not None
        return BrokerCapitalBudget(
            total_capital=Decimal("1000.00"),
            reserved_capital=Decimal("250.00"),
            remaining_capital=Decimal("750.00"),
            reservation_statuses=("pending", "active"),
        )

    monkeypatch.setattr(
        broker_accounts_routes,
        "resolve_broker_capital_budget",
        _fake_resolve_budget,
    )

    response = await broker_accounts_routes.get_broker_account_capital_budget(
        broker_account_id=account.id,
        user=user,
        db=db,
    )

    assert response.broker_account_id == account.id
    assert response.total_capital == 1000.0
    assert response.reserved_capital == 250.0
    assert response.remaining_capital == 750.0
    assert response.reservation_statuses == ["pending", "active"]
    assert response.as_of is not None


@pytest.mark.asyncio
async def test_get_broker_account_capital_budget_rejects_unowned_account() -> None:
    user = _fake_user()
    db = _FakeDbSession(account=None)

    with pytest.raises(broker_accounts_routes.HTTPException) as exc_info:
        await broker_accounts_routes.get_broker_account_capital_budget(
            broker_account_id=uuid4(),
            user=user,
            db=db,
        )

    assert exc_info.value.status_code == 404
    detail = exc_info.value.detail
    assert detail["code"] == "BROKER_ACCOUNT_NOT_FOUND"
