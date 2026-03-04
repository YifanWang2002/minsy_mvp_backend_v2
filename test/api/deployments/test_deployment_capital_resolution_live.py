from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from apps.api.routes import deployments as deployments_route


async def test_000_accessibility_route_capital_resolution_uses_account_metadata_starting_cash() -> None:
    account = SimpleNamespace(
        validation_metadata={},
        metadata_={"starting_cash": "42000"},
    )

    resolved = await deployments_route._resolve_deployment_capital_allocated(
        requested_capital=Decimal("0"),
        account=account,
    )

    assert resolved == Decimal("42000.00")
