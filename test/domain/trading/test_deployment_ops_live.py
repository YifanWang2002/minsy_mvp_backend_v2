from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from packages.domain.trading import deployment_ops


async def test_000_accessibility_resolve_deployment_capital_uses_account_metadata_starting_cash() -> None:
    account = SimpleNamespace(
        validation_metadata={},
        metadata_={"starting_cash": "25000"},
    )

    resolved = await deployment_ops._resolve_deployment_capital_allocated(
        requested_capital=Decimal("0"),
        account=account,
    )

    assert resolved == Decimal("25000.00")


async def test_010_accessibility_resolve_deployment_capital_prefers_explicit_request() -> None:
    account = SimpleNamespace(
        validation_metadata={"paper_equity": "8000"},
        metadata_={"starting_cash": "25000"},
    )

    resolved = await deployment_ops._resolve_deployment_capital_allocated(
        requested_capital=Decimal("1500"),
        account=account,
    )

    assert resolved == Decimal("1500.00")


async def test_015_accessibility_resolve_deployment_capital_reports_resolution_source() -> None:
    account = SimpleNamespace(
        validation_metadata={},
        metadata_={"starting_cash": "25000"},
    )

    resolution = await deployment_ops.resolve_deployment_capital(
        requested_capital=Decimal("0"),
        account=account,
    )

    assert resolution.amount == Decimal("25000.00")
    assert resolution.source == "account_metadata"


def test_020_accessibility_runtime_compatibility_blocks_multi_symbol() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD", "ETHUSD"]},
            "trade": {},
        }
    )

    assert compatibility.status == "blocked"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_MULTI_SYMBOL" in compatibility.blocker_codes


def test_030_accessibility_runtime_compatibility_allows_dsl_exit_rules() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD"]},
            "trade": {
                "long": {
                    "exits": [
                        {"type": "signal_exit"},
                        {"type": "stop_loss"},
                    ]
                }
            },
        }
    )

    assert compatibility.status == "ok"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_EXIT_RULE" not in compatibility.blocker_codes


def test_040_accessibility_runtime_compatibility_blocks_unsupported_timeframe() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD"]},
            "timeframe": "7m",
            "trade": {},
        }
    )

    assert compatibility.status == "blocked"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_TIMEFRAME" in compatibility.blocker_codes
