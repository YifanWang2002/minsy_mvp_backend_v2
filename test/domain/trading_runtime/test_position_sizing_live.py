from __future__ import annotations

from decimal import Decimal

from packages.domain.trading.runtime.position_sizing import (
    PositionSizingRequest,
    resolve_open_position_size,
)


def _strategy_with_sizing(sizing: dict[str, object]) -> dict[str, object]:
    return {
        "trade": {
            "long": {
                "position_sizing": sizing,
            }
        }
    }


def test_000_resolve_open_position_size_uses_risk_limit_order_qty_override() -> None:
    result = resolve_open_position_size(
        request=PositionSizingRequest(
            strategy_payload=_strategy_with_sizing({"mode": "pct_equity", "pct": 0.25}),
            signal="OPEN_LONG",
            mark_price=Decimal("100"),
            account_cash=Decimal("1000"),
            account_equity=Decimal("1000"),
            risk_limits={"order_qty": 3},
        )
    )

    assert result.qty == Decimal("3")
    assert result.source == "risk_limits.order_qty"
    assert result.sizing_mode == "fixed_qty"
    assert result.notional == Decimal("300")


def test_010_resolve_open_position_size_uses_fixed_cash_sizing() -> None:
    result = resolve_open_position_size(
        request=PositionSizingRequest(
            strategy_payload=_strategy_with_sizing({"mode": "fixed_cash", "cash": 500}),
            signal="OPEN_LONG",
            mark_price=Decimal("100"),
            account_cash=Decimal("1000"),
            account_equity=Decimal("1000"),
        )
    )

    assert result.qty == Decimal("5")
    assert result.source == "strategy.position_sizing"
    assert result.sizing_mode == "fixed_cash"
    assert result.notional == Decimal("500")
    assert result.metadata["cash_capped"] is False


def test_020_resolve_open_position_size_uses_pct_equity_and_caps_by_cash() -> None:
    result = resolve_open_position_size(
        request=PositionSizingRequest(
            strategy_payload=_strategy_with_sizing({"mode": "pct_equity", "pct": 0.25}),
            signal="OPEN_LONG",
            mark_price=Decimal("10"),
            account_cash=Decimal("10"),
            account_equity=Decimal("1000"),
        )
    )

    assert result.qty == Decimal("1")
    assert result.source == "strategy.position_sizing"
    assert result.sizing_mode == "pct_equity"
    assert result.notional == Decimal("10")
    assert result.metadata["cash_capped"] is True


def test_030_resolve_open_position_size_falls_back_to_default_2_5_pct_equity() -> None:
    result = resolve_open_position_size(
        request=PositionSizingRequest(
            strategy_payload={},
            signal="OPEN_LONG",
            mark_price=Decimal("50"),
            account_cash=Decimal("1000"),
            account_equity=Decimal("1000"),
        )
    )

    assert result.qty == Decimal("0.5")
    assert result.source == "default_pct_equity"
    assert result.sizing_mode == "pct_equity"
    assert result.notional == Decimal("25")
    assert result.metadata["default_pct"] == "0.025"


def test_040_resolve_open_position_size_uses_structured_override_before_dsl() -> None:
    result = resolve_open_position_size(
        request=PositionSizingRequest(
            strategy_payload=_strategy_with_sizing({"mode": "fixed_qty", "qty": 10}),
            signal="OPEN_LONG",
            mark_price=Decimal("25"),
            account_cash=Decimal("1000"),
            account_equity=Decimal("1000"),
            risk_limits={"position_sizing_override": {"mode": "fixed_cash", "cash": 100}},
        )
    )

    assert result.qty == Decimal("4")
    assert result.source == "risk_limits.position_sizing_override"
    assert result.sizing_mode == "fixed_cash"
    assert result.notional == Decimal("100")
