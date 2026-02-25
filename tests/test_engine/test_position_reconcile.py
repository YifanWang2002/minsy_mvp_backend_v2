from __future__ import annotations

from decimal import Decimal

from src.engine.pnl.reconcile import PositionView, reconcile_positions


def test_reconcile_positions_detects_create_update_remove() -> None:
    local = [
        PositionView(
            symbol="AAPL",
            side="long",
            qty=Decimal("1"),
            avg_entry_price=Decimal("100"),
            mark_price=Decimal("101"),
        ),
        PositionView(
            symbol="TSLA",
            side="long",
            qty=Decimal("2"),
            avg_entry_price=Decimal("200"),
            mark_price=Decimal("205"),
        ),
    ]
    broker = [
        PositionView(
            symbol="aapl",
            side="long",
            qty=Decimal("1.5"),
            avg_entry_price=Decimal("100"),
            mark_price=Decimal("102"),
        ),
        PositionView(
            symbol="NVDA",
            side="short",
            qty=Decimal("3"),
            avg_entry_price=Decimal("500"),
            mark_price=Decimal("490"),
        ),
    ]

    result = reconcile_positions(local_positions=local, broker_positions=broker)

    assert len(result.to_create) == 1
    assert result.to_create[0].symbol == "NVDA"

    assert len(result.to_update) == 1
    assert result.to_update[0].symbol.lower() == "aapl"

    assert len(result.to_remove) == 1
    assert result.to_remove[0].symbol == "TSLA"
