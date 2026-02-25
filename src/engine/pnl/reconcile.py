"""Position reconciliation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PositionView:
    symbol: str
    side: str
    qty: Decimal
    avg_entry_price: Decimal
    mark_price: Decimal


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    to_create: tuple[PositionView, ...]
    to_update: tuple[PositionView, ...]
    to_remove: tuple[PositionView, ...]


def reconcile_positions(
    *,
    local_positions: list[PositionView],
    broker_positions: list[PositionView],
) -> ReconcileResult:
    local_by_symbol = {item.symbol.upper(): item for item in local_positions}
    broker_by_symbol = {item.symbol.upper(): item for item in broker_positions}

    to_create: list[PositionView] = []
    to_update: list[PositionView] = []
    to_remove: list[PositionView] = []

    for symbol, broker_position in broker_by_symbol.items():
        local = local_by_symbol.get(symbol)
        if local is None:
            to_create.append(broker_position)
            continue
        if (
            local.side != broker_position.side
            or local.qty != broker_position.qty
            or local.avg_entry_price != broker_position.avg_entry_price
            or local.mark_price != broker_position.mark_price
        ):
            to_update.append(broker_position)

    for symbol, local_position in local_by_symbol.items():
        if symbol not in broker_by_symbol:
            to_remove.append(local_position)

    return ReconcileResult(
        to_create=tuple(to_create),
        to_update=tuple(to_update),
        to_remove=tuple(to_remove),
    )
