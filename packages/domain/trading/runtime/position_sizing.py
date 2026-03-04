"""Resolve default open-position sizing for trading runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

_DEFAULT_POSITION_PCT = Decimal("0.025")


@dataclass(frozen=True, slots=True)
class PositionSizingRequest:
    """Inputs needed to resolve an opening order quantity."""

    strategy_payload: dict[str, Any]
    signal: str
    mark_price: Decimal
    account_cash: Decimal
    account_equity: Decimal
    risk_limits: dict[str, Any] = field(default_factory=dict)
    default_position_pct: Decimal = _DEFAULT_POSITION_PCT


@dataclass(frozen=True, slots=True)
class PositionSizingResult:
    """Resolved quantity and provenance for one opening order."""

    qty: Decimal
    sizing_mode: str
    source: str
    notional: Decimal
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_positive_decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _zero_result(*, source: str, sizing_mode: str, reason: str) -> PositionSizingResult:
    return PositionSizingResult(
        qty=Decimal("0"),
        sizing_mode=sizing_mode,
        source=source,
        notional=Decimal("0"),
        metadata={"reason": reason},
    )


def _normalize_budget(value: Decimal) -> Decimal:
    return value if value > 0 else Decimal("0")


def _resolve_side_position_sizing(
    *,
    strategy_payload: dict[str, Any],
    signal: str,
) -> dict[str, Any] | None:
    trade = strategy_payload.get("trade")
    if not isinstance(trade, dict):
        return None
    side_key = "short" if str(signal).strip().upper() == "OPEN_SHORT" else "long"
    side_payload = trade.get(side_key)
    if not isinstance(side_payload, dict):
        return None
    sizing = side_payload.get("position_sizing")
    return dict(sizing) if isinstance(sizing, dict) else None


def resolve_open_position_size(*, request: PositionSizingRequest) -> PositionSizingResult:
    """Resolve a quantity for OPEN_LONG / OPEN_SHORT requests.

    Priority:
    1. Explicit deployment override: ``risk_limits.order_qty``
    2. Structured override: ``risk_limits.position_sizing_override``
    3. Strategy DSL ``trade.{long|short}.position_sizing``
    4. Default fallback: ``default_position_pct`` of current equity, capped by available cash
    """

    mark_price = _to_positive_decimal(request.mark_price)
    if mark_price is None:
        return _zero_result(
            source="invalid_mark_price",
            sizing_mode="none",
            reason="invalid_mark_price",
        )

    available_cash = _normalize_budget(request.account_cash)
    available_equity = _normalize_budget(request.account_equity)

    explicit_qty = _to_positive_decimal((request.risk_limits or {}).get("order_qty"))
    if explicit_qty is not None:
        return PositionSizingResult(
            qty=explicit_qty,
            sizing_mode="fixed_qty",
            source="risk_limits.order_qty",
            notional=explicit_qty * mark_price,
            metadata={},
        )

    sizing_payload: dict[str, Any] | None = None
    override = (request.risk_limits or {}).get("position_sizing_override")
    if isinstance(override, dict):
        sizing_payload = dict(override)
        source = "risk_limits.position_sizing_override"
    else:
        sizing_payload = _resolve_side_position_sizing(
            strategy_payload=request.strategy_payload,
            signal=request.signal,
        )
        source = "strategy.position_sizing"

    if isinstance(sizing_payload, dict):
        mode = str(sizing_payload.get("mode") or "").strip().lower()
        if mode == "fixed_qty":
            qty = _to_positive_decimal(sizing_payload.get("qty"))
            if qty is None:
                return _zero_result(source=source, sizing_mode="fixed_qty", reason="invalid_fixed_qty")
            return PositionSizingResult(
                qty=qty,
                sizing_mode=mode,
                source=source,
                notional=qty * mark_price,
                metadata={},
            )
        if mode == "fixed_cash":
            cash_budget = _to_positive_decimal(sizing_payload.get("cash"))
            if cash_budget is None:
                return _zero_result(source=source, sizing_mode="fixed_cash", reason="invalid_fixed_cash")
            budget = min(cash_budget, available_cash)
            if budget <= 0:
                return _zero_result(source=source, sizing_mode="fixed_cash", reason="insufficient_cash")
            qty = budget / mark_price
            return PositionSizingResult(
                qty=qty,
                sizing_mode=mode,
                source=source,
                notional=budget,
                metadata={"cash_capped": budget != cash_budget},
            )
        if mode == "pct_equity":
            pct = _to_positive_decimal(sizing_payload.get("pct"))
            if pct is None or pct > Decimal("1"):
                return _zero_result(source=source, sizing_mode="pct_equity", reason="invalid_pct_equity")
            budget = min(available_equity * pct, available_cash)
            if budget <= 0:
                return _zero_result(source=source, sizing_mode="pct_equity", reason="insufficient_cash")
            qty = budget / mark_price
            return PositionSizingResult(
                qty=qty,
                sizing_mode=mode,
                source=source,
                notional=budget,
                metadata={"cash_capped": budget < (available_equity * pct)},
            )

    default_pct = _to_positive_decimal(request.default_position_pct) or _DEFAULT_POSITION_PCT
    budget = min(available_equity * default_pct, available_cash)
    if budget <= 0:
        return _zero_result(
            source="default_pct_equity",
            sizing_mode="pct_equity",
            reason="insufficient_cash",
        )
    qty = budget / mark_price
    return PositionSizingResult(
        qty=qty,
        sizing_mode="pct_equity",
        source="default_pct_equity",
        notional=budget,
        metadata={"default_pct": str(default_pct), "cash_capped": budget < (available_equity * default_pct)},
    )
