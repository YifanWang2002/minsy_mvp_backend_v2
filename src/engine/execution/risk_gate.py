"""Risk gate checks before creating order intents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_position_notional: float | None = None
    max_symbol_exposure_pct: float | None = None
    min_order_qty: float = 0.0
    max_daily_loss: float | None = None


@dataclass(frozen=True, slots=True)
class RiskContext:
    cash: float
    equity: float
    current_symbol_notional: float
    requested_qty: float
    mark_price: float
    realized_pnl_today: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskGate:
    """Evaluates a request against deployment/account risk constraints."""

    def evaluate(self, *, config: RiskConfig, context: RiskContext) -> RiskDecision:
        if context.requested_qty <= 0:
            return RiskDecision(allowed=False, reason="qty_must_be_positive")
        if context.mark_price <= 0:
            return RiskDecision(allowed=False, reason="invalid_mark_price")
        if context.requested_qty < config.min_order_qty:
            return RiskDecision(allowed=False, reason="below_min_order_qty")

        requested_notional = context.requested_qty * context.mark_price
        if config.max_position_notional is not None and requested_notional > config.max_position_notional:
            return RiskDecision(allowed=False, reason="max_position_notional_exceeded")

        if requested_notional > context.cash:
            return RiskDecision(allowed=False, reason="insufficient_cash")

        if (
            config.max_symbol_exposure_pct is not None
            and context.equity > 0
            and (context.current_symbol_notional + requested_notional)
            > (context.equity * config.max_symbol_exposure_pct)
        ):
            return RiskDecision(allowed=False, reason="symbol_exposure_exceeded")

        if (
            config.max_daily_loss is not None
            and context.realized_pnl_today <= -abs(config.max_daily_loss)
        ):
            return RiskDecision(allowed=False, reason="daily_loss_limit_exceeded")

        return RiskDecision(allowed=True, reason="ok")
