from __future__ import annotations

from src.engine.execution.risk_gate import RiskConfig, RiskContext, RiskGate


def test_risk_gate_rejects_when_symbol_exposure_exceeded() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(max_symbol_exposure_pct=0.2),
        context=RiskContext(
            cash=10_000,
            equity=10_000,
            current_symbol_notional=1_500,
            requested_qty=10,
            mark_price=100,
        ),
    )
    assert decision.allowed is False
    assert decision.reason == "symbol_exposure_exceeded"


def test_risk_gate_rejects_when_daily_loss_exceeded() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(max_daily_loss=200),
        context=RiskContext(
            cash=5_000,
            equity=5_000,
            current_symbol_notional=0,
            requested_qty=1,
            mark_price=100,
            realized_pnl_today=-250,
        ),
    )
    assert decision.allowed is False
    assert decision.reason == "daily_loss_limit_exceeded"


def test_risk_gate_accepts_valid_order() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(
            max_position_notional=2_000,
            max_symbol_exposure_pct=0.5,
            min_order_qty=0.01,
            max_daily_loss=500,
        ),
        context=RiskContext(
            cash=5_000,
            equity=5_000,
            current_symbol_notional=500,
            requested_qty=5,
            mark_price=100,
            realized_pnl_today=-100,
        ),
    )
    assert decision.allowed is True
    assert decision.reason == "ok"
