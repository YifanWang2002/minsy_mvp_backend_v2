from __future__ import annotations

from packages.domain.trading.runtime.risk_gate import RiskConfig, RiskContext, RiskGate


def test_000_accessibility_risk_gate_accepts_valid_order() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(min_order_qty=1.0, max_position_notional=10_000.0),
        context=RiskContext(
            cash=50_000.0,
            equity=60_000.0,
            current_symbol_notional=0.0,
            requested_qty=2.0,
            mark_price=100.0,
            realized_pnl_today=0.0,
        ),
    )
    assert decision.allowed is True
    assert decision.reason == "ok"


def test_010_risk_gate_rejects_symbol_exposure() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(max_symbol_exposure_pct=0.1),
        context=RiskContext(
            cash=10_000.0,
            equity=10_000.0,
            current_symbol_notional=800.0,
            requested_qty=3.0,
            mark_price=100.0,
            realized_pnl_today=0.0,
        ),
    )
    assert decision.allowed is False
    assert decision.reason == "symbol_exposure_exceeded"


def test_020_risk_gate_rejects_daily_loss_limit() -> None:
    gate = RiskGate()
    decision = gate.evaluate(
        config=RiskConfig(max_daily_loss=500.0),
        context=RiskContext(
            cash=10_000.0,
            equity=12_000.0,
            current_symbol_notional=0.0,
            requested_qty=1.0,
            mark_price=100.0,
            realized_pnl_today=-600.0,
        ),
    )
    assert decision.allowed is False
    assert decision.reason == "daily_loss_limit_exceeded"
