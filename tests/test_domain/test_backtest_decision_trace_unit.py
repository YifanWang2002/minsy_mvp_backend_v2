"""Unit tests for trade decision-trace assembly."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from packages.domain.backtest.decision_trace import build_trade_decision_trace


def _build_frame() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    index = [start + timedelta(minutes=i) for i in range(4)]
    return pd.DataFrame(
        {
            "open": [99.0, 100.0, 100.5, 101.0],
            "high": [99.8, 101.0, 100.8, 101.2],
            "low": [98.7, 99.2, 98.8, 100.4],
            "close": [99.5, 100.2, 100.0, 101.0],
            "ema_fast": [99.0, 99.8, 100.6, 100.8],
        },
        index=index,
    )


def _build_strategy() -> SimpleNamespace:
    return SimpleNamespace(
        trade={
            "long": {
                "entry": {
                    "condition": {
                        "cmp": {
                            "left": {"ref": "close"},
                            "op": "gt",
                            "right": {"ref": "ema_fast"},
                        }
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "hard_stop",
                        "stop": {"kind": "points", "value": 1.0},
                    },
                    {
                        "type": "take_profit",
                        "name": "tp",
                        "take": {"kind": "points", "value": 2.0},
                    },
                    {
                        "type": "signal_exit",
                        "name": "ema_flip",
                        "condition": {
                            "cmp": {
                                "left": {"ref": "close"},
                                "op": "lt",
                                "right": {"ref": "ema_fast"},
                            }
                        },
                    },
                ],
            },
            "short": {"entry": {"condition": {"ref": "close"}}, "exits": []},
        }
    )


def test_decision_trace_builds_entry_and_exit_trigger_chain() -> None:
    frame = _build_frame()
    strategy = _build_strategy()
    trade = {
        "side": "long",
        "entry_time": frame.index[1].isoformat(),
        "exit_time": frame.index[2].isoformat(),
        "entry_price": 100.0,
        "exit_reason": "stop_loss",
    }

    trace = build_trade_decision_trace(
        frame=frame,
        strategy=strategy,
        trade=trade,
        entry_index=1,
        exit_index=2,
        start_index=0,
    )

    assert trace["version"] == "1.0"
    assert trace["entry"]["overall_hit"] is True
    assert trace["exit"]["actual_exit_reason"] == "stop_loss"
    assert trace["exit"]["triggered_order"] == 1
    assert trace["exit"]["triggered_rule_id"] == "hard_stop"
    assert len(trace["exit"]["trigger_order"]) >= 3


def test_decision_trace_handles_missing_entry_rule() -> None:
    frame = _build_frame()
    strategy = SimpleNamespace(
        trade={"short": {"entry": {"condition": {"ref": "close"}}}}
    )
    trade = {
        "side": "long",
        "entry_time": frame.index[1].isoformat(),
        "exit_time": frame.index[2].isoformat(),
        "entry_price": 100.0,
        "exit_reason": "signal_exit",
    }

    trace = build_trade_decision_trace(
        frame=frame,
        strategy=strategy,
        trade=trade,
        entry_index=1,
        exit_index=2,
        start_index=0,
    )

    assert trace["entry"]["overall_hit"] is False
    assert trace["entry"]["tree"]["reason_code"] == "missing_entry_condition"
