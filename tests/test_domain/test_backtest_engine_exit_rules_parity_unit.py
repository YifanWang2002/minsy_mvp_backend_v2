"""Unit tests that engine stop/take resolution matches shared exit-rules logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from packages.domain.backtest.engine import EventDrivenBacktestEngine
from packages.domain.backtest.exit_rules import resolve_position_stops
from packages.domain.backtest.types import PositionSide


def _build_frame() -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    index = [start + timedelta(minutes=i) for i in range(6)]
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 102, 101],
            "high": [101, 102, 103, 104, 103, 102],
            "low": [99, 100, 101, 102, 101, 100],
            "close": [100.5, 101.5, 102.5, 103.5, 102.5, 101.5],
            "volume": [1000, 1001, 1002, 1003, 1004, 1005],
            "atr_14": [1.2, 1.3, 1.4, 1.5, 1.4, 1.3],
        },
        index=index,
    )


def _build_strategy() -> SimpleNamespace:
    return SimpleNamespace(
        factors={},
        trade={},
        universe=SimpleNamespace(market="crypto", tickers=("BTCUSD",), timeframe="1m"),
    )


def test_engine_and_exit_rules_resolve_same_stop_take_prices() -> None:
    frame = _build_frame()
    engine = EventDrivenBacktestEngine(strategy=_build_strategy(), data=frame)
    side_payload = {
        "exits": [
            {
                "type": "stop_loss",
                "name": "hard_stop",
                "stop": {"kind": "points", "value": 2.0},
            },
            {
                "type": "bracket_rr",
                "name": "rr_take",
                "risk_reward": 2.0,
                "stop": {"kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 1.0},
            },
        ]
    }
    entry_price = 101.0
    bar_index = 2

    engine_stop, engine_take = engine._resolve_position_stops(
        side_payload=side_payload,
        side=PositionSide.LONG,
        entry_price=entry_price,
        bar_index=bar_index,
    )
    shared_stop, shared_take = resolve_position_stops(
        side_payload=side_payload,
        side=PositionSide.LONG,
        frame=engine.frame,
        entry_price=entry_price,
        bar_index=bar_index,
    )

    assert engine_stop == shared_stop
    assert engine_take == shared_take
