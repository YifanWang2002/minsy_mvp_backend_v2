from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from src.engine.backtest import EventDrivenBacktestEngine
from src.engine.performance import build_quantstats_performance
from src.engine.strategy import parse_strategy_payload


def _sample_timestamps(length: int) -> list[datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [start + timedelta(hours=index) for index in range(length)]


def _minimal_payload() -> dict:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Perf Strategy"},
        "universe": {"market": "crypto", "tickers": ["BTCUSDT"]},
        "timeframe": "1h",
        "factors": {
            "ema_2": {"type": "ema", "params": {"period": 2, "source": "close"}},
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "gt",
                            "right": 0,
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_quick",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "gt",
                                "right": -1,
                            }
                        },
                    }
                ],
            }
        },
    }


def _sample_ohlcv() -> dict:
    return {
        "open": [100, 100, 100, 100, 101, 102, 103, 102, 101],
        "high": [100.2, 100.2, 100.2, 100.2, 101.2, 102.2, 103.2, 102.2, 101.2],
        "low": [99.8, 99.8, 99.8, 99.8, 100.8, 101.8, 102.8, 101.8, 100.8],
        "close": [100, 100, 100, 100, 101, 102, 103, 102, 101],
        "volume": [1000] * 9,
    }


def test_quantstats_wrapper_returns_stable_serializable_shape() -> None:
    returns = [0.01, -0.005, 0.003, 0.008, -0.002]
    timestamps = _sample_timestamps(len(returns))
    performance = build_quantstats_performance(
        returns=returns,
        timestamps=timestamps,
    )

    assert performance["library"] == "quantstats"
    assert "metrics" in performance
    assert "series" in performance
    assert "cumulative_returns" in performance["series"]
    assert "drawdown" in performance["series"]
    assert "sharpe" in performance["metrics"]

    # Must be JSON serializable for MCP/DB/API responses.
    json.dumps(performance)


def test_backtest_engine_result_contains_performance_block() -> None:
    import pandas as pd

    payload = _minimal_payload()
    strategy = parse_strategy_payload(payload)

    index = _sample_timestamps(9)
    frame = pd.DataFrame(_sample_ohlcv(), index=index)
    result = EventDrivenBacktestEngine(strategy=strategy, data=frame).run()

    assert result.performance["library"] == "quantstats"
    assert "metrics" in result.performance
    assert "series" in result.performance
