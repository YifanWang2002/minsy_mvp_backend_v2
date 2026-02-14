from __future__ import annotations

from datetime import UTC
from typing import Any

import pandas as pd
import pytest

from src.engine.backtest import (
    BacktestConfig,
    BacktestEventType,
    EventDrivenBacktestEngine,
)
from src.engine.strategy import parse_strategy_payload


def _ohlcv_from_close(close: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    close_values = [float(item) for item in close]
    return pd.DataFrame(
        {
            "open": close_values,
            "high": [item + 0.2 for item in close_values],
            "low": [item - 0.2 for item in close_values],
            "close": close_values,
            "volume": [1000.0] * len(close_values),
        },
        index=index,
    )


def _signal_strategy_payload() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Signal Exit Strategy"},
        "universe": {"market": "crypto", "tickers": ["BTCUSDT"]},
        "timeframe": "1h",
        "factors": {
            "ema_2": {"type": "ema", "params": {"period": 2, "source": "close"}},
            "ema_4": {"type": "ema", "params": {"period": 4, "source": "close"}},
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_2"},
                            "op": "cross_above",
                            "b": {"ref": "ema_4"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_signal_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_2"},
                                "op": "cross_below",
                                "b": {"ref": "ema_4"},
                            }
                        },
                    }
                ],
            }
        },
    }


def _stop_priority_payload() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Stop Priority Strategy"},
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
                        "name": "always_true_signal",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "gt",
                                "right": -1,
                            }
                        },
                    },
                    {
                        "type": "stop_loss",
                        "name": "stop_points",
                        "stop": {"kind": "points", "value": 0.3},
                    },
                ],
            }
        },
    }


def _macd_payload() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "MACD Output Ref Strategy"},
        "universe": {"market": "crypto", "tickers": ["BTCUSDT"]},
        "timeframe": "1h",
        "factors": {
            "macd_12_26_9": {
                "type": "macd",
                "params": {"fast": 12, "slow": 26, "signal": 9, "source": "close"},
                "outputs": ["macd_line", "signal", "histogram"],
            }
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "macd_12_26_9.macd_line"},
                            "op": "gt",
                            "right": {"ref": "macd_12_26_9.signal"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "macd_flip_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "macd_12_26_9.macd_line"},
                                "op": "lt",
                                "right": {"ref": "macd_12_26_9.signal"},
                            }
                        },
                    }
                ],
            }
        },
    }


def _zero_price_dual_side_payload() -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Dual Side Quantity Guard Strategy"},
        "universe": {"market": "crypto", "tickers": ["BTCUSDT"]},
        "timeframe": "1h",
        "factors": {
            "ema_2": {"type": "ema", "params": {"period": 2, "source": "close"}},
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_cash", "cash": 10},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "eq",
                            "right": 0,
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "never_exit_long",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "lt",
                                "right": -1,
                            }
                        },
                    }
                ],
            },
            "short": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "eq",
                            "right": 0,
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "never_exit_short",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "lt",
                                "right": -1,
                            }
                        },
                    }
                ],
            },
        },
    }


def _single_side_payload(
    *,
    exits: list[dict[str, Any]],
    entry_condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Single Side Engine Test"},
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
                    "condition": entry_condition
                    or {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "gt",
                            "right": 0,
                        }
                    },
                },
                "exits": exits,
            }
        },
    }


def test_event_driven_engine_opens_and_closes_on_signal() -> None:
    payload = _signal_strategy_payload()
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([10, 10, 10, 10, 10, 10, 10, 12, 14, 13, 11, 9, 8])

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "exit_signal_down"
    assert any(event.type == BacktestEventType.ENTRY_SIGNAL for event in result.events)
    assert any(event.type == BacktestEventType.POSITION_OPENED for event in result.events)
    assert any(event.type == BacktestEventType.POSITION_CLOSED for event in result.events)


def test_stop_loss_has_priority_over_signal_exit() -> None:
    payload = _stop_priority_payload()
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([10, 10, 10, 10, 10, 10, 10])
    data.loc[data.index[5], "low"] = 9.0

    result = EventDrivenBacktestEngine(
        strategy=strategy,
        data=data,
        config=BacktestConfig(initial_capital=1000.0),
    ).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "stop_loss"


def test_multi_output_factor_refs_macd_are_resolved() -> None:
    payload = _macd_payload()
    strategy = parse_strategy_payload(payload)
    closes = [100 + i * 0.2 for i in range(60)] + [112 - i * 0.25 for i in range(60)]
    data = _ohlcv_from_close(closes)

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.started_at.tzinfo is not None
    assert result.finished_at.tzinfo is not None
    assert result.started_at.tzinfo == UTC
    assert result.finished_at.tzinfo == UTC
    assert len(result.events) > 0


def test_engine_skips_entries_when_price_is_non_positive() -> None:
    payload = _zero_price_dual_side_payload()
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([0.0] * 24)

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades == 0


def test_take_profit_triggered_when_target_is_hit() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "stop_loss",
                "name": "sl_points",
                "stop": {"kind": "points", "value": 1.0},
            },
            {
                "type": "take_profit",
                "name": "tp_points",
                "take": {"kind": "points", "value": 1.0},
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 16)
    data.loc[data.index[5], "high"] = 101.2
    data.loc[data.index[5], "low"] = 99.8

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == pytest.approx(101.0)


def test_stop_is_prioritized_when_stop_and_take_hit_in_same_bar() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "stop_loss",
                "name": "sl_points",
                "stop": {"kind": "points", "value": 1.0},
            },
            {
                "type": "take_profit",
                "name": "tp_points",
                "take": {"kind": "points", "value": 1.0},
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 16)
    data.loc[data.index[5], "high"] = 101.3
    data.loc[data.index[5], "low"] = 98.7

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(99.0)


def test_bracket_rr_derives_take_from_stop() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "bracket_rr",
                "name": "rr_take_from_stop",
                "stop": {"kind": "points", "value": 1.0},
                "risk_reward": 2.0,
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 16)
    data.loc[data.index[5], "high"] = 102.2
    data.loc[data.index[5], "low"] = 99.7

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == pytest.approx(102.0)


def test_bracket_rr_derives_stop_from_take() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "bracket_rr",
                "name": "rr_stop_from_take",
                "take": {"kind": "points", "value": 2.0},
                "risk_reward": 2.0,
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 16)
    data.loc[data.index[5], "high"] = 100.3
    data.loc[data.index[5], "low"] = 98.8

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(99.0)


def test_end_of_data_auto_closes_open_position() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "signal_exit",
                "name": "never",
                "order": {"type": "market"},
                "condition": {
                    "cmp": {
                        "left": {"ref": "price.close"},
                        "op": "lt",
                        "right": -1,
                    }
                },
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 24)

    result = EventDrivenBacktestEngine(strategy=strategy, data=data).run()

    assert result.summary.total_trades == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].exit_time == data.index[-1].to_pydatetime()


def test_slippage_and_commission_are_applied_to_trade_pnl() -> None:
    payload = _single_side_payload(
        exits=[
            {
                "type": "signal_exit",
                "name": "always_exit_next_bar",
                "order": {"type": "market"},
                "condition": {
                    "cmp": {
                        "left": {"ref": "price.close"},
                        "op": "gt",
                        "right": -1,
                    }
                },
            },
        ],
    )
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 12)

    result = EventDrivenBacktestEngine(
        strategy=strategy,
        data=data,
        config=BacktestConfig(
            initial_capital=10_000.0,
            commission_rate=0.001,
            slippage_bps=10.0,
        ),
    ).run()

    assert result.summary.total_trades >= 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.1)
    assert trade.exit_price == pytest.approx(99.9)
    assert trade.commission == pytest.approx(0.2)
    assert trade.pnl == pytest.approx(-0.4)


def test_short_position_is_liquidated_and_drawdown_capped_at_100_pct() -> None:
    payload = {
        "dsl_version": "1.0.0",
        "strategy": {"name": "Liquidation Guard Strategy"},
        "universe": {"market": "crypto", "tickers": ["BTCUSDT"]},
        "timeframe": "1h",
        "factors": {
            "ema_2": {"type": "ema", "params": {"period": 2, "source": "close"}},
        },
        "trade": {
            "short": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1000},
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
                        "name": "never",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": {"ref": "price.close"},
                                "op": "lt",
                                "right": -1,
                            }
                        },
                    }
                ],
            }
        },
    }
    strategy = parse_strategy_payload(payload)
    data = _ohlcv_from_close([100.0] * 10 + [300.0] * 10)

    result = EventDrivenBacktestEngine(
        strategy=strategy,
        data=data,
        config=BacktestConfig(initial_capital=100_000.0),
    ).run()

    assert result.summary.total_trades >= 1
    assert result.trades[0].exit_reason == "liquidation"
    assert result.summary.final_equity == pytest.approx(0.0)
    assert result.summary.total_return_pct == pytest.approx(-100.0)
    assert result.summary.max_drawdown_pct <= 100.0
