from __future__ import annotations

from datetime import UTC, datetime, timedelta

from packages.domain.market_data.runtime import RuntimeBar
from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime


def _bars(count: int) -> list[RuntimeBar]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    output: list[RuntimeBar] = []
    for index in range(count):
        price = 100.0 + float(index)
        output.append(
            RuntimeBar(
                timestamp=start + timedelta(minutes=index),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0 + float(index),
            )
        )
    return output


def _bar(
    *,
    minute: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> RuntimeBar:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return RuntimeBar(
        timestamp=start + timedelta(minutes=minute),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1000.0 + float(minute),
    )


def _strategy_payload(
    *,
    side: str,
    exits: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "Runtime Test",
            "description": "Live signal runtime test",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTC/USD"],
        },
        "timeframe": "1m",
        "factors": {
            "ema_2": {
                "type": "ema",
                "params": {"period": 2, "source": "close"},
            }
        },
        "trade": {
            side: {
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "ema_2"},
                            "op": "lt",
                            "right": 0,
                        }
                    },
                },
                "exits": exits,
            }
        },
    }


def test_live_signal_runtime_infers_factor_warmup_requirement() -> None:
    runtime = LiveSignalRuntime()
    strategy_payload = {
        "timeframe": "1m",
        "factors": {
            "ema_fast": {
                "factor_type": "ema",
                "params": {"period": 20},
            },
            "macd_signal": {
                "factor_type": "macd",
                "params": {"fast": 12, "slow": 26, "signal": 9},
            },
        },
        "trade": {},
    }

    assert runtime.required_bars(strategy_payload=strategy_payload) == 28


def test_live_signal_runtime_returns_insufficient_bars_until_warm() -> None:
    runtime = LiveSignalRuntime()
    strategy_payload = {
        "timeframe": "1m",
        "factors": {
            "ema_fast": {
                "factor_type": "ema",
                "params": {"period": 14},
            },
        },
        "trade": {},
    }

    insufficient = runtime.evaluate(
        strategy_payload=strategy_payload,
        bars=_bars(10),
        current_position_side="flat",
    )
    assert insufficient.signal == "NOOP"
    assert insufficient.reason == "insufficient_bars"
    assert insufficient.metadata["required_bars"] == 16
    assert insufficient.metadata["available_bars"] == 10

    assert runtime.required_bars(strategy_payload=strategy_payload) == 16


def test_live_signal_runtime_triggers_stop_loss_for_long_position() -> None:
    runtime = LiveSignalRuntime()
    strategy_payload = _strategy_payload(
        side="long",
        exits=[
            {
                "type": "stop_loss",
                "name": "protective_stop",
                "stop": {"kind": "pct", "value": 0.02},
            }
        ],
    )

    decision = runtime.evaluate(
        strategy_payload=strategy_payload,
        bars=[
            _bar(minute=0, open_price=100.0, high=101.0, low=99.0, close=100.0),
            _bar(minute=1, open_price=100.0, high=100.5, low=99.5, close=100.0),
            _bar(minute=2, open_price=100.0, high=100.5, low=99.5, close=100.0),
            _bar(minute=3, open_price=100.0, high=100.5, low=97.5, close=99.0),
        ],
        current_position_side="long",
        current_position_entry_price=100.0,
    )

    assert decision.signal == "CLOSE"
    assert decision.reason == "stop_loss"
    assert decision.metadata["exit_price"] == 98.0
    assert decision.metadata["managed_stop_price"] == 98.0


def test_live_signal_runtime_triggers_bracket_take_profit_for_short_position() -> None:
    runtime = LiveSignalRuntime()
    strategy_payload = _strategy_payload(
        side="short",
        exits=[
            {
                "type": "bracket_rr",
                "name": "short_bracket",
                "stop": {"kind": "points", "value": 5.0},
                "risk_reward": 2.0,
            }
        ],
    )

    decision = runtime.evaluate(
        strategy_payload=strategy_payload,
        bars=[
            _bar(minute=0, open_price=100.0, high=101.0, low=99.0, close=100.0),
            _bar(minute=1, open_price=100.0, high=100.5, low=99.5, close=100.0),
            _bar(minute=2, open_price=100.0, high=100.5, low=99.5, close=100.0),
            _bar(minute=3, open_price=100.0, high=101.0, low=89.0, close=91.0),
        ],
        current_position_side="short",
        current_position_entry_price=100.0,
    )

    assert decision.signal == "CLOSE"
    assert decision.reason == "take_profit"
    assert decision.metadata["exit_price"] == 90.0
    assert decision.metadata["managed_take_price"] == 90.0


def test_live_signal_runtime_builds_managed_exit_targets_from_bracket_rule() -> None:
    runtime = LiveSignalRuntime()
    strategy_payload = _strategy_payload(
        side="long",
        exits=[
            {
                "type": "bracket_rr",
                "name": "long_bracket",
                "stop": {"kind": "pct", "value": 0.02},
                "risk_reward": 1.5,
            }
        ],
    )

    targets = runtime.build_managed_exit_targets(
        strategy_payload=strategy_payload,
        bars=_bars(4),
        signal="OPEN_LONG",
        entry_price=100.0,
    )

    assert targets["side"] == "long"
    assert targets["entry_price"] == 100.0
    assert targets["stop_price"] == 98.0
    assert targets["take_price"] == 103.0
