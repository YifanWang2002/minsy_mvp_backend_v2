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
