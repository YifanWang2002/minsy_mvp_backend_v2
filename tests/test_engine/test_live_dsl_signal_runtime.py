from __future__ import annotations

from datetime import UTC, datetime

from src.engine.execution.signal_runtime import LiveSignalRuntime
from src.engine.market_data.runtime import RuntimeBar
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload


def _strategy_payload() -> dict:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["timeframe"] = "1m"
    payload["universe"] = {"market": "stocks", "tickers": ["AAPL"]}
    payload["trade"]["long"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}
    }
    payload["trade"]["short"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}
    }
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "exit_long",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
        }
    ]
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "exit_short",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
        }
    ]
    return payload


def _bars() -> list[RuntimeBar]:
    return [
        RuntimeBar(
            timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=10,
        ),
        RuntimeBar(
            timestamp=datetime(2026, 1, 5, 10, 1, tzinfo=UTC),
            open=101,
            high=102,
            low=100,
            close=101.5,
            volume=11,
        ),
    ]


def test_live_signal_runtime_generates_open_long() -> None:
    runtime = LiveSignalRuntime()
    decision = runtime.evaluate(
        strategy_payload=_strategy_payload(),
        bars=_bars(),
        current_position_side="flat",
    )
    assert decision.signal == "OPEN_LONG"


def test_live_signal_runtime_generates_close_for_long_position() -> None:
    payload = _strategy_payload()
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "exit_long",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
        }
    ]
    runtime = LiveSignalRuntime()
    decision = runtime.evaluate(
        strategy_payload=payload,
        bars=_bars(),
        current_position_side="long",
    )
    assert decision.signal == "CLOSE"


def test_live_signal_runtime_returns_noop_on_insufficient_bars() -> None:
    runtime = LiveSignalRuntime()
    decision = runtime.evaluate(
        strategy_payload=_strategy_payload(),
        bars=_bars()[:1],
        current_position_side="flat",
    )
    assert decision.signal == "NOOP"
    assert decision.reason == "insufficient_bars"
