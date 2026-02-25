"""Live DSL signal runtime evaluated on bar-close."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from src.engine.backtest.condition import evaluate_condition_at
from src.engine.backtest.factors import prepare_backtest_frame
from src.engine.market_data.runtime import RuntimeBar
from src.engine.strategy.pipeline import parse_strategy_payload

SignalType = Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE", "NOOP"]


@dataclass(frozen=True, slots=True)
class SignalDecision:
    signal: SignalType
    reason: str
    bar_time: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_condition(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    condition = node.get("condition")
    if isinstance(condition, dict):
        return condition
    return None


class LiveSignalRuntime:
    """Evaluate strategy DSL against latest bar-close frame."""

    def evaluate(
        self,
        *,
        strategy_payload: dict[str, Any],
        bars: list[RuntimeBar],
        current_position_side: str = "flat",
    ) -> SignalDecision:
        if len(bars) < 2:
            return SignalDecision(
                signal="NOOP",
                reason="insufficient_bars",
                bar_time=bars[-1].timestamp if bars else None,
            )

        parsed = parse_strategy_payload(strategy_payload)
        frame = pd.DataFrame(
            {
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
            },
            index=pd.DatetimeIndex([bar.timestamp for bar in bars]),
        )
        enriched = prepare_backtest_frame(frame, strategy=parsed)
        bar_index = len(enriched) - 1

        trade = parsed.trade if isinstance(parsed.trade, dict) else {}
        long_side = trade.get("long", {}) if isinstance(trade.get("long"), dict) else {}
        short_side = trade.get("short", {}) if isinstance(trade.get("short"), dict) else {}

        long_entry_condition = _extract_condition(long_side.get("entry"))
        short_entry_condition = _extract_condition(short_side.get("entry"))

        long_entry = bool(
            long_entry_condition
            and evaluate_condition_at(
                long_entry_condition,
                frame=enriched,
                bar_index=bar_index,
            )
        )
        short_entry = bool(
            short_entry_condition
            and evaluate_condition_at(
                short_entry_condition,
                frame=enriched,
                bar_index=bar_index,
            )
        )

        long_exit = self._evaluate_signal_exit(
            exits=long_side.get("exits"),
            frame=enriched,
            bar_index=bar_index,
        )
        short_exit = self._evaluate_signal_exit(
            exits=short_side.get("exits"),
            frame=enriched,
            bar_index=bar_index,
        )

        bar_time = bars[-1].timestamp
        if current_position_side == "flat":
            if long_entry:
                return SignalDecision(
                    signal="OPEN_LONG",
                    reason="long_entry_condition",
                    bar_time=bar_time,
                )
            if short_entry:
                return SignalDecision(
                    signal="OPEN_SHORT",
                    reason="short_entry_condition",
                    bar_time=bar_time,
                )
            return SignalDecision(signal="NOOP", reason="no_entry", bar_time=bar_time)

        if current_position_side == "long":
            if long_exit or short_entry:
                return SignalDecision(signal="CLOSE", reason="long_exit", bar_time=bar_time)
            return SignalDecision(signal="NOOP", reason="hold_long", bar_time=bar_time)

        if current_position_side == "short":
            if short_exit or long_entry:
                return SignalDecision(signal="CLOSE", reason="short_exit", bar_time=bar_time)
            return SignalDecision(signal="NOOP", reason="hold_short", bar_time=bar_time)

        return SignalDecision(signal="NOOP", reason="unknown_position_side", bar_time=bar_time)

    def _evaluate_signal_exit(
        self,
        *,
        exits: Any,
        frame: pd.DataFrame,
        bar_index: int,
    ) -> bool:
        if not isinstance(exits, list):
            return False
        for rule in exits:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("type", "")).strip().lower() != "signal_exit":
                continue
            condition = rule.get("condition")
            if not isinstance(condition, dict):
                continue
            if evaluate_condition_at(condition, frame=frame, bar_index=bar_index):
                return True
        return False
