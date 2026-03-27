"""Live DSL signal runtime evaluated on bar-close."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from packages.domain.backtest.condition import evaluate_condition_at
from packages.domain.backtest.factors import prepare_backtest_frame
from packages.domain.market_data.runtime import RuntimeBar
from packages.domain.strategy.pipeline import parse_strategy_payload

SignalType = Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE", "NOOP"]
_LOOKBACK_KEYS = {
    "period",
    "length",
    "fast",
    "slow",
    "signal",
    "k_period",
    "k_smooth",
    "d_period",
}


@dataclass(frozen=True, slots=True)
class SignalDecision:
    signal: SignalType
    reason: str
    bar_time: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ResolvedExit:
    reason: str
    exit_price: float
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

    def required_bars(self, *, strategy_payload: dict[str, Any]) -> int:
        max_lookback = 1
        factors = strategy_payload.get("factors")
        if isinstance(factors, dict):
            for raw_factor in factors.values():
                if not isinstance(raw_factor, dict):
                    continue
                params = raw_factor.get("params")
                if not isinstance(params, dict):
                    continue
                for key, value in params.items():
                    if key not in _LOOKBACK_KEYS:
                        continue
                    if isinstance(value, bool) or not isinstance(value, int | float):
                        continue
                    max_lookback = max(max_lookback, int(value))
        return max(2, max_lookback + 2)

    def evaluate(
        self,
        *,
        strategy_payload: dict[str, Any],
        bars: list[RuntimeBar],
        current_position_side: str = "flat",
        current_position_entry_price: float | None = None,
        current_position_stop_price: float | None = None,
        current_position_take_price: float | None = None,
    ) -> SignalDecision:
        required_bars = self.required_bars(strategy_payload=strategy_payload)
        if len(bars) < required_bars:
            return SignalDecision(
                signal="NOOP",
                reason="insufficient_bars",
                bar_time=bars[-1].timestamp if bars else None,
                metadata={
                    "required_bars": required_bars,
                    "available_bars": len(bars),
                },
            )

        parsed, enriched, bar_index = self._build_enriched_frame(
            strategy_payload=strategy_payload,
            bars=bars,
        )
        trade = parsed.trade if isinstance(parsed.trade, dict) else {}
        long_side = self._side_payload(trade=trade, side_name="long")
        short_side = self._side_payload(trade=trade, side_name="short")

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

        long_exit = self._evaluate_exit(
            side_name="long",
            side_payload=long_side,
            frame=enriched,
            bar_index=bar_index,
            current_position_entry_price=current_position_entry_price,
            current_position_stop_price=current_position_stop_price,
            current_position_take_price=current_position_take_price,
        )
        short_exit = self._evaluate_exit(
            side_name="short",
            side_payload=short_side,
            frame=enriched,
            bar_index=bar_index,
            current_position_entry_price=current_position_entry_price,
            current_position_stop_price=current_position_stop_price,
            current_position_take_price=current_position_take_price,
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
            if long_exit is not None:
                return SignalDecision(
                    signal="CLOSE",
                    reason=long_exit.reason,
                    bar_time=bar_time,
                    metadata=long_exit.metadata,
                )
            if short_entry:
                return SignalDecision(
                    signal="CLOSE",
                    reason="short_entry_reversal",
                    bar_time=bar_time,
                    metadata={
                        **self._exit_metadata(
                            stop_price=current_position_stop_price,
                            take_price=current_position_take_price,
                            side_name="long",
                            entry_price=current_position_entry_price,
                        ),
                        "exit_price": float(enriched.iloc[bar_index]["close"]),
                    },
                )
            return SignalDecision(signal="NOOP", reason="hold_long", bar_time=bar_time)

        if current_position_side == "short":
            if short_exit is not None:
                return SignalDecision(
                    signal="CLOSE",
                    reason=short_exit.reason,
                    bar_time=bar_time,
                    metadata=short_exit.metadata,
                )
            if long_entry:
                return SignalDecision(
                    signal="CLOSE",
                    reason="long_entry_reversal",
                    bar_time=bar_time,
                    metadata={
                        **self._exit_metadata(
                            stop_price=current_position_stop_price,
                            take_price=current_position_take_price,
                            side_name="short",
                            entry_price=current_position_entry_price,
                        ),
                        "exit_price": float(enriched.iloc[bar_index]["close"]),
                    },
                )
            return SignalDecision(signal="NOOP", reason="hold_short", bar_time=bar_time)

        return SignalDecision(signal="NOOP", reason="unknown_position_side", bar_time=bar_time)

    def build_managed_exit_targets(
        self,
        *,
        strategy_payload: dict[str, Any],
        bars: list[RuntimeBar],
        signal: SignalType,
        entry_price: float,
    ) -> dict[str, Any]:
        if signal not in {"OPEN_LONG", "OPEN_SHORT"}:
            return {}
        if entry_price <= 0 or not bars:
            return {}

        parsed, enriched, bar_index = self._build_enriched_frame(
            strategy_payload=strategy_payload,
            bars=bars,
        )
        trade = parsed.trade if isinstance(parsed.trade, dict) else {}
        side_name = "long" if signal == "OPEN_LONG" else "short"
        side_payload = self._side_payload(trade=trade, side_name=side_name)
        stop_price, take_price = self._resolve_price_exits(
            exits=side_payload.get("exits"),
            side_name=side_name,
            entry_price=entry_price,
            frame=enriched,
            bar_index=bar_index,
        )
        if stop_price is None and take_price is None:
            return {}
        targets: dict[str, Any] = {
            "side": side_name,
            "entry_price": entry_price,
        }
        if stop_price is not None:
            targets["stop_price"] = stop_price
        if take_price is not None:
            targets["take_price"] = take_price
        return targets

    def _build_enriched_frame(
        self,
        *,
        strategy_payload: dict[str, Any],
        bars: list[RuntimeBar],
    ) -> tuple[Any, pd.DataFrame, int]:
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
        return parsed, enriched, len(enriched) - 1

    def _side_payload(
        self,
        *,
        trade: dict[str, Any],
        side_name: str,
    ) -> dict[str, Any]:
        raw = trade.get(side_name)
        return raw if isinstance(raw, dict) else {}

    def _evaluate_exit(
        self,
        *,
        side_name: str,
        side_payload: dict[str, Any],
        frame: pd.DataFrame,
        bar_index: int,
        current_position_entry_price: float | None,
        current_position_stop_price: float | None,
        current_position_take_price: float | None,
    ) -> _ResolvedExit | None:
        exits = side_payload.get("exits")
        if not isinstance(exits, list):
            return None

        stop_price = self._coerce_positive_float(current_position_stop_price)
        take_price = self._coerce_positive_float(current_position_take_price)
        entry_price = self._coerce_positive_float(current_position_entry_price)
        if entry_price is not None and (stop_price is None or take_price is None):
            resolved_stop, resolved_take = self._resolve_price_exits(
                exits=exits,
                side_name=side_name,
                entry_price=entry_price,
                frame=frame,
                bar_index=bar_index,
            )
            if stop_price is None:
                stop_price = resolved_stop
            if take_price is None:
                take_price = resolved_take

        row = frame.iloc[bar_index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        metadata = self._exit_metadata(
            stop_price=stop_price,
            take_price=take_price,
            side_name=side_name,
            entry_price=entry_price,
        )

        if stop_price is not None:
            stop_hit = (
                low <= stop_price
                if side_name == "long"
                else high >= stop_price
            )
            if stop_hit:
                return _ResolvedExit(
                    reason="stop_loss",
                    exit_price=stop_price,
                    metadata={
                        **metadata,
                        "exit_price": stop_price,
                    },
                )

        if take_price is not None:
            take_hit = (
                high >= take_price
                if side_name == "long"
                else low <= take_price
            )
            if take_hit:
                return _ResolvedExit(
                    reason="take_profit",
                    exit_price=take_price,
                    metadata={
                        **metadata,
                        "exit_price": take_price,
                    },
                )

        signal_exit_reason = self._evaluate_signal_exit(
            exits=exits,
            frame=frame,
            bar_index=bar_index,
        )
        if signal_exit_reason is None:
            return None
        return _ResolvedExit(
            reason=signal_exit_reason,
            exit_price=close,
            metadata={
                **metadata,
                "exit_price": close,
            },
        )

    def _evaluate_signal_exit(
        self,
        *,
        exits: Any,
        frame: pd.DataFrame,
        bar_index: int,
    ) -> str | None:
        if not isinstance(exits, list):
            return None
        for rule in exits:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("type", "")).strip().lower() != "signal_exit":
                continue
            condition = rule.get("condition")
            if not isinstance(condition, dict):
                continue
            if evaluate_condition_at(condition, frame=frame, bar_index=bar_index):
                name = str(rule.get("name", "signal_exit")).strip()
                return name or "signal_exit"
        return None

    def _resolve_price_exits(
        self,
        *,
        exits: Any,
        side_name: str,
        entry_price: float,
        frame: pd.DataFrame,
        bar_index: int,
    ) -> tuple[float | None, float | None]:
        if not isinstance(exits, list):
            return None, None

        stop_distance: float | None = None
        take_distance: float | None = None

        for rule in exits:
            if not isinstance(rule, dict):
                continue
            rule_type = str(rule.get("type", "")).strip().lower()
            if rule_type == "stop_loss" and stop_distance is None:
                stop_distance = self._distance_from_stop_spec(
                    rule.get("stop"),
                    entry_price=entry_price,
                    frame=frame,
                    bar_index=bar_index,
                )
            elif rule_type == "take_profit" and take_distance is None:
                take_distance = self._distance_from_stop_spec(
                    rule.get("take"),
                    entry_price=entry_price,
                    frame=frame,
                    bar_index=bar_index,
                )
            elif rule_type == "bracket_rr":
                rr_raw = rule.get("risk_reward")
                rr = float(rr_raw) if isinstance(rr_raw, int | float) else 0.0
                bracket_stop = self._distance_from_stop_spec(
                    rule.get("stop"),
                    entry_price=entry_price,
                    frame=frame,
                    bar_index=bar_index,
                )
                bracket_take = self._distance_from_stop_spec(
                    rule.get("take"),
                    entry_price=entry_price,
                    frame=frame,
                    bar_index=bar_index,
                )
                if stop_distance is None and bracket_stop is not None:
                    stop_distance = bracket_stop
                if take_distance is None and bracket_take is not None:
                    take_distance = bracket_take
                if rr > 0:
                    if stop_distance is not None and take_distance is None:
                        take_distance = stop_distance * rr
                    if take_distance is not None and stop_distance is None:
                        stop_distance = take_distance / rr

        stop_price = None
        if stop_distance is not None:
            stop_price = (
                entry_price - stop_distance
                if side_name == "long"
                else entry_price + stop_distance
            )

        take_price = None
        if take_distance is not None:
            take_price = (
                entry_price + take_distance
                if side_name == "long"
                else entry_price - take_distance
            )

        return stop_price, take_price

    def _distance_from_stop_spec(
        self,
        stop_spec: Any,
        *,
        entry_price: float,
        frame: pd.DataFrame,
        bar_index: int,
    ) -> float | None:
        if not isinstance(stop_spec, dict):
            return None

        kind = str(stop_spec.get("kind", "")).strip().lower()
        if kind == "points":
            value = stop_spec.get("value")
            if isinstance(value, int | float) and value > 0:
                return float(value)
            return None
        if kind == "pct":
            value = stop_spec.get("value")
            if isinstance(value, int | float) and value > 0:
                return float(entry_price) * float(value)
            return None
        if kind == "atr_multiple":
            atr_ref = stop_spec.get("atr_ref")
            multiple = stop_spec.get("multiple")
            if not isinstance(atr_ref, str):
                return None
            if not isinstance(multiple, int | float) or multiple <= 0:
                return None
            if atr_ref not in frame.columns:
                return None
            atr_value = frame.iloc[bar_index][atr_ref]
            if not isinstance(atr_value, int | float) or pd.isna(atr_value):
                return None
            return float(atr_value) * float(multiple)
        return None

    def _coerce_positive_float(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        number = float(value)
        return number if number > 0 else None

    def _exit_metadata(
        self,
        *,
        stop_price: float | None,
        take_price: float | None,
        side_name: str,
        entry_price: float | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "position_side": str(side_name).strip().lower(),
        }
        if entry_price is not None:
            metadata["current_position_entry_price"] = float(entry_price)
        if stop_price is not None:
            metadata["managed_stop_price"] = stop_price
        if take_price is not None:
            metadata["managed_take_price"] = take_price
        return metadata
