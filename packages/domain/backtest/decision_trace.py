"""Trade-level decision trace builders for snapshot explainability."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from packages.domain.backtest.condition import evaluate_condition_trace_at
from packages.domain.backtest.exit_rules import (
    ResolvedStopTake,
    resolve_position_stop_take_with_rule_ids,
)
from packages.domain.backtest.types import PositionSide
from packages.domain.strategy.models import ParsedStrategyDsl

_DECISION_TRACE_VERSION = "1.0"


def build_trade_decision_trace(
    *,
    frame: pd.DataFrame,
    strategy: ParsedStrategyDsl,
    trade: dict[str, Any],
    entry_index: int,
    exit_index: int,
    start_index: int,
) -> dict[str, Any]:
    """Build entry/exit decision trace for one trade inside the prepared frame."""

    side = _resolve_side(trade.get("side"))
    side_payload = _resolve_side_payload(strategy, side)

    entry_time = _to_iso_utc(frame.index[entry_index])
    exit_time = _to_iso_utc(frame.index[exit_index])

    entry_trace = _build_entry_trace(
        frame=frame,
        side_payload=side_payload,
        entry_index=entry_index,
        start_index=start_index,
        entry_time=entry_time,
    )
    exit_trace = _build_exit_trace(
        frame=frame,
        side_payload=side_payload,
        side=side,
        trade=trade,
        entry_index=entry_index,
        exit_index=exit_index,
        start_index=start_index,
        exit_time=exit_time,
    )

    return {
        "version": _DECISION_TRACE_VERSION,
        "entry": entry_trace,
        "exit": exit_trace,
    }


def _build_entry_trace(
    *,
    frame: pd.DataFrame,
    side_payload: dict[str, Any] | None,
    entry_index: int,
    start_index: int,
    entry_time: str,
) -> dict[str, Any]:
    entry_condition: dict[str, Any] | None = None
    if isinstance(side_payload, dict):
        entry = side_payload.get("entry")
        if isinstance(entry, dict) and isinstance(entry.get("condition"), dict):
            entry_condition = dict(entry["condition"])

    if entry_condition is None:
        return {
            "bar_offset": int(entry_index - start_index),
            "time": entry_time,
            "overall_hit": False,
            "tree": {
                "node_type": "entry",
                "path": "$",
                "hit": False,
                "reason_code": "missing_entry_condition",
            },
            "unmet_nodes": [
                {
                    "path": "$",
                    "node_type": "entry",
                    "reason_code": "missing_entry_condition",
                    "actual_values": {},
                }
            ],
        }

    traced = evaluate_condition_trace_at(
        entry_condition,
        frame=frame,
        bar_index=entry_index,
    )
    return {
        "bar_offset": int(entry_index - start_index),
        "time": entry_time,
        "overall_hit": bool(traced.get("hit")),
        "tree": traced.get("tree", {}),
        "unmet_nodes": traced.get("unmet_nodes", []),
    }


def _build_exit_trace(
    *,
    frame: pd.DataFrame,
    side_payload: dict[str, Any] | None,
    side: PositionSide,
    trade: dict[str, Any],
    entry_index: int,
    exit_index: int,
    start_index: int,
    exit_time: str,
) -> dict[str, Any]:
    row = frame.iloc[exit_index]
    high = _to_float(row.get("high"))
    low = _to_float(row.get("low"))
    close = _to_float(row.get("close"))

    entry_price = _to_float(trade.get("entry_price"))
    if entry_price is None:
        entry_price = _to_float(frame.iloc[entry_index].get("close")) or 0.0

    resolved = _resolve_stop_take(
        frame=frame,
        side_payload=side_payload,
        side=side,
        entry_price=entry_price,
        entry_index=entry_index,
    )

    trigger_order: list[dict[str, Any]] = []
    order = 1

    stop_hit = _is_stop_hit(
        side=side, stop_price=resolved.stop_price, high=high, low=low
    )
    trigger_order.append(
        {
            "order": order,
            "rule_type": "stop_loss",
            "rule_id": resolved.stop_rule_id or "stop_loss",
            "hit": stop_hit,
            "reason": "stop_price_crossed" if stop_hit else "stop_not_crossed",
            "context": {
                "stop_price": resolved.stop_price,
                "high": high,
                "low": low,
            },
        }
    )

    order += 1
    take_hit = _is_take_hit(
        side=side, take_price=resolved.take_price, high=high, low=low
    )
    trigger_order.append(
        {
            "order": order,
            "rule_type": "take_profit",
            "rule_id": resolved.take_rule_id or "take_profit",
            "hit": take_hit,
            "reason": "take_price_crossed" if take_hit else "take_not_crossed",
            "context": {
                "take_price": resolved.take_price,
                "high": high,
                "low": low,
            },
        }
    )

    exits = side_payload.get("exits", []) if isinstance(side_payload, dict) else []
    if not isinstance(exits, list):
        exits = []

    for index, exit_rule in enumerate(exits):
        if not isinstance(exit_rule, dict):
            continue
        rule_type = str(exit_rule.get("type", "")).strip().lower()
        if rule_type != "signal_exit":
            continue
        order += 1
        rule_id = _resolve_rule_id(
            exit_rule, index=index, fallback_prefix="signal_exit"
        )
        condition = exit_rule.get("condition")
        if isinstance(condition, dict):
            condition_trace = evaluate_condition_trace_at(
                dict(condition),
                frame=frame,
                bar_index=exit_index,
            )
            hit = bool(condition_trace.get("hit"))
            reason = "signal_condition_hit" if hit else "signal_condition_miss"
        else:
            condition_trace = None
            hit = False
            reason = "missing_signal_condition"

        trigger_order.append(
            {
                "order": order,
                "rule_type": "signal_exit",
                "rule_id": rule_id,
                "hit": hit,
                "reason": reason,
                "context": {"close": close},
                "condition_trace": condition_trace,
            }
        )

    actual_exit_reason = str(trade.get("exit_reason", "")).strip().lower()
    triggered = _resolve_triggered_row(
        trigger_order=trigger_order,
        actual_exit_reason=actual_exit_reason,
    )

    return {
        "bar_offset": int(exit_index - start_index),
        "time": exit_time,
        "actual_exit_reason": actual_exit_reason,
        "triggered_rule_id": triggered.get("rule_id")
        if isinstance(triggered, dict)
        else None,
        "triggered_order": triggered.get("order")
        if isinstance(triggered, dict)
        else None,
        "trigger_order": trigger_order,
    }


def _resolve_stop_take(
    *,
    frame: pd.DataFrame,
    side_payload: dict[str, Any] | None,
    side: PositionSide,
    entry_price: float,
    entry_index: int,
) -> ResolvedStopTake:
    if not isinstance(side_payload, dict):
        return ResolvedStopTake(
            stop_price=None,
            take_price=None,
            stop_rule_id=None,
            take_rule_id=None,
        )
    return resolve_position_stop_take_with_rule_ids(
        side_payload=side_payload,
        side=side,
        frame=frame,
        entry_price=entry_price,
        bar_index=entry_index,
    )


def _resolve_triggered_row(
    *,
    trigger_order: list[dict[str, Any]],
    actual_exit_reason: str,
) -> dict[str, Any] | None:
    if not trigger_order:
        return None

    expected_type = _expected_rule_type(actual_exit_reason)
    if expected_type is not None:
        for row in trigger_order:
            if row.get("rule_type") == expected_type and row.get("hit") is True:
                return row

    for row in trigger_order:
        if row.get("hit") is True:
            return row

    return None


def _expected_rule_type(exit_reason: str) -> str | None:
    if exit_reason == "stop_loss":
        return "stop_loss"
    if exit_reason == "take_profit":
        return "take_profit"
    if exit_reason:
        return "signal_exit"
    return None


def _resolve_rule_id(
    exit_rule: dict[str, Any], *, index: int, fallback_prefix: str
) -> str:
    name = exit_rule.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"{fallback_prefix}_{index + 1}"


def _resolve_side(raw_side: Any) -> PositionSide:
    side = str(raw_side or "").strip().lower()
    if side == "short":
        return PositionSide.SHORT
    return PositionSide.LONG


def _resolve_side_payload(
    strategy: ParsedStrategyDsl,
    side: PositionSide,
) -> dict[str, Any] | None:
    trade = strategy.trade
    raw = trade.get(side.value)
    if not isinstance(raw, dict):
        return None
    return raw


def _is_stop_hit(
    *,
    side: PositionSide,
    stop_price: float | None,
    high: float | None,
    low: float | None,
) -> bool:
    if stop_price is None or high is None or low is None:
        return False
    if side == PositionSide.LONG:
        return low <= stop_price
    return high >= stop_price


def _is_take_hit(
    *,
    side: PositionSide,
    take_price: float | None,
    high: float | None,
    low: float | None,
) -> bool:
    if take_price is None or high is None or low is None:
        return False
    if side == PositionSide.LONG:
        return high >= take_price
    return low <= take_price


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _to_iso_utc(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    return str(value)
