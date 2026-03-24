"""Shared exit-rule evaluation helpers for engine and trace builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from packages.domain.backtest.types import PositionSide


@dataclass(frozen=True, slots=True)
class ResolvedStopTake:
    stop_price: float | None
    take_price: float | None
    stop_rule_id: str | None
    take_rule_id: str | None


def resolve_position_stops(
    *,
    side_payload: dict[str, Any],
    side: PositionSide,
    frame: pd.DataFrame,
    entry_price: float,
    bar_index: int,
) -> tuple[float | None, float | None]:
    """Return concrete stop/take prices resolved from DSL exit rules."""

    resolved = resolve_position_stop_take_with_rule_ids(
        side_payload=side_payload,
        side=side,
        frame=frame,
        entry_price=entry_price,
        bar_index=bar_index,
    )
    return (resolved.stop_price, resolved.take_price)


def resolve_position_stop_take_with_rule_ids(
    *,
    side_payload: dict[str, Any],
    side: PositionSide,
    frame: pd.DataFrame,
    entry_price: float,
    bar_index: int,
) -> ResolvedStopTake:
    exits = side_payload.get("exits", [])
    if not isinstance(exits, list):
        return ResolvedStopTake(
            stop_price=None,
            take_price=None,
            stop_rule_id=None,
            take_rule_id=None,
        )

    stop_distance: float | None = None
    take_distance: float | None = None
    stop_rule_id: str | None = None
    take_rule_id: str | None = None

    for index, exit_rule in enumerate(exits):
        if not isinstance(exit_rule, dict):
            continue
        rule_type = str(exit_rule.get("type", "")).strip().lower()
        rule_id = _resolve_rule_id(exit_rule, index=index, fallback_prefix=rule_type)
        if rule_type == "stop_loss" and stop_distance is None:
            stop_distance = distance_from_stop_spec(
                exit_rule.get("stop"),
                frame=frame,
                entry_price=entry_price,
                bar_index=bar_index,
            )
            if stop_distance is not None and stop_rule_id is None:
                stop_rule_id = rule_id
        elif rule_type == "take_profit" and take_distance is None:
            take_distance = distance_from_stop_spec(
                exit_rule.get("take"),
                frame=frame,
                entry_price=entry_price,
                bar_index=bar_index,
            )
            if take_distance is not None and take_rule_id is None:
                take_rule_id = rule_id
        elif rule_type == "bracket_rr":
            rr_raw = exit_rule.get("risk_reward")
            rr = float(rr_raw) if isinstance(rr_raw, int | float) else 0.0
            bracket_stop = distance_from_stop_spec(
                exit_rule.get("stop"),
                frame=frame,
                entry_price=entry_price,
                bar_index=bar_index,
            )
            bracket_take = distance_from_stop_spec(
                exit_rule.get("take"),
                frame=frame,
                entry_price=entry_price,
                bar_index=bar_index,
            )

            if stop_distance is None and bracket_stop is not None:
                stop_distance = bracket_stop
                if stop_rule_id is None:
                    stop_rule_id = rule_id
            if take_distance is None and bracket_take is not None:
                take_distance = bracket_take
                if take_rule_id is None:
                    take_rule_id = rule_id

            if rr > 0:
                if stop_distance is not None and take_distance is None:
                    take_distance = stop_distance * rr
                    if take_rule_id is None:
                        take_rule_id = rule_id
                if take_distance is not None and stop_distance is None:
                    stop_distance = take_distance / rr
                    if stop_rule_id is None:
                        stop_rule_id = rule_id

    stop_price = None
    take_price = None
    if stop_distance is not None:
        stop_price = (
            entry_price - stop_distance
            if side == PositionSide.LONG
            else entry_price + stop_distance
        )
    if take_distance is not None:
        take_price = (
            entry_price + take_distance
            if side == PositionSide.LONG
            else entry_price - take_distance
        )

    return ResolvedStopTake(
        stop_price=stop_price,
        take_price=take_price,
        stop_rule_id=stop_rule_id,
        take_rule_id=take_rule_id,
    )


def distance_from_stop_spec(
    stop_spec: Any,
    *,
    frame: pd.DataFrame,
    entry_price: float,
    bar_index: int,
) -> float | None:
    """Resolve stop/take distance from one DSL stop spec."""

    if not isinstance(stop_spec, dict):
        return None

    kind = str(stop_spec.get("kind", "")).strip().lower()
    if kind == "points":
        raw = stop_spec.get("value")
        if isinstance(raw, int | float) and raw > 0:
            return float(raw)
        return None
    if kind == "pct":
        raw = stop_spec.get("value")
        if isinstance(raw, int | float) and raw > 0:
            return float(entry_price) * float(raw)
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
        if not isinstance(atr_value, int | float):
            return None
        return float(atr_value) * float(multiple)
    return None


def _resolve_rule_id(
    exit_rule: dict[str, Any], *, index: int, fallback_prefix: str
) -> str:
    name = exit_rule.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    prefix = fallback_prefix if fallback_prefix else "exit"
    return f"{prefix}_{index + 1}"
