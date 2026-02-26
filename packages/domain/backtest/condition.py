"""Condition evaluation for event-driven strategy execution."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real
from typing import Any
import weakref

import numpy as np
import pandas as pd

ConditionEvaluator = Callable[[pd.DataFrame, int], bool]
OperandEvaluator = Callable[[pd.DataFrame, int], float]
_FRAME_COLUMN_ARRAY_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[pd.DataFrame], dict[str, np.ndarray[Any, Any]]],
] = {}


def compile_condition(condition: dict[str, Any]) -> ConditionEvaluator:
    """Compile a condition tree into an evaluator callable.

    The compiled evaluator preserves existing DSL runtime semantics while
    removing repeated dict-structure parsing in bar loops.
    """

    if "all" in condition:
        nodes = condition.get("all", [])
        children = [compile_condition(node) for node in nodes if isinstance(node, dict)]
        return lambda frame, bar_index, children=children: all(
            child(frame, bar_index) for child in children
        )

    if "any" in condition:
        nodes = condition.get("any", [])
        children = [compile_condition(node) for node in nodes if isinstance(node, dict)]
        return lambda frame, bar_index, children=children: any(
            child(frame, bar_index) for child in children
        )

    if "not" in condition:
        child = condition.get("not")
        if not isinstance(child, dict):
            return lambda frame, bar_index: False
        child_eval = compile_condition(child)
        return lambda frame, bar_index, child_eval=child_eval: not child_eval(frame, bar_index)

    if "cmp" in condition and isinstance(condition["cmp"], dict):
        return _compile_cmp(condition["cmp"])

    if "cross" in condition and isinstance(condition["cross"], dict):
        return _compile_cross(condition["cross"])

    if "ref" in condition:
        ref = condition.get("ref")
        if not isinstance(ref, str):
            return lambda frame, bar_index: False
        return lambda frame, bar_index, ref=ref: _is_truthy(
            _resolve_ref_value(ref, frame=frame, bar_index=bar_index)
        )

    if "temporal" in condition:
        return _compile_temporal_placeholder()

    return lambda frame, bar_index: False


def evaluate_compiled_condition_at(
    compiled: ConditionEvaluator,
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> bool:
    return compiled(frame, bar_index)


def evaluate_condition_at(
    condition: dict[str, Any],
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> bool:
    """Evaluate a DSL condition tree at one bar index."""

    if "all" in condition:
        nodes = condition.get("all", [])
        return all(
            evaluate_condition_at(node, frame=frame, bar_index=bar_index)
            for node in nodes
            if isinstance(node, dict)
        )

    if "any" in condition:
        nodes = condition.get("any", [])
        return any(
            evaluate_condition_at(node, frame=frame, bar_index=bar_index)
            for node in nodes
            if isinstance(node, dict)
        )

    if "not" in condition:
        child = condition.get("not")
        if not isinstance(child, dict):
            return False
        return not evaluate_condition_at(child, frame=frame, bar_index=bar_index)

    if "cmp" in condition:
        return _evaluate_cmp(condition["cmp"], frame=frame, bar_index=bar_index)

    if "cross" in condition:
        return _evaluate_cross(condition["cross"], frame=frame, bar_index=bar_index)

    if "ref" in condition:
        value = _resolve_ref_value(str(condition["ref"]), frame=frame, bar_index=bar_index)
        return _is_truthy(value)

    if "temporal" in condition:
        raise ValueError("Temporal conditions are reserved and not supported in v1 runtime.")

    return False


def evaluate_condition_series(
    condition: dict[str, Any],
    *,
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Evaluate a DSL condition tree for all bars at once."""

    size = len(frame)
    if size == 0:
        return np.zeros(0, dtype=bool)

    if "all" in condition:
        nodes = condition.get("all", [])
        children = [node for node in nodes if isinstance(node, dict)]
        result = np.ones(size, dtype=bool)
        for child in children:
            result &= evaluate_condition_series(child, frame=frame)
        return result

    if "any" in condition:
        nodes = condition.get("any", [])
        children = [node for node in nodes if isinstance(node, dict)]
        result = np.zeros(size, dtype=bool)
        for child in children:
            result |= evaluate_condition_series(child, frame=frame)
        return result

    if "not" in condition:
        child = condition.get("not")
        if not isinstance(child, dict):
            return np.zeros(size, dtype=bool)
        return np.logical_not(evaluate_condition_series(child, frame=frame))

    if "cmp" in condition and isinstance(condition["cmp"], dict):
        return _evaluate_cmp_series(condition["cmp"], frame=frame)

    if "cross" in condition and isinstance(condition["cross"], dict):
        return _evaluate_cross_series(condition["cross"], frame=frame)

    if "ref" in condition:
        ref = condition.get("ref")
        if not isinstance(ref, str):
            return np.zeros(size, dtype=bool)
        return _truthy_mask(_resolve_ref_series(ref, frame=frame))

    if "temporal" in condition:
        raise ValueError("Temporal conditions are reserved and not supported in v1 runtime.")

    return np.zeros(size, dtype=bool)


def _compile_temporal_placeholder() -> ConditionEvaluator:
    def _raise_temporal(_: pd.DataFrame, __: int) -> bool:
        raise ValueError("Temporal conditions are reserved and not supported in v1 runtime.")

    return _raise_temporal


def _compile_cmp(cmp_node: dict[str, Any]) -> ConditionEvaluator:
    left_eval = _compile_operand(cmp_node.get("left"))
    right_eval = _compile_operand(cmp_node.get("right"))
    op = str(cmp_node.get("op", "")).lower()

    def _eval(frame: pd.DataFrame, bar_index: int) -> bool:
        left = left_eval(frame, bar_index)
        right = right_eval(frame, bar_index)

        if _is_nan(left) or _is_nan(right):
            return False

        if op == "gt":
            return bool(left > right)
        if op == "gte":
            return bool(left >= right)
        if op == "lt":
            return bool(left < right)
        if op == "lte":
            return bool(left <= right)
        if op == "eq":
            return bool(left == right)
        if op == "neq":
            return bool(left != right)
        return False

    return _eval


def _compile_cross(cross_node: dict[str, Any]) -> ConditionEvaluator:
    a_eval = _compile_operand(cross_node.get("a"))
    b_eval = _compile_operand(cross_node.get("b"))
    op = str(cross_node.get("op", "")).lower()

    def _eval(frame: pd.DataFrame, bar_index: int) -> bool:
        if bar_index <= 0:
            return False

        a_now = a_eval(frame, bar_index)
        b_now = b_eval(frame, bar_index)
        a_prev = a_eval(frame, bar_index - 1)
        b_prev = b_eval(frame, bar_index - 1)

        if any(_is_nan(value) for value in (a_now, b_now, a_prev, b_prev)):
            return False

        if op == "cross_above":
            return bool(a_now > b_now and a_prev <= b_prev)
        if op == "cross_below":
            return bool(a_now < b_now and a_prev >= b_prev)
        return False

    return _eval


def _compile_operand(operand: Any) -> OperandEvaluator:
    if isinstance(operand, int | float) and not isinstance(operand, bool):
        value = float(operand)
        return lambda frame, bar_index, value=value: value

    if isinstance(operand, dict):
        ref = operand.get("ref")
        if not isinstance(ref, str):
            return _nan_operand
        offset = operand.get("offset", 0)
        if not isinstance(offset, int):
            offset = 0
        return lambda frame, bar_index, ref=ref, offset=offset: _resolve_ref_value(
            ref,
            frame=frame,
            bar_index=bar_index + offset,
        )

    return _nan_operand


def _evaluate_cmp(
    cmp_node: dict[str, Any],
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> bool:
    left = _resolve_operand(cmp_node.get("left"), frame=frame, bar_index=bar_index)
    right = _resolve_operand(cmp_node.get("right"), frame=frame, bar_index=bar_index)
    op = str(cmp_node.get("op", "")).lower()

    if _is_nan(left) or _is_nan(right):
        return False

    if op == "gt":
        return bool(left > right)
    if op == "gte":
        return bool(left >= right)
    if op == "lt":
        return bool(left < right)
    if op == "lte":
        return bool(left <= right)
    if op == "eq":
        return bool(left == right)
    if op == "neq":
        return bool(left != right)
    return False


def _evaluate_cross(
    cross_node: dict[str, Any],
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> bool:
    if bar_index <= 0:
        return False

    a_now = _resolve_operand(cross_node.get("a"), frame=frame, bar_index=bar_index)
    b_now = _resolve_operand(cross_node.get("b"), frame=frame, bar_index=bar_index)
    a_prev = _resolve_operand(cross_node.get("a"), frame=frame, bar_index=bar_index - 1)
    b_prev = _resolve_operand(cross_node.get("b"), frame=frame, bar_index=bar_index - 1)
    op = str(cross_node.get("op", "")).lower()

    if any(_is_nan(value) for value in (a_now, b_now, a_prev, b_prev)):
        return False

    if op == "cross_above":
        return bool(a_now > b_now and a_prev <= b_prev)
    if op == "cross_below":
        return bool(a_now < b_now and a_prev >= b_prev)
    return False


def _resolve_operand(
    operand: Any,
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> float:
    if isinstance(operand, int | float) and not isinstance(operand, bool):
        return float(operand)

    if isinstance(operand, dict):
        ref = operand.get("ref")
        if not isinstance(ref, str):
            return float("nan")
        offset = operand.get("offset", 0)
        if not isinstance(offset, int):
            offset = 0
        return _resolve_ref_value(ref, frame=frame, bar_index=bar_index + offset)

    return float("nan")


def _evaluate_cmp_series(
    cmp_node: dict[str, Any],
    *,
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    left = _resolve_operand_series(cmp_node.get("left"), frame=frame)
    right = _resolve_operand_series(cmp_node.get("right"), frame=frame)
    op = str(cmp_node.get("op", "")).lower()

    valid = ~(np.isnan(left) | np.isnan(right))
    if op == "gt":
        return valid & (left > right)
    if op == "gte":
        return valid & (left >= right)
    if op == "lt":
        return valid & (left < right)
    if op == "lte":
        return valid & (left <= right)
    if op == "eq":
        return valid & (left == right)
    if op == "neq":
        return valid & (left != right)
    return np.zeros(len(frame), dtype=bool)


def _evaluate_cross_series(
    cross_node: dict[str, Any],
    *,
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    size = len(frame)
    if size == 0:
        return np.zeros(0, dtype=bool)

    a_now = _resolve_operand_series(cross_node.get("a"), frame=frame)
    b_now = _resolve_operand_series(cross_node.get("b"), frame=frame)
    a_prev = _previous_series(a_now)
    b_prev = _previous_series(b_now)
    op = str(cross_node.get("op", "")).lower()

    valid = ~(np.isnan(a_now) | np.isnan(b_now) | np.isnan(a_prev) | np.isnan(b_prev))
    if op == "cross_above":
        result = valid & (a_now > b_now) & (a_prev <= b_prev)
    elif op == "cross_below":
        result = valid & (a_now < b_now) & (a_prev >= b_prev)
    else:
        result = np.zeros(size, dtype=bool)

    if size > 0:
        result[0] = False
    return result


def _resolve_operand_series(
    operand: Any,
    *,
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    size = len(frame)
    if isinstance(operand, int | float) and not isinstance(operand, bool):
        return np.full(size, float(operand), dtype=np.float64)

    if isinstance(operand, dict):
        ref = operand.get("ref")
        if not isinstance(ref, str):
            return np.full(size, np.nan, dtype=np.float64)
        offset = operand.get("offset", 0)
        if not isinstance(offset, int):
            offset = 0
        return _apply_offset(_resolve_ref_series(ref, frame=frame), offset)

    return np.full(size, np.nan, dtype=np.float64)


def _resolve_ref_value(
    ref: str,
    *,
    frame: pd.DataFrame,
    bar_index: int,
) -> float:
    if bar_index < 0 or bar_index >= len(frame):
        return float("nan")

    array = _column_array(frame, ref)
    if array is None or bar_index >= len(array):
        return float("nan")

    raw = array[bar_index]
    if isinstance(raw, (bool, np.bool_)):
        return float(raw)
    if isinstance(raw, Real):
        return float(raw)
    return float("nan")


def _resolve_ref_series(
    ref: str,
    *,
    frame: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    size = len(frame)
    array = _column_array(frame, ref)
    if array is None:
        return np.full(size, np.nan, dtype=np.float64)
    return _as_numeric_series(array, size=size)


def _is_nan(value: float) -> bool:
    return math.isnan(value)


def _is_truthy(value: float) -> bool:
    if _is_nan(value):
        return False
    return bool(value)


def _nan_operand(frame: pd.DataFrame, bar_index: int) -> float:
    del frame, bar_index
    return float("nan")


def _previous_series(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    size = len(values)
    if size == 0:
        return values.copy()
    shifted = np.empty(size, dtype=np.float64)
    shifted[0] = np.nan
    shifted[1:] = values[:-1]
    return shifted


def _apply_offset(
    values: np.ndarray[Any, np.dtype[np.float64]],
    offset: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if offset == 0:
        return values

    size = len(values)
    shifted = np.full(size, np.nan, dtype=np.float64)
    if offset > 0:
        if offset < size:
            shifted[: size - offset] = values[offset:]
        return shifted

    back = -offset
    if back < size:
        shifted[back:] = values[: size - back]
    return shifted


def _truthy_mask(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.bool_]]:
    return (~np.isnan(values)) & (values != 0.0)


def _as_numeric_series(raw_array: Any, *, size: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(raw_array)
    if array.ndim != 1:
        array = array.reshape(-1)
    if len(array) > size:
        array = array[:size]

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float64, copy=False)

    if np.issubdtype(array.dtype, np.number):
        return array.astype(np.float64, copy=False)

    values = np.full(size, np.nan, dtype=np.float64)
    limit = min(size, len(array))
    for idx in range(limit):
        raw = array[idx]
        if isinstance(raw, (bool, np.bool_)):
            values[idx] = float(raw)
        elif isinstance(raw, Real):
            values[idx] = float(raw)
    return values


def _column_array(frame: pd.DataFrame, ref: str) -> Any | None:
    cache = _frame_column_cache(frame)
    if cache is None:
        if ref not in frame.columns:
            return None
        return frame[ref].to_numpy(copy=False)

    array = cache.get(ref)
    if array is None:
        if ref not in frame.columns:
            return None
        array = frame[ref].to_numpy(copy=False)
        cache[ref] = array
    return array


def _frame_column_cache(frame: pd.DataFrame) -> dict[str, np.ndarray[Any, Any]] | None:
    frame_id = id(frame)
    cached = _FRAME_COLUMN_ARRAY_CACHE.get(frame_id)
    if cached is not None:
        frame_ref, cache = cached
        if frame_ref() is frame:
            return cache

    try:
        frame_ref = weakref.ref(frame)
    except TypeError:
        return None

    cache: dict[str, np.ndarray[Any, Any]] = {}
    _FRAME_COLUMN_ARRAY_CACHE[frame_id] = (frame_ref, cache)
    if len(_FRAME_COLUMN_ARRAY_CACHE) > 64:
        stale_ids = [
            cached_frame_id
            for cached_frame_id, (cached_ref, _) in _FRAME_COLUMN_ARRAY_CACHE.items()
            if cached_ref() is None
        ]
        for stale_id in stale_ids:
            _FRAME_COLUMN_ARRAY_CACHE.pop(stale_id, None)
    return cache
