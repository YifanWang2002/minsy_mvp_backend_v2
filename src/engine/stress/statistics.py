"""Statistical helpers for stress simulations."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np


def confidence_intervals(
    values: Sequence[float],
    *,
    levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> dict[str, dict[str, float]]:
    """Return percentile-based confidence intervals."""

    array = _as_float_array(values)
    if array.size == 0:
        return {f"{int(level * 100)}": {"lower": 0.0, "upper": 0.0} for level in levels}

    output: dict[str, dict[str, float]] = {}
    for level in levels:
        alpha = max(0.0, min(1.0, 1.0 - float(level)))
        lower = float(np.quantile(array, alpha / 2.0))
        upper = float(np.quantile(array, 1.0 - alpha / 2.0))
        output[str(int(level * 100))] = {"lower": lower, "upper": upper}
    return output


def risk_of_ruin(
    values: Sequence[float],
    *,
    ruin_threshold_pct: float,
) -> float:
    """Probability of ending below ruin threshold."""

    array = _as_float_array(values)
    if array.size == 0:
        return 0.0

    threshold = float(ruin_threshold_pct)
    return float(np.mean(array <= threshold))


def value_at_risk(values: Sequence[float], *, confidence: float = 0.95) -> float:
    """Compute one-sided VaR (left tail)."""

    array = _as_float_array(values)
    if array.size == 0:
        return 0.0
    alpha = 1.0 - max(0.0, min(1.0, float(confidence)))
    return float(np.quantile(array, alpha))


def conditional_value_at_risk(values: Sequence[float], *, confidence: float = 0.95) -> float:
    """Compute CVaR/Expected Shortfall for left tail."""

    array = _as_float_array(values)
    if array.size == 0:
        return 0.0
    var = value_at_risk(array, confidence=confidence)
    tail = array[array <= var]
    if tail.size == 0:
        return var
    return float(np.mean(tail))


def histogram(values: Sequence[float], *, bins: int = 30) -> list[dict[str, float]]:
    """Build histogram bins for MCP-friendly payloads."""

    array = _as_float_array(values)
    if array.size == 0:
        return []

    counts, edges = np.histogram(array, bins=max(5, int(bins)))
    total = float(array.size)
    output: list[dict[str, float]] = []
    for idx, count in enumerate(counts):
        output.append(
            {
                "left": float(edges[idx]),
                "right": float(edges[idx + 1]),
                "count": int(count),
                "density": float(count / total),
            }
        )
    return output


def annualized_stability_score(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    """Simple stability score using Sharpe-like normalization."""

    array = _as_float_array(returns)
    if array.size < 2:
        return 0.0

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float((mean / std) * sqrt(float(periods_per_year)))


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    """Compact quantile summary."""

    array = _as_float_array(values)
    if array.size == 0:
        return {
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def to_jsonable_number(value: Any) -> float:
    """Best-effort float conversion."""

    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = values.astype(float)
    else:
        array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    return array[np.isfinite(array)]
