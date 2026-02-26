"""QuantStats wrapper that returns stable, serializable metrics."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs

_DEFAULT_PERIODS_PER_YEAR = 252
_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def build_quantstats_performance(
    *,
    returns: list[float],
    timestamps: list[datetime],
    risk_free_rate: float = 0.0,
    max_series_points: int = 5_000,
) -> dict[str, Any]:
    """Compute performance metrics with QuantStats in a stable schema."""

    aligned = _build_returns_series(returns=returns, timestamps=timestamps)
    if aligned.empty:
        return {
            "library": "quantstats",
            "metrics": {},
            "series": {
                "cumulative_returns": [],
                "drawdown": [],
            },
        }

    periods_per_year = _infer_periods_per_year(aligned.index)
    metrics = {
        "cagr": _safe_metric(
            lambda: qs.stats.cagr(
                aligned,
                periods=periods_per_year,
            )
        ),
        "sharpe": _safe_metric(
            lambda: qs.stats.sharpe(
                aligned,
                rf=risk_free_rate,
                periods=periods_per_year,
            )
        ),
        "sortino": _safe_metric(
            lambda: qs.stats.sortino(
                aligned,
                rf=risk_free_rate,
                periods=periods_per_year,
            )
        ),
        "calmar": _safe_metric(
            lambda: qs.stats.calmar(
                aligned,
                periods=periods_per_year,
            )
        ),
        "volatility": _safe_metric(
            lambda: qs.stats.volatility(
                aligned,
                periods=periods_per_year,
            )
        ),
        "max_drawdown": _safe_metric(lambda: qs.stats.max_drawdown(aligned)),
        "win_rate": _safe_metric(lambda: qs.stats.win_rate(aligned)),
        "profit_factor": _safe_metric(lambda: qs.stats.profit_factor(aligned)),
        "avg_return": _safe_metric(lambda: qs.stats.avg_return(aligned)),
        "value_at_risk": _safe_metric(lambda: qs.stats.var(aligned)),
        "conditional_var": _safe_conditional_var(aligned),
    }

    cumulative = (1.0 + aligned).cumprod() - 1.0
    drawdown = qs.stats.to_drawdown_series(aligned)
    sampled_cumulative = _sample_series(cumulative, max_points=max_series_points)
    sampled_drawdown = _sample_series(drawdown, max_points=max_series_points)
    return {
        "library": "quantstats",
        "metrics": metrics,
        "series": {
            "cumulative_returns": _series_to_records(sampled_cumulative),
            "drawdown": _series_to_records(sampled_drawdown),
        },
    }


def _build_returns_series(
    *,
    returns: list[float],
    timestamps: list[datetime],
) -> pd.Series:
    if not returns:
        return pd.Series(dtype=float)
    if len(timestamps) != len(returns):
        return pd.Series(dtype=float)

    series = pd.Series(returns, index=pd.DatetimeIndex(timestamps))
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series.astype(float)


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return _DEFAULT_PERIODS_PER_YEAR

    unique_sorted = pd.DatetimeIndex(index.unique()).sort_values()
    if len(unique_sorted) < 2:
        return _DEFAULT_PERIODS_PER_YEAR

    total_seconds = (unique_sorted[-1] - unique_sorted[0]).total_seconds()
    if total_seconds <= 0:
        return _DEFAULT_PERIODS_PER_YEAR

    periods = (len(unique_sorted) - 1) * (_SECONDS_PER_YEAR / total_seconds)
    if not np.isfinite(periods) or periods <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    return max(1, int(round(periods)))


def _safe_metric(getter: Callable[[], Any]) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = getter()
    except Exception:  # noqa: BLE001
        return None
    return _to_serializable_float(value)


def _safe_conditional_var(returns: pd.Series) -> float | None:
    var = _safe_metric(lambda: qs.stats.var(returns))
    if var is None:
        return None

    tail = returns[returns < var]
    if tail.empty:
        return None
    return _to_serializable_float(tail.mean())


def _to_serializable_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int | float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return None


def _series_to_records(series: pd.Series) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for timestamp, value in series.items():
        float_value = _to_serializable_float(value)
        if float_value is None:
            continue
        if isinstance(timestamp, pd.Timestamp):
            ts = timestamp.isoformat()
        else:
            ts = str(timestamp)
        records.append(
            {
                "timestamp": ts,
                "value": float_value,
            }
        )
    return records


def _sample_series(series: pd.Series, *, max_points: int) -> pd.Series:
    cap = max(0, int(max_points))
    if cap == 0:
        return series.iloc[0:0]
    total = len(series)
    if total <= cap:
        return series
    if cap == 1:
        return series.iloc[-1:]
    positions = np.linspace(0, total - 1, num=cap, dtype=int)
    return series.iloc[positions]
