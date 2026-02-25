"""Objective and constraint helpers for optimization."""

from __future__ import annotations

from typing import Any


def compute_objective_score(
    *,
    metrics: dict[str, Any],
    objectives: list[str],
) -> tuple[dict[str, float], float]:
    """Compute per-objective values and aggregate score."""

    normalized = _normalize_metrics(metrics)
    if not objectives:
        objectives = ["max_return", "min_drawdown", "max_stability"]

    per_objective: dict[str, float] = {}
    total = 0.0
    for objective in objectives:
        key = str(objective).strip().lower()
        value = _score_one(key, normalized)
        per_objective[key] = value
        total += value

    return per_objective, float(total)


def constraints_satisfied(
    *,
    metrics: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> bool:
    """Evaluate lightweight threshold constraints."""

    if not isinstance(constraints, dict) or not constraints:
        return True

    normalized = _normalize_metrics(metrics)

    max_drawdown = constraints.get("max_drawdown_pct")
    if isinstance(max_drawdown, int | float):
        if abs(normalized.get("max_drawdown_pct", 0.0)) > float(max_drawdown):
            return False

    min_return = constraints.get("min_return_pct")
    if isinstance(min_return, int | float):
        if normalized.get("total_return_pct", 0.0) < float(min_return):
            return False

    min_stability = constraints.get("min_stability")
    if isinstance(min_stability, int | float):
        if normalized.get("stability_score", 0.0) < float(min_stability):
            return False

    return True


def _score_one(objective: str, metrics: dict[str, float]) -> float:
    if objective in {"max_return", "max_total_return", "max_return_pct"}:
        return float(metrics.get("total_return_pct", 0.0))
    if objective in {"min_drawdown", "min_drawdown_pct", "max_drawdown"}:
        return -abs(float(metrics.get("max_drawdown_pct", 0.0)))
    if objective in {"max_stability", "max_stability_score"}:
        return float(metrics.get("stability_score", 0.0))
    if objective in {"max_sharpe"}:
        return float(metrics.get("sharpe", 0.0))
    return 0.0


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("total_return_pct", "max_drawdown_pct", "stability_score", "sharpe"):
        raw = metrics.get(key)
        if isinstance(raw, int | float):
            output[key] = float(raw)
        else:
            output[key] = 0.0
    return output
