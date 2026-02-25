"""Pareto frontier utilities."""

from __future__ import annotations

from typing import Any


def pareto_front_indices(
    *,
    rows: list[dict[str, Any]],
    objective_keys: list[str],
) -> set[int]:
    """Return indices of non-dominated rows (maximize all objective keys)."""

    if not rows:
        return set()
    keys = [str(item).strip() for item in objective_keys if str(item).strip()]
    if not keys:
        return set(range(len(rows)))

    front: set[int] = set()
    for i, left in enumerate(rows):
        dominated = False
        for j, right in enumerate(rows):
            if i == j:
                continue
            if _dominates(right, left, keys=keys):
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def build_metric_points(
    *,
    rows: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
) -> list[dict[str, Any]]:
    """Project rows into 2D metric space for plotting."""

    x_key = str(x_metric).strip()
    y_key = str(y_metric).strip()
    output: list[dict[str, Any]] = []
    for item in rows:
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        x = float(metrics.get(x_key, 0.0)) if isinstance(metrics, dict) else 0.0
        y = float(metrics.get(y_key, 0.0)) if isinstance(metrics, dict) else 0.0
        output.append(
            {
                "x": x,
                "y": y,
                "trial_no": int(item.get("trial_no", 0)) if isinstance(item, dict) else 0,
                "params": item.get("params", {}) if isinstance(item, dict) else {},
            }
        )
    return output


def _dominates(left: dict[str, Any], right: dict[str, Any], *, keys: list[str]) -> bool:
    left_metrics = left.get("objective_values", {}) if isinstance(left, dict) else {}
    right_metrics = right.get("objective_values", {}) if isinstance(right, dict) else {}
    if not isinstance(left_metrics, dict) or not isinstance(right_metrics, dict):
        return False

    ge_all = True
    gt_any = False
    for key in keys:
        left_value = float(left_metrics.get(key, 0.0))
        right_value = float(right_metrics.get(key, 0.0))
        if left_value < right_value:
            ge_all = False
            break
        if left_value > right_value:
            gt_any = True
    return ge_all and gt_any
