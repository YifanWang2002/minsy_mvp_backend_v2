"""Parameter sensitivity scan helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from src.engine.strategy.param_mutation import TunableParam, apply_param_values


@dataclass(frozen=True, slots=True)
class ParamVariantResult:
    """One mutated-parameter backtest result."""

    key: str
    factor_id: str
    param_name: str
    pct_delta: float
    value: float
    metrics: dict[str, Any]


def scan_param_sensitivity(
    *,
    base_payload: dict[str, Any],
    params: list[TunableParam],
    scan_pct: float,
    steps_per_side: int,
    evaluator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[ParamVariantResult], list[dict[str, Any]], float]:
    """Mutate each param around base value and collect response metrics."""

    scan_ratio = max(0.0, float(scan_pct)) / 100.0
    steps = max(1, int(steps_per_side))
    variants: list[ParamVariantResult] = []

    for param in params:
        deltas = _build_delta_grid(scan_ratio=scan_ratio, steps=steps)
        for delta in deltas:
            value = _mutated_value(param.current_value, pct_delta=delta)
            mutated = apply_param_values(
                base_payload,
                values={param.key: value},
            )
            metrics = evaluator(mutated)
            variants.append(
                ParamVariantResult(
                    key=param.key,
                    factor_id=param.factor_id,
                    param_name=param.param_name,
                    pct_delta=float(delta),
                    value=float(value),
                    metrics=dict(metrics),
                )
            )

    response_curves = _build_response_curves(variants)
    fragile_rank = _build_fragile_rank(response_curves)
    stability_score = _compute_stability_score(response_curves)
    return variants, fragile_rank, stability_score


def _build_delta_grid(*, scan_ratio: float, steps: int) -> list[float]:
    if scan_ratio <= 0:
        return [0.0]
    step = scan_ratio / float(steps)
    values = [0.0]
    for idx in range(1, steps + 1):
        values.append(step * idx)
        values.append(-step * idx)
    return sorted(set(values))


def _mutated_value(base: float, *, pct_delta: float) -> float:
    mutated = float(base) * (1.0 + float(pct_delta))
    if abs(mutated) < 1e-12:
        return 0.0
    return mutated


def _build_response_curves(variants: list[ParamVariantResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ParamVariantResult]] = defaultdict(list)
    for item in variants:
        grouped[item.key].append(item)

    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda item: item.pct_delta)
        output.append(
            {
                "param_key": key,
                "points": [
                    {
                        "pct_delta": item.pct_delta,
                        "value": item.value,
                        "metrics": item.metrics,
                    }
                    for item in sorted_rows
                ],
            }
        )
    output.sort(key=lambda item: str(item.get("param_key", "")))
    return output


def _build_fragile_rank(response_curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for curve in response_curves:
        points = curve.get("points", [])
        if not isinstance(points, list) or not points:
            continue
        baseline = next(
            (
                item
                for item in points
                if isinstance(item, dict) and float(item.get("pct_delta", 0.0)) == 0.0
            ),
            None,
        )
        if not isinstance(baseline, dict):
            continue
        baseline_metrics = baseline.get("metrics", {})
        base_return = float((baseline_metrics or {}).get("total_return_pct", 0.0))

        worst_drop = 0.0
        for item in points:
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            candidate_return = float((metrics or {}).get("total_return_pct", 0.0))
            worst_drop = max(worst_drop, base_return - candidate_return)

        scored.append(
            {
                "param_key": curve.get("param_key"),
                "fragility_score": float(worst_drop),
                "baseline_return_pct": base_return,
            }
        )

    scored.sort(key=lambda item: float(item.get("fragility_score", 0.0)), reverse=True)
    return scored


def _compute_stability_score(response_curves: list[dict[str, Any]]) -> float:
    if not response_curves:
        return 0.0

    all_variances: list[float] = []
    for curve in response_curves:
        points = curve.get("points", [])
        if not isinstance(points, list) or not points:
            continue
        values: list[float] = []
        for item in points:
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            values.append(float((metrics or {}).get("total_return_pct", 0.0)))
        if len(values) <= 1:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        all_variances.append(var)

    if not all_variances:
        return 100.0
    avg_variance = sum(all_variances) / len(all_variances)
    return float(max(0.0, 100.0 - avg_variance))
