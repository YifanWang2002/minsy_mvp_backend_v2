"""Search candidate generators for optimization jobs."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from src.engine.optimization.search_space import SearchDimension


def generate_grid_candidates(
    *,
    dimensions: list[SearchDimension],
    budget: int,
) -> list[dict[str, float]]:
    """Generate deterministic cartesian candidates (budget-capped)."""

    axes: list[list[float]] = []
    for dim in dimensions:
        if dim.max_value <= dim.min_value:
            axes.append([dim.min_value])
            continue

        steps = max(1, int(round((dim.max_value - dim.min_value) / dim.step)))
        values = np.linspace(dim.min_value, dim.max_value, num=steps + 1)
        axes.append([float(item) for item in values])

    output: list[dict[str, float]] = []
    for combo in itertools.product(*axes):
        output.append({dim.key: float(value) for dim, value in zip(dimensions, combo, strict=True)})
        if len(output) >= max(1, int(budget)):
            break
    return output


def generate_random_candidates(
    *,
    dimensions: list[SearchDimension],
    budget: int,
    seed: int,
) -> list[dict[str, float]]:
    """Generate random uniform samples."""

    rng = np.random.default_rng(seed)
    output: list[dict[str, float]] = []
    for _ in range(max(1, int(budget))):
        candidate: dict[str, float] = {}
        for dim in dimensions:
            if dim.max_value <= dim.min_value:
                value = dim.min_value
            else:
                value = float(rng.uniform(dim.min_value, dim.max_value))
            candidate[dim.key] = value
        output.append(candidate)
    return output


def generate_bayes_like_candidates(
    *,
    dimensions: list[SearchDimension],
    budget: int,
    seed: int,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, float]]:
    """Lightweight Bayesian-like heuristic without external dependencies."""

    rng = np.random.default_rng(seed)
    output: list[dict[str, float]] = []
    observed = list(history or [])

    for _ in range(max(1, int(budget))):
        if len(observed) < max(5, len(dimensions) * 2):
            candidate = {
                dim.key: float(rng.uniform(dim.min_value, dim.max_value))
                if dim.max_value > dim.min_value
                else dim.min_value
                for dim in dimensions
            }
            output.append(candidate)
            observed.append({"params": candidate, "score": 0.0})
            continue

        best = max(
            observed,
            key=lambda item: float(item.get("score", 0.0)),
        )
        best_params = best.get("params", {}) if isinstance(best, dict) else {}

        candidate: dict[str, float] = {}
        for dim in dimensions:
            center = float(best_params.get(dim.key, dim.min_value))
            span = max(dim.step, (dim.max_value - dim.min_value) * 0.15)
            lower = max(dim.min_value, center - span)
            upper = min(dim.max_value, center + span)
            if upper <= lower:
                value = lower
            else:
                value = float(rng.uniform(lower, upper))
            candidate[dim.key] = value

        output.append(candidate)
        observed.append({"params": candidate, "score": 0.0})

    return output
