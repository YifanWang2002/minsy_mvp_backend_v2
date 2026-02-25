"""Monte Carlo return simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

MonteCarloMethod = Literal["iid_bootstrap", "block_bootstrap", "trade_shuffle"]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Simulation output for downstream statistics."""

    method: MonteCarloMethod
    num_trials: int
    horizon_bars: int
    final_returns_pct: np.ndarray
    avg_path: np.ndarray


def run_monte_carlo(
    *,
    returns: list[float],
    num_trials: int,
    horizon_bars: int,
    method: MonteCarloMethod,
    seed: int,
) -> MonteCarloResult:
    """Run Monte Carlo simulations over strategy return samples."""

    source = np.asarray(list(returns), dtype=float)
    source = source[np.isfinite(source)]
    if source.size == 0:
        raise ValueError("returns cannot be empty")

    trials = max(1, int(num_trials))
    horizon = max(1, int(horizon_bars))
    rng = np.random.default_rng(seed)

    final_returns: list[float] = []
    path_sum = np.zeros(horizon, dtype=float)

    for _ in range(trials):
        sampled = _sample_path(
            source=source,
            horizon=horizon,
            method=method,
            rng=rng,
        )
        growth_curve = np.cumprod(1.0 + sampled)
        final_returns.append(float((growth_curve[-1] - 1.0) * 100.0))
        path_sum += growth_curve

    avg_path = (path_sum / float(trials) - 1.0) * 100.0
    return MonteCarloResult(
        method=method,
        num_trials=trials,
        horizon_bars=horizon,
        final_returns_pct=np.asarray(final_returns, dtype=float),
        avg_path=avg_path,
    )


def _sample_path(
    *,
    source: np.ndarray,
    horizon: int,
    method: MonteCarloMethod,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "iid_bootstrap":
        indices = rng.integers(low=0, high=source.size, size=horizon)
        return source[indices]

    if method == "trade_shuffle":
        if source.size >= horizon:
            shuffled = np.array(source, copy=True)
            rng.shuffle(shuffled)
            return shuffled[:horizon]
        repeats = int(np.ceil(horizon / source.size))
        stacked = np.tile(source, repeats)
        rng.shuffle(stacked)
        return stacked[:horizon]

    if method == "block_bootstrap":
        block = max(2, int(np.sqrt(horizon)))
        sampled: list[float] = []
        while len(sampled) < horizon:
            if source.size <= block:
                start = 0
            else:
                start = int(rng.integers(0, source.size - block + 1))
            sampled.extend(source[start : start + block].tolist())
        return np.asarray(sampled[:horizon], dtype=float)

    raise ValueError(f"Unsupported monte carlo method: {method}")
