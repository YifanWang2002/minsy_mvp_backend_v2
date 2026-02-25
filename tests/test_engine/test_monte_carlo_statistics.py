from __future__ import annotations

from src.engine.stress.monte_carlo import run_monte_carlo
from src.engine.stress.statistics import (
    conditional_value_at_risk,
    confidence_intervals,
    histogram,
    risk_of_ruin,
    value_at_risk,
)


def test_monte_carlo_reproducible_with_same_seed() -> None:
    returns = [0.01, -0.02, 0.005, 0.03, -0.01]
    first = run_monte_carlo(
        returns=returns,
        num_trials=200,
        horizon_bars=20,
        method="iid_bootstrap",
        seed=7,
    )
    second = run_monte_carlo(
        returns=returns,
        num_trials=200,
        horizon_bars=20,
        method="iid_bootstrap",
        seed=7,
    )

    assert first.final_returns_pct.shape == second.final_returns_pct.shape
    assert (first.final_returns_pct == second.final_returns_pct).all()


def test_statistics_helpers_return_expected_shapes() -> None:
    values = [-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    ci = confidence_intervals(values)
    assert "95" in ci
    assert ci["95"]["lower"] <= ci["95"]["upper"]

    bins = histogram(values, bins=6)
    assert bins
    assert sum(item["count"] for item in bins) == len(values)

    var_95 = value_at_risk(values, confidence=0.95)
    cvar_95 = conditional_value_at_risk(values, confidence=0.95)
    assert cvar_95 <= var_95

    ruin = risk_of_ruin(values, ruin_threshold_pct=-15.0)
    assert 0.0 <= ruin <= 1.0
