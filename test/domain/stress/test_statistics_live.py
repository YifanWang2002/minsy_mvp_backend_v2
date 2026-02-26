from __future__ import annotations

from packages.domain.stress.statistics import (
    annualized_stability_score,
    conditional_value_at_risk,
    confidence_intervals,
    histogram,
    percentile_summary,
    risk_of_ruin,
    value_at_risk,
)


_VALUES = [-10.0, -5.0, -1.0, 0.0, 2.0, 5.0, 9.0]


def test_000_accessibility_statistics_confidence_intervals() -> None:
    intervals = confidence_intervals(_VALUES)
    assert set(intervals.keys()) == {"90", "95", "99"}
    assert intervals["95"]["lower"] <= intervals["95"]["upper"]


def test_010_statistics_var_cvar_and_ruin() -> None:
    var95 = value_at_risk(_VALUES, confidence=0.95)
    cvar95 = conditional_value_at_risk(_VALUES, confidence=0.95)
    ruin = risk_of_ruin(_VALUES, ruin_threshold_pct=-5.0)

    assert cvar95 <= var95
    assert 0.0 <= ruin <= 1.0


def test_020_statistics_histogram_percentile_and_stability() -> None:
    hist = histogram(_VALUES, bins=5)
    summary = percentile_summary(_VALUES)
    stability = annualized_stability_score([0.001, -0.001, 0.002, -0.0005])

    assert hist
    assert all("density" in row for row in hist)
    assert summary["p01"] <= summary["p99"]
    assert isinstance(stability, float)
