from __future__ import annotations

from packages.domain.backtest.analytics import (
    build_backtest_overview,
    compute_equity_curve,
    compute_rolling_metrics,
)


def test_000_accessibility_backtest_overview_invalid_result_fallback() -> None:
    overview = build_backtest_overview(None, sample_trades=5, sample_events=5)
    assert overview["summary"] == {}
    assert overview["counts"]["trades"] == 0


def test_010_backtest_equity_and_rolling_invalid_result_fallback() -> None:
    equity = compute_equity_curve(None)
    rolling = compute_rolling_metrics(None)

    assert equity["point_count"] == 0
    assert equity["curve_points"] == []
    assert rolling["point_count_total"] == 0
    assert rolling["sharpe_curve_points"] == []
