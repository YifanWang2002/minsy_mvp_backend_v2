from __future__ import annotations

from packages.domain.backtest.analytics import (
    build_backtest_overview,
    compute_entry_weekday_pnl,
    compute_equity_curve,
    compute_rolling_metrics,
)


_RESULT = {
    "market": "stocks",
    "symbol": "SPY",
    "timeframe": "1d",
    "summary": {"net_profit": 120.0},
    "trades": [
        {
            "pnl": 12.0,
            "pnl_pct": 1.2,
            "commission": 0.2,
            "entry_price": 100.0,
            "exit_price": 112.0,
            "quantity": 1.0,
            "bars_held": 2,
            "entry_time": "2024-01-02T00:00:00Z",
            "exit_time": "2024-01-03T00:00:00Z",
            "side": "long",
            "exit_reason": "signal_exit",
        },
        {
            "pnl": -4.0,
            "pnl_pct": -0.4,
            "commission": 0.2,
            "entry_price": 110.0,
            "exit_price": 106.0,
            "quantity": 1.0,
            "bars_held": 1,
            "entry_time": "2024-01-04T00:00:00Z",
            "exit_time": "2024-01-05T00:00:00Z",
            "side": "long",
            "exit_reason": "stop_loss",
        },
    ],
    "events": [{"event": "open"}, {"event": "close"}],
    "equity_curve": [
        {"ts": "2024-01-01T00:00:00Z", "equity": 10000.0},
        {"ts": "2024-01-02T00:00:00Z", "equity": 10012.0},
        {"ts": "2024-01-03T00:00:00Z", "equity": 10008.0},
        {"ts": "2024-01-04T00:00:00Z", "equity": 10020.0},
    ],
    "returns": [0.0012, -0.0004, 0.0012],
}


def test_000_accessibility_backtest_overview_contract() -> None:
    overview = build_backtest_overview(_RESULT, sample_trades=2, sample_events=2)
    assert overview["market"] == "stocks"
    assert overview["symbol"] == "SPY"
    assert overview["counts"]["trades"] == 2
    assert "performance" in overview


def test_010_backtest_equity_and_rolling_metrics_contract() -> None:
    equity = compute_equity_curve(_RESULT, max_points=200, sampling_mode="auto")
    rolling = compute_rolling_metrics(_RESULT, window_bars=2)

    assert equity["point_count_total"] >= equity["point_count"]
    assert "curve_points" in equity
    assert rolling["window_bars"] >= 2
    assert "sharpe_curve_points" in rolling


def test_020_backtest_weekday_analytics_contract() -> None:
    weekday = compute_entry_weekday_pnl(_RESULT)
    assert weekday["total_trades"] == 2
    assert len(weekday["weekday_stats"]) == 7
