from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.engine.backtest.analytics import (
    build_backtest_overview,
    build_compact_performance_payload,
    compute_entry_hour_pnl_heatmap,
    compute_entry_weekday_pnl,
    compute_equity_curve,
    compute_exit_reason_breakdown,
    compute_holding_period_pnl_bins,
    compute_long_short_breakdown,
    compute_monthly_return_table,
    compute_rolling_metrics,
    compute_underwater_curve,
)


def _sample_result() -> dict:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    equity = [100000.0, 100500.0, 99800.0, 101200.0, 100900.0, 102300.0]
    timestamps = [start + timedelta(hours=4 * i) for i in range(len(equity))]

    trades = [
        {
            "side": "long",
            "entry_time": timestamps[1].isoformat(),
            "exit_time": timestamps[2].isoformat(),
            "entry_price": 100.0,
            "exit_price": 102.0,
            "quantity": 1.0,
            "bars_held": 1,
            "exit_reason": "take_profit",
            "pnl": 1.8,
            "pnl_pct": 1.8,
            "commission": 0.2,
        },
        {
            "side": "short",
            "entry_time": timestamps[2].isoformat(),
            "exit_time": timestamps[4].isoformat(),
            "entry_price": 105.0,
            "exit_price": 106.0,
            "quantity": 1.0,
            "bars_held": 2,
            "exit_reason": "stop_loss",
            "pnl": -1.2,
            "pnl_pct": -1.14,
            "commission": 0.2,
        },
        {
            "side": "long",
            "entry_time": timestamps[3].isoformat(),
            "exit_time": timestamps[5].isoformat(),
            "entry_price": 106.0,
            "exit_price": 109.0,
            "quantity": 1.0,
            "bars_held": 120,
            "exit_reason": "signal_exit",
            "pnl": 2.7,
            "pnl_pct": 2.55,
            "commission": 0.3,
        },
    ]

    events = [
        {"type": "position_opened", "timestamp": ts.isoformat(), "bar_index": idx, "payload": {}}
        for idx, ts in enumerate(timestamps)
    ]

    returns = []
    for i in range(1, len(equity)):
        returns.append(equity[i] / equity[i - 1] - 1.0)

    return {
        "market": "crypto",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "summary": {
            "total_trades": 3,
            "winning_trades": 2,
            "losing_trades": 1,
            "win_rate": 2 / 3,
            "total_pnl": 3.3,
            "total_return_pct": 2.3,
            "final_equity": 102300.0,
            "max_drawdown_pct": 0.7,
        },
        "trades": trades,
        "events": events,
        "equity_curve": [
            {"timestamp": ts.isoformat(), "equity": value}
            for ts, value in zip(timestamps, equity, strict=False)
        ],
        "returns": returns,
        "performance": {
            "library": "quantstats",
            "metrics": {
                "sharpe": 1.2,
                "sortino": 1.4,
            },
            "series": {
                "cumulative_returns": [{"timestamp": timestamps[-1].isoformat(), "value": 0.023}],
                "drawdown": [{"timestamp": timestamps[-1].isoformat(), "value": -0.007}],
            },
        },
        "started_at": timestamps[0].isoformat(),
        "finished_at": timestamps[-1].isoformat(),
    }


def test_build_backtest_overview_compacts_performance_series() -> None:
    overview = build_backtest_overview(_sample_result(), sample_trades=2, sample_events=3)

    assert overview["result_truncated"] is True
    assert overview["counts"]["trades"] == 3
    assert overview["counts"]["events"] == 6
    assert len(overview["sample_trades"]) == 2
    assert len(overview["sample_events"]) == 3

    performance = overview["performance"]
    assert "series" not in performance
    assert "metrics" in performance
    assert "meta" in performance
    assert performance["metrics"]["trade_count"] == 3
    assert performance["metrics"]["trade_win_rate_pct"] == 2 / 3 * 100.0


def test_compact_performance_and_analytics_outputs() -> None:
    result = _sample_result()
    compact = build_compact_performance_payload(result)
    assert "series" not in compact
    assert compact["meta"]["returns_count"] == len(result["returns"])
    assert compact["metrics"]["expectancy"] is not None
    assert compact["metrics"]["payoff_ratio"] is not None

    hour = compute_entry_hour_pnl_heatmap(result)
    assert len(hour["heatmap"]) == 24
    assert sum(int(item["trade_count"]) for item in hour["heatmap"]) == 3

    weekday = compute_entry_weekday_pnl(result)
    assert len(weekday["weekday_stats"]) == 7

    monthly = compute_monthly_return_table(result)
    assert monthly["month_count"] >= 1

    holding = compute_holding_period_pnl_bins(result)
    assert any(item["bars_held_bin"] == "gt_100" for item in holding["holding_bins"])
    assert holding["total_trades"] == 3

    side = compute_long_short_breakdown(result)
    assert len(side["side_stats"]) == 2
    assert sum(int(item["trade_count"]) for item in side["side_stats"]) == 3

    by_reason = compute_exit_reason_breakdown(result)
    assert by_reason["total_trades"] == 3
    assert len(by_reason["exit_reason_stats"]) == 3

    underwater = compute_underwater_curve(result, max_points=3)
    assert underwater["point_count"] <= 3
    assert underwater["point_count_total"] == len(result["equity_curve"])

    rolling = compute_rolling_metrics(result, window_bars=2, max_points=3)
    assert rolling["window_bars"] == 2
    assert len(rolling["sharpe_curve_points"]) <= 3
    assert len(rolling["win_rate_curve_points"]) <= 3


def test_equity_curve_supports_sampling_and_downsampling() -> None:
    result = _sample_result()

    eod_curve = compute_equity_curve(
        result,
        sampling_mode="eod",
        max_points=10,
    )
    assert eod_curve["sampling_mode"] == "eod"
    assert eod_curve["point_count_total"] == len(result["equity_curve"])
    assert eod_curve["point_count_after_sampling"] <= eod_curve["point_count_total"]
    assert eod_curve["point_count"] <= 10

    uniform_curve = compute_equity_curve(
        result,
        sampling_mode="uniform",
        max_points=3,
    )
    assert uniform_curve["sampling_mode"] == "uniform"
    assert uniform_curve["point_count"] <= 3
    assert isinstance(uniform_curve["curve_points"], list)


def test_rolling_metrics_handles_non_datetime_returns_index() -> None:
    result = {
        "returns": [0.01, -0.005, 0.002, -0.004, 0.003],
        "equity_curve": [],
        "trades": [],
        "performance": {"library": "quantstats", "metrics": {}},
    }

    rolling = compute_rolling_metrics(result, window_bars=3, max_points=10)
    assert rolling["window_bars"] == 3
    if rolling["sharpe_curve_points"]:
        assert isinstance(rolling["sharpe_curve_points"][0]["timestamp"], str)
    if rolling["win_rate_curve_points"]:
        assert isinstance(rolling["win_rate_curve_points"][0]["timestamp"], str)


def test_rolling_metrics_uses_trade_based_win_rate_pct() -> None:
    result = _sample_result()

    rolling = compute_rolling_metrics(result, window_bars=2, max_points=20)

    assert rolling["win_rate_basis"] == "trades"
    assert rolling["win_rate_unit"] == "pct"
    assert rolling["win_rate_window_trades"] == 2
    assert rolling["win_rate_curve_points"]
    assert all(0.0 <= point["value"] <= 100.0 for point in rolling["win_rate_curve_points"])
    assert rolling["win_rate_last"] == 50.0
