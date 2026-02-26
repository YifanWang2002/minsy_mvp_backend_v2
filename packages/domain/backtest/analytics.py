"""High-level analytics for backtest job results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_DEFAULT_PERIODS_PER_YEAR = 252
_DEFAULT_MAX_POINTS = 240
_WEEKDAY_NAMES: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_HOLDING_BINS: tuple[tuple[int, str], ...] = (
    (1, "1"),
    (3, "2_3"),
    (5, "4_5"),
    (10, "6_10"),
    (20, "11_20"),
    (50, "21_50"),
    (100, "51_100"),
)


@dataclass(frozen=True, slots=True)
class _PreparedBacktestData:
    trades: pd.DataFrame
    equity: pd.DataFrame
    returns: pd.Series
    periods_per_year: int


def build_backtest_overview(
    result: Any,
    *,
    sample_trades: int,
    sample_events: int,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "summary": {},
            "performance": {},
            "counts": {"trades": 0, "equity_points": 0, "events": 0, "returns": 0},
            "sample_trades": [],
            "sample_events": [],
            "result_truncated": True,
        }

    trades = result.get("trades")
    if not isinstance(trades, list):
        trades = []
    events = result.get("events")
    if not isinstance(events, list):
        events = []
    equity_curve = result.get("equity_curve")
    if not isinstance(equity_curve, list):
        equity_curve = []
    returns = result.get("returns")
    if not isinstance(returns, list):
        returns = []

    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    compact_performance = build_compact_performance_payload(result)
    return {
        "market": result.get("market"),
        "symbol": result.get("symbol"),
        "timeframe": result.get("timeframe"),
        "summary": summary,
        "performance": compact_performance,
        "counts": {
            "trades": len(trades),
            "equity_points": len(equity_curve),
            "events": len(events),
            "returns": len(returns),
        },
        "sample_trades": trades[: max(0, sample_trades)],
        "sample_events": events[: max(0, sample_events)],
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "result_truncated": True,
    }


def build_compact_performance_payload(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"library": "quantstats", "metrics": {}, "meta": {}}

    performance = result.get("performance")
    if not isinstance(performance, dict):
        performance = {}
    metrics = performance.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    equity = prepared.equity
    returns = prepared.returns

    derived: dict[str, Any] = {}
    total_trades = len(trades)
    win_count = int((trades["pnl"] > 0).sum()) if total_trades else 0
    lose_count = int((trades["pnl"] < 0).sum()) if total_trades else 0
    avg_win = _safe_float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if win_count else None
    avg_loss_abs = (
        _safe_float(abs(trades.loc[trades["pnl"] < 0, "pnl"].mean()))
        if lose_count
        else None
    )
    payout = None
    if avg_win is not None and avg_loss_abs is not None and avg_loss_abs > 0:
        payout = avg_win / avg_loss_abs

    expectancy = _safe_float(trades["pnl"].mean()) if total_trades else None
    expectancy_pct = _safe_float(trades["pnl_pct"].mean()) if total_trades else None
    commission_total = _safe_float(trades["commission"].sum()) if total_trades else 0.0
    turnover_notional = (
        _safe_float(((trades["entry_price"] + trades["exit_price"]) * trades["quantity"]).sum())
        if total_trades
        else 0.0
    )
    gross_abs = (
        _safe_float(abs((trades["pnl"] + trades["commission"]).sum()))
        if total_trades
        else None
    )
    commission_pct_gross = None
    if (
        commission_total is not None
        and gross_abs is not None
        and gross_abs > 0
    ):
        commission_pct_gross = commission_total / gross_abs * 100.0

    total_bars = max(1, len(equity))
    exposure_pct = (
        _safe_float(min(100.0, max(0.0, trades["bars_held"].sum() / total_bars * 100.0)))
        if total_trades
        else 0.0
    )
    avg_holding = _safe_float(trades["bars_held"].mean()) if total_trades else None
    med_holding = _safe_float(trades["bars_held"].median()) if total_trades else None

    max_dd_duration, recovery_duration, current_dd_duration = _drawdown_duration_stats(equity)
    rolling_summary = _rolling_summary(
        returns=returns,
        trades=trades,
        periods_per_year=prepared.periods_per_year,
        window_bars=0,
    )

    bar_win_rate = _safe_float((returns > 0).mean()) if len(returns) else None
    trade_win_rate_pct = _safe_float(win_count / total_trades * 100.0) if total_trades else 0.0
    derived.update(
        {
            "trade_count": total_trades,
            "trade_win_rate_pct": trade_win_rate_pct,
            "bar_win_rate_ratio": bar_win_rate,
            "avg_win": avg_win,
            "avg_loss_abs": avg_loss_abs,
            "payoff_ratio": _safe_float(payout),
            "expectancy": expectancy,
            "expectancy_pct": expectancy_pct,
            "avg_holding_bars": avg_holding,
            "median_holding_bars": med_holding,
            "exposure_pct": exposure_pct,
            "turnover_notional": turnover_notional,
            "commission_total": commission_total,
            "commission_as_pct_gross_pnl": _safe_float(commission_pct_gross),
            "max_drawdown_duration_bars": max_dd_duration,
            "recovery_duration_bars": recovery_duration,
            "current_drawdown_duration_bars": current_dd_duration,
            "rolling_window_bars": rolling_summary["window_bars"],
            "rolling_sharpe_last": rolling_summary["sharpe_last"],
            "rolling_sharpe_min": rolling_summary["sharpe_min"],
            "rolling_win_rate_pct_last": rolling_summary["win_rate_last"],
            "rolling_win_rate_pct_min": rolling_summary["win_rate_min"],
            "rolling_win_rate_window_trades": rolling_summary["win_rate_window_trades"],
        }
    )

    merged_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "series"
    }
    merged_metrics.update(derived)
    return {
        "library": performance.get("library", "quantstats"),
        "metrics": merged_metrics,
        "meta": {
            "periods_per_year": prepared.periods_per_year,
            "calculation_version": "mvp_v2",
            "equity_points": len(equity),
            "returns_count": len(returns),
        },
    }


def compute_entry_hour_pnl_heatmap(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    entry_times = pd.to_datetime(trades.get("entry_time"), utc=True, errors="coerce")
    if not isinstance(entry_times, pd.Series):
        entry_times = pd.Series(index=trades.index, dtype="datetime64[ns, UTC]")
    rows: list[dict[str, Any]] = []
    for hour in range(24):
        mask = (entry_times.dt.hour == hour).fillna(False)
        subset = trades[mask]
        rows.append(_aggregate_trade_bucket(str(hour), subset, "entry_hour"))
    return {
        "heatmap": rows,
        "total_trades": len(trades),
    }


def compute_entry_weekday_pnl(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    entry_times = pd.to_datetime(trades.get("entry_time"), utc=True, errors="coerce")
    if not isinstance(entry_times, pd.Series):
        entry_times = pd.Series(index=trades.index, dtype="datetime64[ns, UTC]")
    rows: list[dict[str, Any]] = []
    for weekday, name in enumerate(_WEEKDAY_NAMES):
        mask = (entry_times.dt.weekday == weekday).fillna(False)
        subset = trades[mask]
        rows.append(_aggregate_trade_bucket(name, subset, "entry_weekday"))
    return {
        "weekday_stats": rows,
        "total_trades": len(trades),
    }


def compute_monthly_return_table(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    equity = prepared.equity
    if equity.empty:
        return {
            "monthly_stats": [],
            "month_count": 0,
            "positive_months": 0,
            "negative_months": 0,
        }

    equity_series = equity.set_index("timestamp")["equity"]
    rows: list[dict[str, Any]] = []
    positive = 0
    negative = 0
    month_index = equity_series.index.tz_localize(None).to_period("M")
    for period, group in equity_series.groupby(month_index):
        start = float(group.iloc[0])
        end = float(group.iloc[-1])
        monthly_return_pct = ((end - start) / start * 100.0) if start != 0 else 0.0
        if monthly_return_pct > 0:
            positive += 1
        elif monthly_return_pct < 0:
            negative += 1
        running_peak = group.cummax()
        underwater = (group / running_peak - 1.0) * 100.0
        max_drawdown_pct = abs(float(underwater.min())) if not underwater.empty else 0.0
        rows.append(
            {
                "month": str(period),
                "start_equity": start,
                "end_equity": end,
                "return_pct": monthly_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
            }
        )

    return {
        "monthly_stats": rows,
        "month_count": len(rows),
        "positive_months": positive,
        "negative_months": negative,
    }


def compute_holding_period_pnl_bins(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    if trades.empty:
        return {
            "holding_bins": [
                _aggregate_trade_bucket(label, pd.DataFrame(), "bars_held_bin")
                for _, label in _HOLDING_BINS
            ]
            + [_aggregate_trade_bucket("gt_100", pd.DataFrame(), "bars_held_bin")],
            "total_trades": 0,
        }

    enriched = trades.copy()
    enriched["bars_held_bin"] = enriched["bars_held"].apply(_holding_label)

    ordered_labels = [label for _, label in _HOLDING_BINS] + ["gt_100"]
    rows: list[dict[str, Any]] = []
    for label in ordered_labels:
        rows.append(
            _aggregate_trade_bucket(
                label,
                enriched[enriched["bars_held_bin"] == label],
                "bars_held_bin",
            )
        )
    return {
        "holding_bins": rows,
        "total_trades": len(trades),
    }


def compute_long_short_breakdown(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        rows.append(_aggregate_trade_bucket(side, trades[trades["side"] == side], "side"))
    return {
        "side_stats": rows,
        "total_trades": len(trades),
    }


def compute_exit_reason_breakdown(result: Any) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    trades = prepared.trades
    total = len(trades)
    rows: list[dict[str, Any]] = []
    if total == 0:
        return {"exit_reason_stats": rows, "total_trades": 0}
    for reason, subset in trades.groupby("exit_reason", sort=False):
        item = _aggregate_trade_bucket(str(reason), subset, "exit_reason")
        trades_in_bucket = int(item.get("trade_count", 0))
        item["trade_share_pct"] = trades_in_bucket / total * 100.0
        rows.append(item)
    rows.sort(key=lambda item: item.get("trade_count", 0), reverse=True)
    return {
        "exit_reason_stats": rows,
        "total_trades": total,
    }


def compute_underwater_curve(
    result: Any,
    *,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    equity = prepared.equity
    if equity.empty:
        return {
            "max_drawdown_pct": 0.0,
            "time_underwater_pct": 0.0,
            "curve_points": [],
            "point_count": 0,
            "point_count_total": 0,
        }

    series = equity.set_index("timestamp")["equity"].astype(float)
    running_peak = series.cummax()
    underwater = (series / running_peak - 1.0) * 100.0
    curve = pd.DataFrame({"timestamp": underwater.index, "underwater_pct": underwater.values})
    total_points = len(curve)
    curve = _downsample_frame(curve, max_points=max_points)
    points = [
        {"timestamp": _timestamp_to_iso(ts), "underwater_pct": float(value)}
        for ts, value in zip(curve["timestamp"], curve["underwater_pct"], strict=False)
    ]
    return {
        "max_drawdown_pct": abs(float(underwater.min())) if not underwater.empty else 0.0,
        "time_underwater_pct": float((underwater < 0).mean() * 100.0),
        "curve_points": points,
        "point_count": len(points),
        "point_count_total": total_points,
    }


def compute_equity_curve(
    result: Any,
    *,
    sampling_mode: str = "auto",
    max_points: int = _DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    equity = prepared.equity
    if equity.empty:
        return {
            "sampling_mode": "uniform",
            "start_equity": None,
            "end_equity": None,
            "total_return_pct": None,
            "curve_points": [],
            "point_count": 0,
            "point_count_total": 0,
            "point_count_after_sampling": 0,
        }

    normalized_mode = _normalize_sampling_mode(sampling_mode)
    total_points = len(equity)
    resolved_mode = normalized_mode
    if normalized_mode == "auto":
        # Long intraday ranges are easier to read when first reduced to EOD.
        resolved_mode = "eod" if total_points > max(120, int(max_points) * 3) else "uniform"

    sampled = equity
    if resolved_mode == "eod":
        sampled = _sample_end_of_day(equity)

    sampled_point_count = len(sampled)
    sampled = _downsample_frame(sampled, max_points=max_points)
    points = [
        {"timestamp": _timestamp_to_iso(ts), "equity": float(value)}
        for ts, value in zip(sampled["timestamp"], sampled["equity"], strict=False)
    ]

    start_equity = _safe_float(equity["equity"].iloc[0]) if total_points else None
    end_equity = _safe_float(equity["equity"].iloc[-1]) if total_points else None
    total_return_pct = None
    if (
        start_equity is not None
        and end_equity is not None
        and start_equity != 0
    ):
        total_return_pct = (end_equity / start_equity - 1.0) * 100.0

    return {
        "sampling_mode": resolved_mode,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_return_pct": _safe_float(total_return_pct),
        "curve_points": points,
        "point_count": len(points),
        "point_count_total": total_points,
        "point_count_after_sampling": sampled_point_count,
    }


def compute_rolling_metrics(
    result: Any,
    *,
    window_bars: int = 0,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    prepared = _prepare_backtest_data(result)
    summary = _rolling_summary(
        returns=prepared.returns,
        trades=prepared.trades,
        periods_per_year=prepared.periods_per_year,
        window_bars=window_bars,
    )

    sharpe_series = summary.pop("sharpe_series")
    win_series = summary.pop("win_rate_series")
    sharpe_series = _downsample_frame(sharpe_series, max_points=max_points)
    win_series = _downsample_frame(win_series, max_points=max_points)
    sharpe_points = [
        {"timestamp": _timestamp_to_iso(ts), "value": float(value)}
        for ts, value in zip(sharpe_series["timestamp"], sharpe_series["value"], strict=False)
    ]
    win_points = [
        {"timestamp": _timestamp_to_iso(ts), "value": float(value)}
        for ts, value in zip(win_series["timestamp"], win_series["value"], strict=False)
    ]
    return {
        "window_bars": summary["window_bars"],
        "periods_per_year": prepared.periods_per_year,
        "sharpe_last": summary["sharpe_last"],
        "sharpe_min": summary["sharpe_min"],
        "sharpe_max": summary["sharpe_max"],
        "win_rate_basis": "trades",
        "win_rate_unit": "pct",
        "win_rate_window_trades": summary["win_rate_window_trades"],
        "win_rate_last": summary["win_rate_last"],
        "win_rate_min": summary["win_rate_min"],
        "win_rate_max": summary["win_rate_max"],
        "sharpe_curve_points": sharpe_points,
        "win_rate_curve_points": win_points,
        "point_count_total": int(summary["point_count_total"]),
    }


def _prepare_backtest_data(result: Any) -> _PreparedBacktestData:
    if not isinstance(result, dict):
        return _PreparedBacktestData(
            trades=_empty_trades_frame(),
            equity=_empty_equity_frame(),
            returns=pd.Series(dtype=float),
            periods_per_year=_DEFAULT_PERIODS_PER_YEAR,
        )

    trades = _to_trades_frame(result.get("trades"))
    equity = _to_equity_frame(result.get("equity_curve"))
    returns = _to_returns_series(result.get("returns"), equity)
    periods = _infer_periods_per_year(returns.index)
    return _PreparedBacktestData(
        trades=trades,
        equity=equity,
        returns=returns,
        periods_per_year=periods,
    )


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": pd.Series(dtype="object"),
            "entry_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_price": pd.Series(dtype="float64"),
            "exit_price": pd.Series(dtype="float64"),
            "quantity": pd.Series(dtype="float64"),
            "bars_held": pd.Series(dtype="int64"),
            "exit_reason": pd.Series(dtype="object"),
            "pnl": pd.Series(dtype="float64"),
            "pnl_pct": pd.Series(dtype="float64"),
            "commission": pd.Series(dtype="float64"),
        }
    )


def _empty_equity_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "equity"])


def _to_trades_frame(raw: Any) -> pd.DataFrame:
    if not isinstance(raw, list) or not raw:
        return _empty_trades_frame()

    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry_time = _to_timestamp(item.get("entry_time"))
        exit_time = _to_timestamp(item.get("exit_time"))
        if entry_time is None or exit_time is None:
            continue
        rows.append(
            {
                "side": str(item.get("side", "")).strip().lower(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": _to_float(item.get("entry_price")),
                "exit_price": _to_float(item.get("exit_price")),
                "quantity": _to_float(item.get("quantity")),
                "bars_held": int(_to_float(item.get("bars_held")) or 0),
                "exit_reason": str(item.get("exit_reason", "")).strip() or "unknown",
                "pnl": _to_float(item.get("pnl")),
                "pnl_pct": _to_float(item.get("pnl_pct")),
                "commission": _to_float(item.get("commission")),
            }
        )

    if not rows:
        return _empty_trades_frame()
    frame = pd.DataFrame(rows)
    for column in ("entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "commission"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["bars_held"] = pd.to_numeric(frame["bars_held"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values("entry_time").reset_index(drop=True)
    return frame


def _to_equity_frame(raw: Any) -> pd.DataFrame:
    if not isinstance(raw, list) or not raw:
        return _empty_equity_frame()
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ts = _to_timestamp(item.get("timestamp"))
        equity = _to_float(item.get("equity"))
        if ts is None or equity is None:
            continue
        rows.append({"timestamp": ts, "equity": equity})
    if not rows:
        return _empty_equity_frame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return frame


def _to_returns_series(raw_returns: Any, equity: pd.DataFrame) -> pd.Series:
    if isinstance(raw_returns, list):
        clean = pd.to_numeric(pd.Series(raw_returns), errors="coerce").dropna().astype(float)
        if (
            not equity.empty
            and len(clean) == max(0, len(equity) - 1)
        ):
            index = pd.DatetimeIndex(equity["timestamp"].iloc[1:])
            return pd.Series(clean.values, index=index, dtype=float)
        if len(clean) > 0:
            return clean.astype(float)

    if equity.empty:
        return pd.Series(dtype=float)
    series = equity.set_index("timestamp")["equity"].astype(float)
    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(float)


def _infer_periods_per_year(index: Any) -> int:
    if not isinstance(index, pd.DatetimeIndex):
        return _DEFAULT_PERIODS_PER_YEAR
    if len(index) < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    unique_sorted = pd.DatetimeIndex(index.unique()).sort_values()
    if len(unique_sorted) < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    total_seconds = (unique_sorted[-1] - unique_sorted[0]).total_seconds()
    if total_seconds <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    periods = (len(unique_sorted) - 1) * (_SECONDS_PER_YEAR / total_seconds)
    if not np.isfinite(periods) or periods <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    return max(1, int(round(periods)))


def _to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _timestamp_to_iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _aggregate_trade_bucket(label: str, subset: pd.DataFrame, key_name: str) -> dict[str, Any]:
    if subset.empty:
        return {
            key_name: label,
            "trade_count": 0,
            "pnl_sum": 0.0,
            "avg_pnl": None,
            "avg_pnl_pct": None,
            "win_rate_pct": 0.0,
            "commission_sum": 0.0,
        }
    wins = int((subset["pnl"] > 0).sum())
    total = len(subset)
    return {
        key_name: label,
        "trade_count": total,
        "pnl_sum": float(subset["pnl"].sum()),
        "avg_pnl": _safe_float(subset["pnl"].mean()),
        "avg_pnl_pct": _safe_float(subset["pnl_pct"].mean()),
        "win_rate_pct": wins / total * 100.0,
        "commission_sum": float(subset["commission"].sum()),
    }


def _holding_label(bars: int) -> str:
    clean = max(0, int(bars))
    for upper, label in _HOLDING_BINS:
        if clean <= upper:
            return label
    return "gt_100"


def _drawdown_duration_stats(equity: pd.DataFrame) -> tuple[int, int, int]:
    if equity.empty:
        return (0, 0, 0)
    series = equity.set_index("timestamp")["equity"].astype(float)
    drawdown = series < series.cummax()

    max_duration = 0
    current_duration = 0
    max_recovery = 0
    in_drawdown = False
    segment_start = 0

    for idx, flag in enumerate(drawdown.tolist()):
        if flag:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
            if not in_drawdown:
                in_drawdown = True
                segment_start = idx
        else:
            if in_drawdown:
                max_recovery = max(max_recovery, idx - segment_start)
                in_drawdown = False
            current_duration = 0

    return (max_duration, max_recovery, current_duration)


def _normalize_sampling_mode(value: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in {"uniform", "eod", "auto"}:
        return normalized
    return "auto"


def _sample_end_of_day(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    frame = equity.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[timestamps.notna()].copy()
    if frame.empty:
        return _empty_equity_frame()
    frame["_day"] = timestamps.loc[frame.index].dt.normalize()
    sampled = frame.groupby("_day", sort=True).tail(1)
    sampled = sampled.drop(columns=["_day"]).sort_values("timestamp").reset_index(drop=True)
    return sampled


def _rolling_summary(
    *,
    returns: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
    window_bars: int,
) -> dict[str, Any]:
    if returns.empty and trades.empty:
        empty = pd.DataFrame(columns=["timestamp", "value"])
        return {
            "window_bars": 0,
            "win_rate_window_trades": 0,
            "sharpe_last": None,
            "sharpe_min": None,
            "sharpe_max": None,
            "win_rate_last": None,
            "win_rate_min": None,
            "win_rate_max": None,
            "sharpe_series": empty,
            "win_rate_series": empty,
            "point_count_total": 0,
        }

    if window_bars <= 1:
        auto_window = max(20, int(round(periods_per_year / 4)))
        window = min(len(returns), auto_window)
    else:
        window = min(len(returns), int(window_bars))
    if window < 2:
        window = min(len(returns), 2)

    # Rolling win rate should follow overall win rate semantics (trade-based),
    # not bar-based return wins, otherwise it is biased toward 0 during flat bars.
    trade_window = 0
    if len(trades) > 0:
        if window_bars <= 1:
            auto_trade_window = max(5, int(round(len(trades) / 4)))
            trade_window = min(len(trades), auto_trade_window)
        else:
            trade_window = min(len(trades), int(window_bars))
        if trade_window < 1:
            trade_window = min(len(trades), 1)

    if window < 2 and trade_window < 1:
        empty = pd.DataFrame(columns=["timestamp", "value"])
        return {
            "window_bars": window,
            "win_rate_window_trades": trade_window,
            "sharpe_last": None,
            "sharpe_min": None,
            "sharpe_max": None,
            "win_rate_last": None,
            "win_rate_min": None,
            "win_rate_max": None,
            "sharpe_series": empty,
            "win_rate_series": empty,
            "point_count_total": 0,
        }

    if window >= 2:
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std(ddof=0).replace(0.0, np.nan)
        roll_sharpe = np.sqrt(max(periods_per_year, 1)) * (roll_mean / roll_std)
        sharpe_series = (
            pd.DataFrame({"timestamp": roll_sharpe.index, "value": roll_sharpe.values})
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["value"])
            .reset_index(drop=True)
        )
    else:
        sharpe_series = pd.DataFrame(columns=["timestamp", "value"])

    if trade_window >= 1 and not trades.empty:
        trade_returns = (trades["pnl"] > 0).astype(float) * 100.0
        trade_roll = trade_returns.rolling(trade_window).mean()
        win_series = (
            pd.DataFrame({"timestamp": trades["exit_time"].values, "value": trade_roll.values})
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["value"])
            .reset_index(drop=True)
        )
    else:
        win_series = pd.DataFrame(columns=["timestamp", "value"])

    sharpe_last = _safe_float(sharpe_series["value"].iloc[-1]) if not sharpe_series.empty else None
    sharpe_min = _safe_float(sharpe_series["value"].min()) if not sharpe_series.empty else None
    sharpe_max = _safe_float(sharpe_series["value"].max()) if not sharpe_series.empty else None
    win_last = _safe_float(win_series["value"].iloc[-1]) if not win_series.empty else None
    win_min = _safe_float(win_series["value"].min()) if not win_series.empty else None
    win_max = _safe_float(win_series["value"].max()) if not win_series.empty else None

    return {
        "window_bars": window,
        "win_rate_window_trades": trade_window,
        "sharpe_last": sharpe_last,
        "sharpe_min": sharpe_min,
        "sharpe_max": sharpe_max,
        "win_rate_last": win_last,
        "win_rate_min": win_min,
        "win_rate_max": win_max,
        "sharpe_series": sharpe_series,
        "win_rate_series": win_series,
        "point_count_total": max(len(sharpe_series), len(win_series)),
    }


def _downsample_frame(frame: pd.DataFrame, *, max_points: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    cap = max(2, int(max_points))
    if len(frame) <= cap:
        return frame
    indices = np.linspace(0, len(frame) - 1, num=cap, dtype=int)
    return frame.iloc[indices].reset_index(drop=True)
