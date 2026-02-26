"""Black-swan stress runner across predefined/custom windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.domain.backtest.engine import EventDrivenBacktestEngine
from packages.domain.backtest.types import BacktestConfig
from packages.domain.market_data.data import DataLoader
from packages.domain.strategy import parse_strategy_payload
from packages.domain.stress.scenario_windows import ScenarioWindow


@dataclass(frozen=True, slots=True)
class BlackSwanWindowResult:
    """Result for one crisis window replay."""

    window_id: str
    window_label: str
    start: str
    end: str
    return_pct: float
    max_drawdown_pct: float
    win_rate: float
    trade_count: int
    sharpe: float
    pass_window: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BlackSwanAggregate:
    """Aggregate summary over all windows."""

    pass_fail: bool
    pass_count: int
    total_count: int
    aggregate_score: float


def run_black_swan_analysis(
    *,
    strategy_payload: dict[str, Any],
    windows: list[ScenarioWindow],
    min_return_pct: float = -20.0,
    max_drawdown_pct: float = 35.0,
    backtest_config: BacktestConfig | None = None,
) -> tuple[list[BlackSwanWindowResult], BlackSwanAggregate]:
    """Replay strategy on each stress window and compute pass/fail stats."""

    parsed = parse_strategy_payload(strategy_payload)
    symbol = parsed.universe.tickers[0]
    market = parsed.universe.market
    timeframe = parsed.universe.timeframe

    loader = DataLoader()
    config = backtest_config or BacktestConfig()

    results: list[BlackSwanWindowResult] = []
    for window in windows:
        try:
            frame = loader.load(
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                start_date=window.start,
                end_date=window.end,
            )
            run = EventDrivenBacktestEngine(strategy=parsed, data=frame, config=config).run()
            performance_metrics = run.performance.get("metrics", {}) if isinstance(run.performance, dict) else {}
            sharpe = float(performance_metrics.get("sharpe") or 0.0)
            return_pct = float(run.summary.total_return_pct)
            drawdown_pct = abs(float(run.summary.max_drawdown_pct))
            pass_window = return_pct >= float(min_return_pct) and drawdown_pct <= float(max_drawdown_pct)
            results.append(
                BlackSwanWindowResult(
                    window_id=window.window_id,
                    window_label=window.label,
                    start=window.start.isoformat(),
                    end=window.end.isoformat(),
                    return_pct=return_pct,
                    max_drawdown_pct=drawdown_pct,
                    win_rate=float(run.summary.win_rate),
                    trade_count=int(run.summary.total_trades),
                    sharpe=sharpe,
                    pass_window=pass_window,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                BlackSwanWindowResult(
                    window_id=window.window_id,
                    window_label=window.label,
                    start=window.start.isoformat(),
                    end=window.end.isoformat(),
                    return_pct=0.0,
                    max_drawdown_pct=100.0,
                    win_rate=0.0,
                    trade_count=0,
                    sharpe=0.0,
                    pass_window=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    pass_count = sum(1 for item in results if item.pass_window)
    total_count = len(results)
    pass_ratio = (pass_count / total_count) if total_count else 0.0

    return_values = [item.return_pct for item in results]
    dd_penalty = [max(0.0, item.max_drawdown_pct - max_drawdown_pct) for item in results]
    avg_return = (sum(return_values) / len(return_values)) if return_values else 0.0
    avg_penalty = (sum(dd_penalty) / len(dd_penalty)) if dd_penalty else 0.0
    aggregate_score = pass_ratio * 70.0 + max(-30.0, min(30.0, avg_return / 2.0 - avg_penalty / 2.0))

    aggregate = BlackSwanAggregate(
        pass_fail=pass_count == total_count and total_count > 0,
        pass_count=pass_count,
        total_count=total_count,
        aggregate_score=float(round(aggregate_score, 4)),
    )
    return results, aggregate


def serialize_window_result(item: BlackSwanWindowResult) -> dict[str, Any]:
    """Serialize one window result."""

    output = {
        "window_id": item.window_id,
        "window_label": item.window_label,
        "start": item.start,
        "end": item.end,
        "return_pct": item.return_pct,
        "max_drawdown_pct": item.max_drawdown_pct,
        "win_rate": item.win_rate,
        "trade_count": item.trade_count,
        "sharpe": item.sharpe,
        "pass": item.pass_window,
    }
    if item.error:
        output["error"] = item.error
    return output
