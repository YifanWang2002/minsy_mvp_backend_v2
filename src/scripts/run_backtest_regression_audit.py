"""Run multi-market long-window backtest regression and anomaly audit.

Usage:
    .venv/bin/python -m src.scripts.run_backtest_regression_audit
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.engine.backtest import BacktestConfig, BacktestEventType, EventDrivenBacktestEngine
from src.engine.data import DataLoader
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, parse_strategy_payload

MARKET_SYMBOLS: dict[str, list[str]] = {
    "crypto": ["BTCUSD", "ETHUSD"],
    "forex": ["EURUSD", "GBPUSD", "USDJPY"],
    "futures": ["ES", "NQ", "CL"],
    "us_stocks": ["AAPL", "SPY", "QQQ", "NVDA"],
}
TIMEFRAMES_MAIN: tuple[str, ...] = ("1m", "5m", "1h", "1d")
TIMEFRAMES_HOLD: tuple[str, ...] = ("1h",)
WINDOW_YEARS: dict[str, int] = {
    "1m": 2,
    "5m": 4,
    "1h": 12,
    "1d": 20,
}

BASE_CONFIG = BacktestConfig(
    initial_capital=100_000.0,
    commission_rate=0.0005,
    slippage_bps=2.0,
)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    strategy_kind: str
    market: str
    symbol: str
    timeframe: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    strategy_kind: str
    market: str
    symbol: str
    timeframe: str
    start: str
    end: str
    bars: int
    elapsed_s: float
    total_trades: int
    total_return_pct: float
    max_drawdown_pct: float
    final_equity: float
    win_rate: float
    open_events: int
    close_events: int
    anomaly_count: int
    anomalies: list[str]
    status: str
    error: str | None


def _select_window(timerange_start: str, timerange_end: str, timeframe: str) -> tuple[str, str]:
    start_ts = pd.Timestamp(timerange_start)
    end_ts = pd.Timestamp(timerange_end)
    years = WINDOW_YEARS.get(timeframe, 4)
    target_start = end_ts - pd.DateOffset(years=years)
    if target_start < start_ts:
        target_start = start_ts
    return (target_start.isoformat(), end_ts.isoformat())


def _build_example_payload(market: str, symbol: str, timeframe: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["strategy"]["name"] = f"example_{market}_{symbol}_{timeframe}"
    payload["universe"]["market"] = market
    payload["universe"]["tickers"] = [symbol]
    payload["timeframe"] = timeframe
    return payload


def _build_hold_to_end_payload(market: str, symbol: str, timeframe: str) -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {"name": f"hold_to_end_{market}_{symbol}_{timeframe}"},
        "universe": {"market": market, "tickers": [symbol]},
        "timeframe": timeframe,
        "factors": {
            "ema_2": {"type": "ema", "params": {"period": 2, "source": "close"}},
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cmp": {
                            "left": {"ref": "price.close"},
                            "op": "gt",
                            "right": 0,
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "never_exit_signal",
                        "order": {"type": "market"},
                        "condition": {
                            "cmp": {
                                "left": 0,
                                "op": "lt",
                                "right": 0,
                            }
                        },
                    }
                ],
            }
        },
    }


def _strategy_payload(case: CaseSpec) -> dict[str, Any]:
    if case.strategy_kind == "example":
        return _build_example_payload(case.market, case.symbol, case.timeframe)
    if case.strategy_kind == "hold_to_end":
        return _build_hold_to_end_payload(case.market, case.symbol, case.timeframe)
    raise ValueError(f"Unsupported strategy_kind: {case.strategy_kind}")


def _is_finite(value: Any) -> bool:
    try:
        return pd.notna(value) and pd.notnull(value) and float(value) == float(value)
    except Exception:  # noqa: BLE001
        return False


def _audit_anomalies(
    *,
    case: CaseSpec,
    bars: int,
    result,
) -> list[str]:
    anomalies: list[str] = []
    summary = result.summary

    if summary.max_drawdown_pct < 0 or summary.max_drawdown_pct > 100.0001:
        anomalies.append(f"summary.max_drawdown_pct_out_of_range={summary.max_drawdown_pct:.6f}")
    if summary.total_return_pct < -100.0001:
        anomalies.append(f"summary.total_return_pct_below_-100={summary.total_return_pct:.6f}")
    if not _is_finite(summary.final_equity) or summary.final_equity < 0:
        anomalies.append(f"summary.final_equity_invalid={summary.final_equity}")

    if case.timeframe in {"1m", "5m"} and bars >= 200_000 and summary.total_trades < 10:
        anomalies.append(
            "low_trade_frequency_intraday"
            f"(bars={bars}, trades={summary.total_trades})"
        )
    if abs(summary.total_return_pct) > 500 and summary.total_trades < 30:
        anomalies.append(
            "extreme_return_with_too_few_trades"
            f"(return={summary.total_return_pct:.2f}%, trades={summary.total_trades})"
        )

    open_count = sum(1 for event in result.events if event.type == BacktestEventType.POSITION_OPENED)
    close_count = sum(1 for event in result.events if event.type == BacktestEventType.POSITION_CLOSED)
    if open_count != close_count:
        anomalies.append(f"position_open_close_mismatch(open={open_count}, close={close_count})")

    metrics = result.performance.get("metrics", {}) if isinstance(result.performance, dict) else {}
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if value is None:
                continue
            if not _is_finite(value):
                anomalies.append(f"metric_non_finite({key}={value})")
        drawdown = metrics.get("max_drawdown")
        if drawdown is not None and _is_finite(drawdown):
            drawdown_value = float(drawdown)
            if drawdown_value > 0.0 or drawdown_value < -1.0001:
                anomalies.append(f"metric.max_drawdown_out_of_range={drawdown_value:.6f}")
        win_rate = metrics.get("win_rate")
        if win_rate is not None and _is_finite(win_rate):
            win_rate_value = float(win_rate)
            if win_rate_value < 0.0 or win_rate_value > 1.0:
                anomalies.append(f"metric.win_rate_out_of_range={win_rate_value:.6f}")
        sharpe = metrics.get("sharpe")
        if sharpe is not None and _is_finite(sharpe):
            sharpe_value = abs(float(sharpe))
            if sharpe_value > 10:
                anomalies.append(f"metric.sharpe_too_large={sharpe_value:.6f}")

    if case.strategy_kind == "hold_to_end":
        if summary.total_trades != 1:
            anomalies.append(f"hold_to_end_expected_one_trade(actual={summary.total_trades})")
        elif result.trades[0].exit_reason != "end_of_data":
            anomalies.append(f"hold_to_end_expected_end_of_data(actual={result.trades[0].exit_reason})")

    return anomalies


def _run_case(loader: DataLoader, case: CaseSpec) -> CaseResult:
    metadata = loader.get_symbol_metadata(case.market, case.symbol)
    if case.timeframe not in metadata["available_timeframes"]:
        return CaseResult(
            strategy_kind=case.strategy_kind,
            market=case.market,
            symbol=case.symbol,
            timeframe=case.timeframe,
            start="",
            end="",
            bars=0,
            elapsed_s=0.0,
            total_trades=0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            final_equity=0.0,
            win_rate=0.0,
            open_events=0,
            close_events=0,
            anomaly_count=1,
            anomalies=["timeframe_not_available"],
            status="skipped",
            error=None,
        )

    start, end = _select_window(
        metadata["available_timerange"]["start"],
        metadata["available_timerange"]["end"],
        case.timeframe,
    )
    t0 = perf_counter()
    frame = loader.load(
        market=case.market,
        symbol=case.symbol,
        timeframe=case.timeframe,
        start_date=start,
        end_date=end,
    )
    strategy = parse_strategy_payload(_strategy_payload(case))
    result = EventDrivenBacktestEngine(strategy=strategy, data=frame, config=BASE_CONFIG).run()
    elapsed = perf_counter() - t0

    open_count = sum(1 for event in result.events if event.type == BacktestEventType.POSITION_OPENED)
    close_count = sum(1 for event in result.events if event.type == BacktestEventType.POSITION_CLOSED)
    anomalies = _audit_anomalies(case=case, bars=len(frame), result=result)
    return CaseResult(
        strategy_kind=case.strategy_kind,
        market=case.market,
        symbol=case.symbol,
        timeframe=case.timeframe,
        start=start,
        end=end,
        bars=len(frame),
        elapsed_s=round(elapsed, 6),
        total_trades=result.summary.total_trades,
        total_return_pct=round(result.summary.total_return_pct, 6),
        max_drawdown_pct=round(result.summary.max_drawdown_pct, 6),
        final_equity=round(result.summary.final_equity, 6),
        win_rate=round(result.summary.win_rate, 6),
        open_events=open_count,
        close_events=close_count,
        anomaly_count=len(anomalies),
        anomalies=anomalies,
        status="ok",
        error=None,
    )


def _build_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for market, symbols in MARKET_SYMBOLS.items():
        for symbol in symbols:
            for timeframe in TIMEFRAMES_MAIN:
                cases.append(
                    CaseSpec(
                        strategy_kind="example",
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
            for timeframe in TIMEFRAMES_HOLD:
                cases.append(
                    CaseSpec(
                        strategy_kind="hold_to_end",
                        market=market,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
    return cases


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    ok_cases = [item for item in results if item.status == "ok"]
    failed_cases = [item for item in results if item.status == "failed"]
    skipped_cases = [item for item in results if item.status == "skipped"]
    anomalies = [item for item in ok_cases if item.anomaly_count > 0]

    return {
        "total_cases": len(results),
        "ok_cases": len(ok_cases),
        "failed_cases": len(failed_cases),
        "skipped_cases": len(skipped_cases),
        "cases_with_anomalies": len(anomalies),
        "total_anomalies": sum(item.anomaly_count for item in ok_cases),
        "elapsed_total_s": round(sum(item.elapsed_s for item in ok_cases), 3),
        "bars_total": int(sum(item.bars for item in ok_cases)),
        "avg_elapsed_s": round(
            sum(item.elapsed_s for item in ok_cases) / len(ok_cases), 3
        )
        if ok_cases
        else 0.0,
        "avg_bars": round(sum(item.bars for item in ok_cases) / len(ok_cases), 1)
        if ok_cases
        else 0.0,
        "anomaly_top": Counter(
            issue
            for item in anomalies
            for issue in item.anomalies
        ).most_common(20),
    }


def _to_report_markdown(summary: dict[str, Any], results: list[CaseResult]) -> str:
    lines: list[str] = []
    lines.append("# Backtest Regression Anomaly Audit")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now(UTC).isoformat()}")
    lines.append(f"- total_cases: {summary['total_cases']}")
    lines.append(f"- ok_cases: {summary['ok_cases']}")
    lines.append(f"- failed_cases: {summary['failed_cases']}")
    lines.append(f"- skipped_cases: {summary['skipped_cases']}")
    lines.append(f"- cases_with_anomalies: {summary['cases_with_anomalies']}")
    lines.append(f"- total_anomalies: {summary['total_anomalies']}")
    lines.append(f"- bars_total: {summary['bars_total']}")
    lines.append(f"- elapsed_total_s: {summary['elapsed_total_s']}")
    lines.append("")
    lines.append("## Top Anomalies")
    if summary["anomaly_top"]:
        for issue, count in summary["anomaly_top"]:
            lines.append(f"- {issue}: {count}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Cases With Anomalies")
    anomaly_cases = [item for item in results if item.status == "ok" and item.anomaly_count > 0]
    if anomaly_cases:
        for item in anomaly_cases:
            lines.append(
                f"- {item.strategy_kind} {item.market}/{item.symbol}/{item.timeframe} "
                f"bars={item.bars} trades={item.total_trades} return={item.total_return_pct:.4f}% "
                f"drawdown={item.max_drawdown_pct:.4f}% anomalies={item.anomalies}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Failed Cases")
    failed = [item for item in results if item.status == "failed"]
    if failed:
        for item in failed:
            lines.append(
                f"- {item.strategy_kind} {item.market}/{item.symbol}/{item.timeframe}: {item.error}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Long-Window Sanity Sample (top 20 by bars)")
    top_bars = sorted(
        [item for item in results if item.status == "ok"],
        key=lambda item: item.bars,
        reverse=True,
    )[:20]
    for item in top_bars:
        lines.append(
            f"- {item.strategy_kind} {item.market}/{item.symbol}/{item.timeframe} "
            f"{item.start} -> {item.end} bars={item.bars} trades={item.total_trades} "
            f"return={item.total_return_pct:.4f}% drawdown={item.max_drawdown_pct:.4f}% "
            f"elapsed={item.elapsed_s:.2f}s"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    loader = DataLoader("data")
    cases = _build_cases()
    print(f"[audit] cases={len(cases)}", flush=True)

    results: list[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        case_id = f"{case.strategy_kind}:{case.market}:{case.symbol}:{case.timeframe}"
        print(f"[audit] {idx}/{len(cases)} start {case_id}", flush=True)
        try:
            result = _run_case(loader, case)
        except Exception as exc:  # noqa: BLE001
            result = CaseResult(
                strategy_kind=case.strategy_kind,
                market=case.market,
                symbol=case.symbol,
                timeframe=case.timeframe,
                start="",
                end="",
                bars=0,
                elapsed_s=0.0,
                total_trades=0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                final_equity=0.0,
                win_rate=0.0,
                open_events=0,
                close_events=0,
                anomaly_count=1,
                anomalies=["execution_error"],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        print(
            f"[audit] {idx}/{len(cases)} done {case_id} "
            f"status={result.status} bars={result.bars} trades={result.total_trades} "
            f"return={result.total_return_pct:.4f}% dd={result.max_drawdown_pct:.4f}% "
            f"anomalies={result.anomaly_count} elapsed={result.elapsed_s:.2f}s"
        , flush=True)

    summary = _summarize(results)
    output = {
        "summary": summary,
        "results": [asdict(item) for item in results],
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_json = Path(f"/tmp/backtest_regression_audit_{timestamp}.json")
    out_md = Path(f"/tmp/backtest_regression_audit_{timestamp}.md")
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_report_markdown(summary, results), encoding="utf-8")

    latest_json = Path("/tmp/backtest_regression_audit_latest.json")
    latest_md = Path("/tmp/backtest_regression_audit_latest.md")
    latest_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md.write_text(_to_report_markdown(summary, results), encoding="utf-8")

    print(f"[audit] summary={json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[audit] report_json={out_json}", flush=True)
    print(f"[audit] report_md={out_md}", flush=True)
    print(f"[audit] latest_json={latest_json}", flush=True)
    print(f"[audit] latest_md={latest_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
