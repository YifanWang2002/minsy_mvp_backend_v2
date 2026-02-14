"""Benchmark backtest pipeline stage timings on a fixed dataset/case.

Usage:
    .venv/bin/python -m src.scripts.benchmark_backtest_stages --tag baseline
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from src.engine.backtest import BacktestConfig, EventDrivenBacktestEngine
import src.engine.backtest.engine as engine_mod
from src.engine.backtest.factors import prepare_backtest_frame
from src.engine.data import DataLoader
from src.engine.performance import build_quantstats_performance
from src.engine.strategy import (
    EXAMPLE_PATH,
    load_strategy_payload,
    parse_strategy_payload,
    validate_strategy_payload,
)


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    tag: str
    case: str
    bars: int
    metadata_repeats: int
    metadata_total_s: float
    metadata_avg_s: float
    data_load_s: float
    dsl_file_load_s: float
    dsl_validate_s: float
    dsl_parse_s: float
    factor_calc_s: float
    backtest_core_s: float
    quantstats_s: float
    full_run_s: float
    full_total_s: float
    trades: int
    events: int


def _run_core_without_quantstats(
    *,
    strategy,
    prepared_frame,
    config: BacktestConfig,
):
    original = engine_mod.build_quantstats_performance
    engine_mod.build_quantstats_performance = lambda **_: {}
    try:
        engine = EventDrivenBacktestEngine.__new__(EventDrivenBacktestEngine)
        engine.strategy = strategy
        engine.config = config
        engine.frame = prepared_frame
        engine._events = []
        engine._trades = []
        engine._equity_curve = []
        engine._cash = float(config.initial_capital)
        engine._position = None
        engine._timestamps = engine._build_timestamp_cache()
        engine._compiled_side_rules = engine._build_compiled_side_rules()
        return EventDrivenBacktestEngine.run(engine)
    finally:
        engine_mod.build_quantstats_performance = original


def benchmark_once(
    *,
    tag: str,
    market: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    metadata_repeats: int,
) -> BenchmarkRow:
    case = f"{market}:{symbol}:{timeframe}:{start[:10]}->{end[:10]}"
    loader = DataLoader("data")
    config = BacktestConfig(
        initial_capital=100_000.0,
        commission_rate=0.0005,
        slippage_bps=2.0,
    )

    t_meta0 = perf_counter()
    for _ in range(metadata_repeats):
        loader.get_symbol_metadata(market, symbol)
    t_meta1 = perf_counter()

    t0 = perf_counter()
    frame = loader.load(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start,
        end_date=end,
    )
    t1 = perf_counter()

    t2 = perf_counter()
    payload = load_strategy_payload(EXAMPLE_PATH)
    t3 = perf_counter()

    payload = deepcopy(payload)
    payload["universe"]["market"] = market
    payload["universe"]["tickers"] = [symbol]
    payload["timeframe"] = timeframe

    t4 = perf_counter()
    validation = validate_strategy_payload(payload)
    t5 = perf_counter()
    if not validation.is_valid:
        raise RuntimeError(f"Validation failed: {validation.errors}")

    t6 = perf_counter()
    strategy = parse_strategy_payload(payload)
    t7 = perf_counter()

    t8 = perf_counter()
    prepared = prepare_backtest_frame(frame, strategy=strategy)
    t9 = perf_counter()

    t10 = perf_counter()
    core_result = _run_core_without_quantstats(
        strategy=strategy,
        prepared_frame=prepared,
        config=config,
    )
    t11 = perf_counter()

    t12 = perf_counter()
    _ = build_quantstats_performance(
        returns=list(core_result.returns),
        timestamps=[point.timestamp for point in core_result.equity_curve[1:]],
    )
    t13 = perf_counter()

    t14 = perf_counter()
    full_result = EventDrivenBacktestEngine(
        strategy=strategy,
        data=prepared,
        config=config,
    ).run()
    t15 = perf_counter()

    total = (t_meta1 - t_meta0) + (t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6) + (t9 - t8) + (t11 - t10) + (t13 - t12) + (t15 - t14)
    metadata_total = t_meta1 - t_meta0

    return BenchmarkRow(
        tag=tag,
        case=case,
        bars=len(frame),
        metadata_repeats=metadata_repeats,
        metadata_total_s=round(metadata_total, 6),
        metadata_avg_s=round(metadata_total / max(metadata_repeats, 1), 6),
        data_load_s=round(t1 - t0, 6),
        dsl_file_load_s=round(t3 - t2, 6),
        dsl_validate_s=round(t5 - t4, 6),
        dsl_parse_s=round(t7 - t6, 6),
        factor_calc_s=round(t9 - t8, 6),
        backtest_core_s=round(t11 - t10, 6),
        quantstats_s=round(t13 - t12, 6),
        full_run_s=round(t15 - t14, 6),
        full_total_s=round(total, 6),
        trades=full_result.summary.total_trades,
        events=len(full_result.events),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark backtest stage timings.")
    parser.add_argument("--tag", default="run", help="Result tag, e.g. baseline/step1.")
    parser.add_argument("--market", default="crypto")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--start", default="2024-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--metadata-repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        default="/tmp/backtest_stage_bench.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row = benchmark_once(
        tag=args.tag,
        market=args.market,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        metadata_repeats=max(args.metadata_repeats, 1),
    )
    output = {
        "row": asdict(row),
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
