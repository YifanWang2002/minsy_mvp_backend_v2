"""Phase 2: Multi-Timeframe Validation

Test validated factors across multiple timeframes (5min, 15min, 1h, 4h, daily).
Goal: Confirm factors work across different time horizons.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest import PriceActionBacktester, BacktestExperiment, FactorFilterConfig
from strategies import create_ema_crossover_strategy


def run_multi_timeframe_research():
    """Test validated factors across multiple timeframes."""

    backtester = PriceActionBacktester(data_dir="data")

    # Define baseline strategy (EMA crossover)
    baseline_strategy = create_ema_crossover_strategy()

    # Test the best factors from Phase 1 across multiple timeframes
    # Based on previous research, these are the top performers:
    # 1. Trend Strength
    # 2. Trend Structure
    # 3. Range ATR

    timeframes = ["5m", "15m", "1h"]  # Start with these, add 4h/daily if data available

    experiments = []

    for tf in timeframes:
        # Baseline for each timeframe
        experiments.append(
            BacktestExperiment(
                name=f"Baseline_{tf}_IS",
                description=f"EMA 20/50 baseline on {tf}",
                market="crypto",
                symbol="BTCUSD",
                timeframe=tf,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=None,
            )
        )

        # Trend Strength filter
        experiments.append(
            BacktestExperiment(
                name=f"TrendStrength_{tf}_IS",
                description=f"Trend strength > 0.4 on {tf}",
                market="crypto",
                symbol="BTCUSD",
                timeframe=tf,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=FactorFilterConfig(
                    trend_strength_long_min=0.4,
                ),
            )
        )

        # Trend Structure filter
        experiments.append(
            BacktestExperiment(
                name=f"TrendStructure_{tf}_IS",
                description=f"Trend structure > 0.3 on {tf}",
                market="crypto",
                symbol="BTCUSD",
                timeframe=tf,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=FactorFilterConfig(
                    trend_structure_long_min=0.3,
                ),
            )
        )

        # Range ATR filter
        experiments.append(
            BacktestExperiment(
                name=f"RangeATR_{tf}_IS",
                description=f"Range > 1.2x ATR on {tf}",
                market="crypto",
                symbol="BTCUSD",
                timeframe=tf,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=FactorFilterConfig(
                    range_atr_min=1.2,
                ),
            )
        )

        # MTF Alignment (new factor)
        experiments.append(
            BacktestExperiment(
                name=f"MTFAlignment_{tf}_IS",
                description=f"MTF alignment > 0.5 on {tf}",
                market="crypto",
                symbol="BTCUSD",
                timeframe=tf,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=FactorFilterConfig(
                    mtf_alignment_long_min=0.5,
                ),
            )
        )

    # Add OOS experiments
    oos_experiments = []
    for exp in experiments:
        oos_exp = BacktestExperiment(
            name=exp.name.replace("_IS", "_OOS"),
            description=exp.description,
            market=exp.market,
            symbol=exp.symbol,
            timeframe=exp.timeframe,
            start_date="2024-01-01",
            end_date="2024-12-31",
            strategy_dsl=exp.strategy_dsl,
            initial_capital=exp.initial_capital,
            factor_filter=exp.factor_filter,
        )
        oos_experiments.append(oos_exp)

    all_experiments = experiments + oos_experiments

    # Run experiments
    print("=" * 100)
    print("PHASE 2: MULTI-TIMEFRAME VALIDATION")
    print("=" * 100)
    print()
    print(f"Testing top factors across {len(timeframes)} timeframes: {', '.join(timeframes)}")
    print("In-Sample: 2023 | Out-of-Sample: 2024")
    print("=" * 100)
    print()

    print("[PHASE 2A] In-Sample Testing (2023)")
    print("-" * 100)
    results = {}
    for exp in experiments:
        print(f"Running: {exp.name}")
        try:
            result = backtester.run_experiment(exp)
            results[exp.name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[exp.name] = None

    print()
    print("[PHASE 2B] Out-of-Sample Validation (2024)")
    print("-" * 100)
    for exp in oos_experiments:
        print(f"Running: {exp.name}")
        try:
            result = backtester.run_experiment(exp)
            results[exp.name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[exp.name] = None

    # Analyze results by timeframe
    print()
    print("=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)
    print()

    factors = ["TrendStrength", "TrendStructure", "RangeATR", "MTFAlignment"]

    for tf in timeframes:
        print(f"\n{'=' * 100}")
        print(f"TIMEFRAME: {tf}")
        print('=' * 100)

        baseline_is = results.get(f"Baseline_{tf}_IS")
        baseline_oos = results.get(f"Baseline_{tf}_OOS")

        if baseline_is is None or baseline_oos is None:
            print(f"  ✗ No data available for {tf}")
            continue

        print(f"\nBaseline Performance:")
        print(f"  IS:  Return: {baseline_is['summary']['total_return']:.2f}%, "
              f"Trades: {baseline_is['summary']['total_trades']}, "
              f"Win Rate: {baseline_is['summary']['win_rate']:.2f}%")
        print(f"  OOS: Return: {baseline_oos['summary']['total_return']:.2f}%, "
              f"Trades: {baseline_oos['summary']['total_trades']}, "
              f"Win Rate: {baseline_oos['summary']['win_rate']:.2f}%")
        print()

        for factor in factors:
            is_result = results.get(f"{factor}_{tf}_IS")
            oos_result = results.get(f"{factor}_{tf}_OOS")

            if is_result is None or oos_result is None:
                continue

            is_improvement = is_result['summary']['total_return'] - baseline_is['summary']['total_return']
            oos_improvement = oos_result['summary']['total_return'] - baseline_oos['summary']['total_return']

            status = "✓" if (is_improvement > 0 and oos_improvement > 0) else "✗"

            print(f"{status} {factor}:")
            print(f"  IS:  {is_result['summary']['total_return']:+.2f}% (change: {is_improvement:+.2f}%), "
                  f"Trades: {is_result['summary']['total_trades']}")
            print(f"  OOS: {oos_result['summary']['total_return']:+.2f}% (change: {oos_improvement:+.2f}%), "
                  f"Trades: {oos_result['summary']['total_trades']}")

    # Cross-timeframe summary
    print()
    print("=" * 100)
    print("CROSS-TIMEFRAME SUMMARY")
    print("=" * 100)
    print()

    for factor in factors:
        print(f"\n{factor}:")
        print("-" * 100)
        validated_count = 0
        for tf in timeframes:
            is_result = results.get(f"{factor}_{tf}_IS")
            oos_result = results.get(f"{factor}_{tf}_OOS")
            baseline_is = results.get(f"Baseline_{tf}_IS")
            baseline_oos = results.get(f"Baseline_{tf}_OOS")

            if all([is_result, oos_result, baseline_is, baseline_oos]):
                is_imp = is_result['summary']['total_return'] - baseline_is['summary']['total_return']
                oos_imp = oos_result['summary']['total_return'] - baseline_oos['summary']['total_return']

                if is_imp > 0 and oos_imp > 0:
                    validated_count += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"  {status} {tf}: IS {is_imp:+.2f}%, OOS {oos_imp:+.2f}%")

        print(f"  Validated on {validated_count}/{len(timeframes)} timeframes")

    # Save results
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 2: Multi-Timeframe Validation",
        "timeframes": timeframes,
        "results": {},
    }

    for name, result in results.items():
        if result is not None:
            results_data["results"][name] = result["summary"]

    output_file = output_dir / "multi_timeframe_phase2.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print()
    print("=" * 100)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    run_multi_timeframe_research()
