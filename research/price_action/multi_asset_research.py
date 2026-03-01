"""Phase 3: Multi-Asset Validation

Test validated factors across multiple assets (BTC, ETH).
Goal: Confirm factors generalize across different cryptocurrencies.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest import PriceActionBacktester, BacktestExperiment, FactorFilterConfig
from strategies import create_ema_crossover_strategy


def run_multi_asset_research():
    """Test validated factors across multiple assets."""

    backtester = PriceActionBacktester(data_dir="data")

    # Define baseline strategy (EMA crossover)
    baseline_strategy = create_ema_crossover_strategy()

    # Test on BTC and ETH
    assets = ["BTCUSD", "ETHUSD"]
    timeframe = "5m"  # Use 5m as primary timeframe

    # Test the best factors from Phase 1
    factors_config = {
        "TrendStrength": FactorFilterConfig(trend_strength_long_min=0.4),
        "TrendStructure": FactorFilterConfig(trend_structure_long_min=0.3),
        "RangeATR": FactorFilterConfig(range_atr_min=1.2),
        "MTFAlignment": FactorFilterConfig(mtf_alignment_long_min=0.5),
    }

    experiments = []

    for asset in assets:
        # Baseline for each asset
        experiments.append(
            BacktestExperiment(
                name=f"Baseline_{asset}_IS",
                description=f"EMA 20/50 baseline on {asset}",
                market="crypto",
                symbol=asset,
                timeframe=timeframe,
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=None,
            )
        )

        # Test each factor
        for factor_name, factor_config in factors_config.items():
            experiments.append(
                BacktestExperiment(
                    name=f"{factor_name}_{asset}_IS",
                    description=f"{factor_name} on {asset}",
                    market="crypto",
                    symbol=asset,
                    timeframe=timeframe,
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    strategy_dsl=baseline_strategy,
                    initial_capital=10000.0,
                    factor_filter=factor_config,
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
    print("PHASE 3: MULTI-ASSET VALIDATION")
    print("=" * 100)
    print()
    print(f"Testing top factors across {len(assets)} assets: {', '.join(assets)}")
    print(f"Timeframe: {timeframe}")
    print("In-Sample: 2023 | Out-of-Sample: 2024")
    print("=" * 100)
    print()

    print("[PHASE 3A] In-Sample Testing (2023)")
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
    print("[PHASE 3B] Out-of-Sample Validation (2024)")
    print("-" * 100)
    for exp in oos_experiments:
        print(f"Running: {exp.name}")
        try:
            result = backtester.run_experiment(exp)
            results[exp.name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[exp.name] = None

    # Analyze results by asset
    print()
    print("=" * 100)
    print("RESULTS BY ASSET")
    print("=" * 100)
    print()

    for asset in assets:
        print(f"\n{'=' * 100}")
        print(f"ASSET: {asset}")
        print('=' * 100)

        baseline_is = results.get(f"Baseline_{asset}_IS")
        baseline_oos = results.get(f"Baseline_{asset}_OOS")

        if baseline_is is None or baseline_oos is None:
            print(f"  ✗ No data available for {asset}")
            continue

        print(f"\nBaseline Performance:")
        print(f"  IS:  Return: {baseline_is['summary']['total_return']:.2f}%, "
              f"Trades: {baseline_is['summary']['total_trades']}, "
              f"Win Rate: {baseline_is['summary']['win_rate']:.2f}%")
        print(f"  OOS: Return: {baseline_oos['summary']['total_return']:.2f}%, "
              f"Trades: {baseline_oos['summary']['total_trades']}, "
              f"Win Rate: {baseline_oos['summary']['win_rate']:.2f}%")
        print()

        for factor_name in factors_config.keys():
            is_result = results.get(f"{factor_name}_{asset}_IS")
            oos_result = results.get(f"{factor_name}_{asset}_OOS")

            if is_result is None or oos_result is None:
                continue

            is_improvement = is_result['summary']['total_return'] - baseline_is['summary']['total_return']
            oos_improvement = oos_result['summary']['total_return'] - baseline_oos['summary']['total_return']

            status = "✓" if (is_improvement > 0 and oos_improvement > 0) else "✗"

            print(f"{status} {factor_name}:")
            print(f"  IS:  {is_result['summary']['total_return']:+.2f}% (change: {is_improvement:+.2f}%), "
                  f"Trades: {is_result['summary']['total_trades']}")
            print(f"  OOS: {oos_result['summary']['total_return']:+.2f}% (change: {oos_improvement:+.2f}%), "
                  f"Trades: {oos_result['summary']['total_trades']}")

    # Cross-asset summary
    print()
    print("=" * 100)
    print("CROSS-ASSET SUMMARY")
    print("=" * 100)
    print()

    for factor_name in factors_config.keys():
        print(f"\n{factor_name}:")
        print("-" * 100)
        validated_count = 0
        for asset in assets:
            is_result = results.get(f"{factor_name}_{asset}_IS")
            oos_result = results.get(f"{factor_name}_{asset}_OOS")
            baseline_is = results.get(f"Baseline_{asset}_IS")
            baseline_oos = results.get(f"Baseline_{asset}_OOS")

            if all([is_result, oos_result, baseline_is, baseline_oos]):
                is_imp = is_result['summary']['total_return'] - baseline_is['summary']['total_return']
                oos_imp = oos_result['summary']['total_return'] - baseline_oos['summary']['total_return']

                if is_imp > 0 and oos_imp > 0:
                    validated_count += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"  {status} {asset}: IS {is_imp:+.2f}%, OOS {oos_imp:+.2f}%")

        print(f"  Validated on {validated_count}/{len(assets)} assets")

    # Save results
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 3: Multi-Asset Validation",
        "assets": assets,
        "timeframe": timeframe,
        "results": {},
    }

    for name, result in results.items():
        if result is not None:
            results_data["results"][name] = result["summary"]

    output_file = output_dir / "multi_asset_phase3.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print()
    print("=" * 100)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    run_multi_asset_research()
