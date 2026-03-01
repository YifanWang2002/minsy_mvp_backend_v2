"""Phase 1: Advanced Factors Research - Single Factor Testing

Test 7 new advanced price action factors individually on BTC 5m data.
Goal: Identify factors with >5% IS improvement and OOS validation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest import PriceActionBacktester, BacktestExperiment, FactorFilterConfig
from strategies import create_ema_crossover_strategy


def run_advanced_factors_research():
    """Test 7 new advanced factors individually."""

    backtester = PriceActionBacktester(data_dir="data")

    # Define baseline strategy (EMA crossover)
    baseline_strategy = create_ema_crossover_strategy()

    # Define experiments with new advanced factors
    experiments = [
        # Baseline
        BacktestExperiment(
            name="Baseline_IS",
            description="EMA 20/50 crossover baseline",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=None,
        ),

        # 1. Tail Ratio - favor bullish rejection (lower wick > upper wick)
        BacktestExperiment(
            name="TailRatio_IS",
            description="Tail ratio < -0.2 (bullish rejection)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                tail_ratio_long_max=-0.2,  # Lower wick dominates (bullish rejection)
            ),
        ),

        # 2. Gap Quality - sustained bullish gaps
        BacktestExperiment(
            name="GapQuality_IS",
            description="Gap quality > 0.3 (sustained gap)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                gap_quality_long_min=0.3,  # Positive sustained gap
            ),
        ),

        # 3. False Breakout - bullish false breakout signal
        BacktestExperiment(
            name="FalseBreakout_IS",
            description="False breakout < -0.5 (bullish signal)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                false_breakout_long_max=-0.5,  # Bullish false breakout signal
            ),
        ),

        # 4. Volatility Percentile - high volatility environment
        BacktestExperiment(
            name="VolatilityPct_IS",
            description="Volatility > 60th percentile",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                volatility_pct_min=0.6,  # Top 40% volatility
            ),
        ),

        # 5. Volume Divergence - bullish divergence
        BacktestExperiment(
            name="VolumeDivergence_IS",
            description="Volume divergence > 0.2 (bullish)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                volume_divergence_long_min=0.2,  # Bullish divergence
            ),
        ),

        # 6. Follow Through - strong continuation
        BacktestExperiment(
            name="FollowThrough_IS",
            description="Follow through > 0.6 (strong)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                follow_through_min=0.6,  # Strong follow-through
            ),
        ),

        # 7. Multi-Timeframe Alignment - bullish alignment
        BacktestExperiment(
            name="MTFAlignment_IS",
            description="MTF alignment > 0.5 (bullish)",
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_dsl=baseline_strategy,
            initial_capital=10000.0,
            factor_filter=FactorFilterConfig(
                mtf_alignment_long_min=0.5,  # Strong bullish alignment
            ),
        ),
    ]

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
    print("ADVANCED FACTORS RESEARCH - PHASE 1: SINGLE FACTOR TESTING")
    print("=" * 100)
    print()
    print("Testing 7 new advanced price action factors on BTC/USDT 5m")
    print("In-Sample: 2023 | Out-of-Sample: 2024")
    print("=" * 100)
    print()

    print("[PHASE 1A] In-Sample Testing (2023)")
    print("-" * 100)
    results = {}
    for exp in experiments:
        print(f"Running: {exp.name}")
        result = backtester.run_experiment(exp)
        results[exp.name] = result

    print()
    print("[PHASE 1B] Out-of-Sample Validation (2024)")
    print("-" * 100)
    for exp in oos_experiments:
        print(f"Running: {exp.name}")
        result = backtester.run_experiment(exp)
        results[exp.name] = result

    # Analyze results
    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()

    baseline_is = results["Baseline_IS"]
    baseline_oos = results["Baseline_OOS"]

    print("BASELINE PERFORMANCE:")
    print(f"  In-Sample (2023):")
    print(f"    Trades: {baseline_is['summary']['total_trades']}, "
          f"Win Rate: {baseline_is['summary']['win_rate']:.2f}%, "
          f"Return: {baseline_is['summary']['total_return']:.2f}%")
    print(f"  Out-of-Sample (2024):")
    print(f"    Trades: {baseline_oos['summary']['total_trades']}, "
          f"Win Rate: {baseline_oos['summary']['win_rate']:.2f}%, "
          f"Return: {baseline_oos['summary']['total_return']:.2f}%")
    print()

    print("=" * 100)
    print("FACTOR PERFORMANCE")
    print("=" * 100)
    print()

    factor_configs = {
        "TailRatio": "Tail Ratio < -0.2 (bullish rejection)",
        "GapQuality": "Gap Quality > 0.3 (sustained gap)",
        "FalseBreakout": "False Breakout < -0.5 (bullish signal)",
        "VolatilityPct": "Volatility > 60th percentile",
        "VolumeDivergence": "Volume Divergence > 0.2 (bullish)",
        "FollowThrough": "Follow Through > 0.6 (strong)",
        "MTFAlignment": "MTF Alignment > 0.5 (bullish)",
    }

    validated_factors = []

    for factor_name, description in factor_configs.items():
        is_result = results[f"{factor_name}_IS"]
        oos_result = results[f"{factor_name}_OOS"]

        is_return = is_result["summary"]["total_return"]
        oos_return = oos_result["summary"]["total_return"]
        is_improvement = is_return - baseline_is["summary"]["total_return"]
        oos_improvement = oos_return - baseline_oos["summary"]["total_return"]

        print(f"{factor_name}: {description}")
        print("-" * 100)
        print(f"  In-Sample (2023):")
        print(f"    Return: {is_return:.2f}% (baseline: {baseline_is['summary']['total_return']:.2f}%, change: {is_improvement:+.2f}%)")
        print(f"    Win Rate: {is_result['summary']['win_rate']:.2f}% (baseline: {baseline_is['summary']['win_rate']:.2f}%, change: {is_result['summary']['win_rate'] - baseline_is['summary']['win_rate']:+.2f}%)")
        print(f"    Trades: {is_result['summary']['total_trades']} (baseline: {baseline_is['summary']['total_trades']}, change: {is_result['summary']['total_trades'] - baseline_is['summary']['total_trades']:+d})")
        print(f"    Expectancy: ${is_result['summary']['expectancy']:.2f} (baseline: ${baseline_is['summary']['expectancy']:.2f}, change: ${is_result['summary']['expectancy'] - baseline_is['summary']['expectancy']:+.2f})")

        print(f"  Out-of-Sample (2024):")
        print(f"    Return: {oos_return:.2f}% (baseline: {baseline_oos['summary']['total_return']:.2f}%, change: {oos_improvement:+.2f}%)")
        print(f"    Win Rate: {oos_result['summary']['win_rate']:.2f}% (baseline: {baseline_oos['summary']['win_rate']:.2f}%, change: {oos_result['summary']['win_rate'] - baseline_oos['summary']['win_rate']:+.2f}%)")
        print(f"    Trades: {oos_result['summary']['total_trades']} (baseline: {baseline_oos['summary']['total_trades']}, change: {oos_result['summary']['total_trades'] - baseline_oos['summary']['total_trades']:+d})")
        print(f"    Expectancy: ${oos_result['summary']['expectancy']:.2f} (baseline: ${baseline_oos['summary']['expectancy']:.2f}, change: ${oos_result['summary']['expectancy'] - baseline_oos['summary']['expectancy']:+.2f})")

        # Check validation
        if is_improvement > 5.0 and oos_improvement > 0:
            print(f"  ✓ VALIDATED: IS improvement >5% and OOS positive")
            validated_factors.append((factor_name, is_improvement, oos_improvement))
        elif oos_improvement > 0:
            print(f"  ⚠ PARTIAL: OOS improvement but IS <5%")
        else:
            print(f"  ✗ FAILED: No OOS improvement")
        print()

    # Summary
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()

    if validated_factors:
        validated_factors.sort(key=lambda x: x[2], reverse=True)  # Sort by OOS improvement
        print(f"✓ VALIDATED FACTORS: {len(validated_factors)}")
        for factor_name, is_imp, oos_imp in validated_factors:
            description = factor_configs[factor_name]
            print(f"  - {factor_name}: {description}")
            print(f"    IS: {is_imp:+.2f}%, OOS: {oos_imp:+.2f}%")
        print()

        best_factor, best_is, best_oos = validated_factors[0]
        print(f"✓ BEST VALIDATED FACTOR: {best_factor}")
        print(f"  Description: {factor_configs[best_factor]}")
        print(f"  In-Sample Improvement: {best_is:+.2f}%")
        print(f"  Out-of-Sample Improvement: {best_oos:+.2f}%")
    else:
        print("✗ No factors passed validation criteria (IS >5% and OOS positive)")

    print()
    print("=" * 100)
    print()

    # Save results
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 1: Advanced Factors Single Testing",
        "baseline": {
            "is": baseline_is["summary"],
            "oos": baseline_oos["summary"],
        },
        "factors": {},
    }

    for factor_name in factor_configs.keys():
        is_result = results[f"{factor_name}_IS"]
        oos_result = results[f"{factor_name}_OOS"]
        results_data["factors"][factor_name] = {
            "description": factor_configs[factor_name],
            "is": is_result["summary"],
            "oos": oos_result["summary"],
        }

    output_file = output_dir / "advanced_factors_phase1.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"✓ Results saved to: {output_file}")

    return validated_factors


if __name__ == "__main__":
    run_advanced_factors_research()
