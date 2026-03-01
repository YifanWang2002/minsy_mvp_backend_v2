"""Phase 4: Factor Combination Optimization

Test combinations of validated factors to find optimal filtering strategies.
Goal: Identify synergistic factor combinations that outperform single factors.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backtest import PriceActionBacktester, BacktestExperiment, FactorFilterConfig
from strategies import create_ema_crossover_strategy


def run_factor_combination_research():
    """Test combinations of validated factors."""

    backtester = PriceActionBacktester(data_dir="data")

    # Define baseline strategy (EMA crossover)
    baseline_strategy = create_ema_crossover_strategy()

    # Define factor combinations to test
    combinations = {
        "Baseline": None,

        # Single factors (best performers from previous phases)
        "TrendStrength": FactorFilterConfig(
            trend_strength_long_min=0.4,
        ),
        "TrendStructure": FactorFilterConfig(
            trend_structure_long_min=0.3,
        ),
        "RangeATR": FactorFilterConfig(
            range_atr_min=1.2,
        ),
        "MTFAlignment": FactorFilterConfig(
            mtf_alignment_long_min=0.5,
        ),

        # Two-factor combinations
        "Trend+Range": FactorFilterConfig(
            trend_strength_long_min=0.4,
            range_atr_min=1.2,
        ),
        "Trend+MTF": FactorFilterConfig(
            trend_strength_long_min=0.4,
            mtf_alignment_long_min=0.5,
        ),
        "Structure+Range": FactorFilterConfig(
            trend_structure_long_min=0.3,
            range_atr_min=1.2,
        ),
        "Structure+MTF": FactorFilterConfig(
            trend_structure_long_min=0.3,
            mtf_alignment_long_min=0.5,
        ),

        # Three-factor combinations
        "Trend+Structure+Range": FactorFilterConfig(
            trend_strength_long_min=0.4,
            trend_structure_long_min=0.3,
            range_atr_min=1.2,
        ),
        "Trend+Structure+MTF": FactorFilterConfig(
            trend_strength_long_min=0.4,
            trend_structure_long_min=0.3,
            mtf_alignment_long_min=0.5,
        ),
        "Trend+Range+MTF": FactorFilterConfig(
            trend_strength_long_min=0.4,
            range_atr_min=1.2,
            mtf_alignment_long_min=0.5,
        ),

        # Four-factor combination (aggressive filtering)
        "All_Four": FactorFilterConfig(
            trend_strength_long_min=0.4,
            trend_structure_long_min=0.3,
            range_atr_min=1.2,
            mtf_alignment_long_min=0.5,
        ),

        # Relaxed combinations (lower thresholds for more trades)
        "Trend+Range_Relaxed": FactorFilterConfig(
            trend_strength_long_min=0.3,
            range_atr_min=1.0,
        ),
        "Trend+MTF_Relaxed": FactorFilterConfig(
            trend_strength_long_min=0.3,
            mtf_alignment_long_min=0.3,
        ),
    }

    experiments = []

    for combo_name, factor_config in combinations.items():
        # In-sample
        experiments.append(
            BacktestExperiment(
                name=f"{combo_name}_IS",
                description=f"{combo_name} combination",
                market="crypto",
                symbol="BTCUSD",
                timeframe="5m",
                start_date="2023-01-01",
                end_date="2023-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=factor_config,
            )
        )

        # Out-of-sample
        experiments.append(
            BacktestExperiment(
                name=f"{combo_name}_OOS",
                description=f"{combo_name} combination",
                market="crypto",
                symbol="BTCUSD",
                timeframe="5m",
                start_date="2024-01-01",
                end_date="2024-12-31",
                strategy_dsl=baseline_strategy,
                initial_capital=10000.0,
                factor_filter=factor_config,
            )
        )

    # Run experiments
    print("=" * 100)
    print("PHASE 4: FACTOR COMBINATION OPTIMIZATION")
    print("=" * 100)
    print()
    print(f"Testing {len(combinations)} factor combinations on BTCUSD 5m")
    print("In-Sample: 2023 | Out-of-Sample: 2024")
    print("=" * 100)
    print()

    results = {}
    for exp in experiments:
        print(f"Running: {exp.name}")
        try:
            result = backtester.run_experiment(exp)
            results[exp.name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[exp.name] = None

    # Analyze results
    print()
    print("=" * 100)
    print("COMBINATION PERFORMANCE RANKING")
    print("=" * 100)
    print()

    baseline_is = results["Baseline_IS"]
    baseline_oos = results["Baseline_OOS"]

    print("BASELINE:")
    print(f"  IS:  Return: {baseline_is['summary']['total_return']:.2f}%, "
          f"Trades: {baseline_is['summary']['total_trades']}, "
          f"Win Rate: {baseline_is['summary']['win_rate']:.2f}%")
    print(f"  OOS: Return: {baseline_oos['summary']['total_return']:.2f}%, "
          f"Trades: {baseline_oos['summary']['total_trades']}, "
          f"Win Rate: {baseline_oos['summary']['win_rate']:.2f}%")
    print()

    # Calculate improvements and rank
    combo_results = []
    for combo_name in combinations.keys():
        if combo_name == "Baseline":
            continue

        is_result = results.get(f"{combo_name}_IS")
        oos_result = results.get(f"{combo_name}_OOS")

        if is_result is None or oos_result is None:
            continue

        is_improvement = is_result['summary']['total_return'] - baseline_is['summary']['total_return']
        oos_improvement = oos_result['summary']['total_return'] - baseline_oos['summary']['total_return']

        combo_results.append({
            "name": combo_name,
            "is_return": is_result['summary']['total_return'],
            "oos_return": oos_result['summary']['total_return'],
            "is_improvement": is_improvement,
            "oos_improvement": oos_improvement,
            "is_trades": is_result['summary']['total_trades'],
            "oos_trades": oos_result['summary']['total_trades'],
            "is_win_rate": is_result['summary']['win_rate'],
            "oos_win_rate": oos_result['summary']['win_rate'],
            "is_expectancy": is_result['summary']['expectancy'],
            "oos_expectancy": oos_result['summary']['expectancy'],
        })

    # Sort by OOS improvement
    combo_results.sort(key=lambda x: x["oos_improvement"], reverse=True)

    print("=" * 100)
    print("RANKED BY OUT-OF-SAMPLE IMPROVEMENT")
    print("=" * 100)
    print()

    for i, combo in enumerate(combo_results, 1):
        status = "✓" if (combo["is_improvement"] > 0 and combo["oos_improvement"] > 0) else "✗"

        print(f"{i}. {status} {combo['name']}")
        print(f"   IS:  {combo['is_return']:+.2f}% (Δ{combo['is_improvement']:+.2f}%), "
              f"Trades: {combo['is_trades']}, WR: {combo['is_win_rate']:.2f}%, "
              f"Exp: ${combo['is_expectancy']:.2f}")
        print(f"   OOS: {combo['oos_return']:+.2f}% (Δ{combo['oos_improvement']:+.2f}%), "
              f"Trades: {combo['oos_trades']}, WR: {combo['oos_win_rate']:.2f}%, "
              f"Exp: ${combo['oos_expectancy']:.2f}")
        print()

    # Key findings
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()

    validated_combos = [c for c in combo_results if c["is_improvement"] > 0 and c["oos_improvement"] > 0]

    if validated_combos:
        print(f"✓ {len(validated_combos)}/{len(combo_results)} combinations validated (positive IS & OOS)")
        print()

        best_combo = validated_combos[0]
        print(f"✓ BEST COMBINATION: {best_combo['name']}")
        print(f"  IS Improvement:  {best_combo['is_improvement']:+.2f}%")
        print(f"  OOS Improvement: {best_combo['oos_improvement']:+.2f}%")
        print(f"  Trade Reduction: {baseline_is['summary']['total_trades']} → {best_combo['is_trades']} IS, "
              f"{baseline_oos['summary']['total_trades']} → {best_combo['oos_trades']} OOS")
        print()

        # Find best single factor
        single_factors = [c for c in validated_combos if c["name"] in ["TrendStrength", "TrendStructure", "RangeATR", "MTFAlignment"]]
        if single_factors:
            best_single = single_factors[0]
            print(f"✓ BEST SINGLE FACTOR: {best_single['name']}")
            print(f"  OOS Improvement: {best_single['oos_improvement']:+.2f}%")
            print()

        # Find best two-factor combo
        two_factor_combos = [c for c in validated_combos if "+" in c["name"] and c["name"].count("+") == 1 and "Relaxed" not in c["name"]]
        if two_factor_combos:
            best_two = two_factor_combos[0]
            print(f"✓ BEST TWO-FACTOR COMBO: {best_two['name']}")
            print(f"  OOS Improvement: {best_two['oos_improvement']:+.2f}%")
            print()
    else:
        print("✗ No combinations passed validation")

    # Save results
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 4: Factor Combination Optimization",
        "baseline": {
            "is": baseline_is["summary"],
            "oos": baseline_oos["summary"],
        },
        "combinations": combo_results,
    }

    output_file = output_dir / "factor_combinations_phase4.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print()
    print("=" * 100)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    run_factor_combination_research()
