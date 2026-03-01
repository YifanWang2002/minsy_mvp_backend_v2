"""Focused research on EMA crossover strategy with price action factors.

Since the baseline EMA strategy is losing money, this is a perfect test case
to see if price action factors can improve it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)
from research.price_action.strategies import create_ema_crossover_strategy


def main():
    """Run focused research on EMA crossover with price action enhancement."""
    print("=" * 100)
    print("FOCUSED PRICE ACTION RESEARCH - EMA CROSSOVER STRATEGY")
    print("=" * 100)
    print("\nGoal: Test if price action factors can improve a losing baseline strategy")
    print("Baseline: EMA 20/50 crossover (currently losing money)")
    print("Test: Apply various price action filters to improve trade quality")
    print("=" * 100)

    backtester = PriceActionBacktester(data_dir="data")
    results_dir = Path("research/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    strategy_dsl = create_ema_crossover_strategy()

    # Phase 1: In-sample (2023)
    print("\n[PHASE 1] In-Sample Testing (2023)")
    print("-" * 100)

    common_params_is = {
        "strategy_dsl": strategy_dsl,
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "5m",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000.0,
    }

    baseline_is = BacktestExperiment(
        name="Baseline_IS",
        description="No price action filtering",
        factor_filter=None,
        **common_params_is,
    )

    variants_is = [
        BacktestExperiment(
            name="IBS_IS",
            description="IBS > 0.6 (close near high)",
            factor_filter=FactorFilterConfig(ibs_long_min=0.6),
            **common_params_is,
        ),
        BacktestExperiment(
            name="Range_ATR_IS",
            description="Range > 1.2x ATR (momentum)",
            factor_filter=FactorFilterConfig(range_atr_min=1.2),
            **common_params_is,
        ),
        BacktestExperiment(
            name="Body_IS",
            description="Body > 60% (conviction)",
            factor_filter=FactorFilterConfig(body_ratio_min=0.6),
            **common_params_is,
        ),
        BacktestExperiment(
            name="Trend_Structure_IS",
            description="Trend structure > 0.3",
            factor_filter=FactorFilterConfig(trend_structure_long_min=0.3),
            **common_params_is,
        ),
        BacktestExperiment(
            name="Trend_Strength_IS",
            description="Composite trend strength > 0.4",
            factor_filter=FactorFilterConfig(trend_strength_long_min=0.4),
            **common_params_is,
        ),
        BacktestExperiment(
            name="Moderate_Combo_IS",
            description="IBS + Range + Trend (moderate)",
            factor_filter=FactorFilterConfig(
                ibs_long_min=0.6,
                range_atr_min=1.2,
                trend_structure_long_min=0.3,
            ),
            **common_params_is,
        ),
    ]

    print("Running in-sample experiments...")
    is_results = backtester.run_ab_test(baseline=baseline_is, variants=variants_is)

    # Phase 2: Out-of-sample (2024)
    print("\n[PHASE 2] Out-of-Sample Validation (2024)")
    print("-" * 100)

    common_params_oos = {
        "strategy_dsl": strategy_dsl,
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "5m",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000.0,
    }

    baseline_oos = BacktestExperiment(
        name="Baseline_OOS",
        description="No price action filtering",
        factor_filter=None,
        **common_params_oos,
    )

    variants_oos = [
        BacktestExperiment(
            name="IBS_OOS",
            description="IBS > 0.6 (close near high)",
            factor_filter=FactorFilterConfig(ibs_long_min=0.6),
            **common_params_oos,
        ),
        BacktestExperiment(
            name="Range_ATR_OOS",
            description="Range > 1.2x ATR (momentum)",
            factor_filter=FactorFilterConfig(range_atr_min=1.2),
            **common_params_oos,
        ),
        BacktestExperiment(
            name="Body_OOS",
            description="Body > 60% (conviction)",
            factor_filter=FactorFilterConfig(body_ratio_min=0.6),
            **common_params_oos,
        ),
        BacktestExperiment(
            name="Trend_Structure_OOS",
            description="Trend structure > 0.3",
            factor_filter=FactorFilterConfig(trend_structure_long_min=0.3),
            **common_params_oos,
        ),
        BacktestExperiment(
            name="Trend_Strength_OOS",
            description="Composite trend strength > 0.4",
            factor_filter=FactorFilterConfig(trend_strength_long_min=0.4),
            **common_params_oos,
        ),
        BacktestExperiment(
            name="Moderate_Combo_OOS",
            description="IBS + Range + Trend (moderate)",
            factor_filter=FactorFilterConfig(
                ibs_long_min=0.6,
                range_atr_min=1.2,
                trend_structure_long_min=0.3,
            ),
            **common_params_oos,
        ),
    ]

    print("Running out-of-sample validation...")
    oos_results = backtester.run_ab_test(baseline=baseline_oos, variants=variants_oos)

    # Print results
    print_results(is_results, oos_results)

    # Save results
    complete_results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "EMA Crossover 20/50",
        "insample": is_results,
        "outofsample": oos_results,
    }

    output_path = results_dir / "ema_focused_research.json"
    with open(output_path, "w") as f:
        json.dump(complete_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def print_results(is_results: dict, oos_results: dict):
    """Print formatted results comparison."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Baseline comparison
    is_baseline = is_results["baseline"]["summary"]
    oos_baseline = oos_results["baseline"]["summary"]

    print(f"\nBASELINE PERFORMANCE:")
    print(f"  In-Sample (2023):")
    print(f"    Trades: {is_baseline['total_trades']}, Win Rate: {is_baseline['win_rate']:.2f}%, Return: {is_baseline['total_return_pct']:.2f}%")
    print(f"  Out-of-Sample (2024):")
    print(f"    Trades: {oos_baseline['total_trades']}, Win Rate: {oos_baseline['win_rate']:.2f}%, Return: {oos_baseline['total_return_pct']:.2f}%")

    print(f"\n{'=' * 100}")
    print("FACTOR PERFORMANCE")
    print("=" * 100)

    # Compare each variant
    for is_comp, oos_comp in zip(is_results["comparison"], oos_results["comparison"]):
        variant_name = is_comp["variant_name"].replace("_IS", "")
        print(f"\n{variant_name}: {is_comp['variant_description']}")
        print("-" * 100)

        # In-sample metrics
        is_metrics = is_comp["metrics"]
        print(f"  In-Sample (2023):")
        print(f"    Return: {is_metrics['total_return_pct']['variant']:.2f}% (baseline: {is_metrics['total_return_pct']['baseline']:.2f}%, change: {is_metrics['total_return_pct']['change']:+.2f}%)")
        print(f"    Win Rate: {is_metrics['win_rate']['variant']:.2f}% (baseline: {is_metrics['win_rate']['baseline']:.2f}%, change: {is_metrics['win_rate']['change']:+.2f}%)")
        print(f"    Trades: {is_metrics['total_trades']['variant']} (baseline: {is_metrics['total_trades']['baseline']}, change: {is_metrics['total_trades']['change']:+d})")
        print(f"    Expectancy: ${is_metrics['expectancy']['variant']:.2f} (baseline: ${is_metrics['expectancy']['baseline']:.2f}, change: ${is_metrics['expectancy']['change']:+.2f})")

        # Out-of-sample metrics
        oos_metrics = oos_comp["metrics"]
        print(f"  Out-of-Sample (2024):")
        print(f"    Return: {oos_metrics['total_return_pct']['variant']:.2f}% (baseline: {oos_metrics['total_return_pct']['baseline']:.2f}%, change: {oos_metrics['total_return_pct']['change']:+.2f}%)")
        print(f"    Win Rate: {oos_metrics['win_rate']['variant']:.2f}% (baseline: {oos_metrics['win_rate']['baseline']:.2f}%, change: {oos_metrics['win_rate']['change']:+.2f}%)")
        print(f"    Trades: {oos_metrics['total_trades']['variant']} (baseline: {oos_metrics['total_trades']['baseline']}, change: {oos_metrics['total_trades']['change']:+d})")
        print(f"    Expectancy: ${oos_metrics['expectancy']['variant']:.2f} (baseline: ${oos_metrics['expectancy']['baseline']:.2f}, change: ${oos_metrics['expectancy']['change']:+.2f})")

        # Validation status
        is_improvement = is_metrics['total_return_pct']['change']
        oos_improvement = oos_metrics['total_return_pct']['change']

        if is_improvement > 0 and oos_improvement > 0:
            print(f"  ✓ VALIDATED: Improvement holds out-of-sample")
        elif is_improvement > 0 and oos_improvement <= 0:
            print(f"  ✗ NOT VALIDATED: In-sample improvement did not hold")
        elif is_improvement <= 0:
            print(f"  ✗ NO IMPROVEMENT: Factor did not help in-sample")

    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Find best validated factor
    best_validated = None
    best_oos_improvement = -float('inf')

    for is_comp, oos_comp in zip(is_results["comparison"], oos_results["comparison"]):
        is_improvement = is_comp["metrics"]["total_return_pct"]["change"]
        oos_improvement = oos_comp["metrics"]["total_return_pct"]["change"]

        if is_improvement > 0 and oos_improvement > 0 and oos_improvement > best_oos_improvement:
            best_oos_improvement = oos_improvement
            best_validated = (is_comp, oos_comp)

    if best_validated:
        is_comp, oos_comp = best_validated
        print(f"\n✓ BEST VALIDATED FACTOR: {is_comp['variant_name'].replace('_IS', '')}")
        print(f"  Description: {is_comp['variant_description']}")
        print(f"  In-Sample Improvement: +{is_comp['metrics']['total_return_pct']['change']:.2f}%")
        print(f"  Out-of-Sample Improvement: +{oos_comp['metrics']['total_return_pct']['change']:.2f}%")
        print(f"  Trade Count Impact: {is_comp['metrics']['total_trades']['change']:+d} trades")
    else:
        print("\n✗ NO VALIDATED IMPROVEMENTS FOUND")
        print("  None of the price action factors improved performance consistently")
        print("  Possible reasons:")
        print("    - Baseline strategy logic may need adjustment")
        print("    - Factor thresholds may need tuning")
        print("    - Market conditions may not suit these factors")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
