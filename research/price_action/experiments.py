"""Example experiments for price action factor research.

This script demonstrates how to:
1. Define a baseline strategy
2. Create factor-filtered variants
3. Run A/B tests
4. Analyze results
"""

from __future__ import annotations

import json
from pathlib import Path

from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)


def create_baseline_strategy() -> dict:
    """Create a simple trend-following strategy as baseline.

    This is a basic EMA crossover strategy without any price action filtering.
    """
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "EMA Crossover Baseline",
            "description": "Simple EMA crossover without price action filters",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "ema_fast": {
                "type": "ema",
                "params": {"source": "close", "period": 20},
            },
            "ema_slow": {
                "type": "ema",
                "params": {"source": "close", "period": 50},
            },
            "atr_14": {
                "type": "atr",
                "params": {"period": 14},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_fast"},
                            "op": "cross_above",
                            "b": {"ref": "ema_slow"},
                        }
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop_loss",
                        "stop": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 2.0,
                        },
                    },
                    {
                        "type": "take_profit",
                        "name": "take_profit",
                        "take": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 4.0,
                        },
                    },
                    {
                        "type": "signal_exit",
                        "name": "ema_cross_down",
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_fast"},
                                "op": "cross_below",
                                "b": {"ref": "ema_slow"},
                            }
                        },
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }


def run_single_factor_experiments() -> dict:
    """Run experiments testing each factor individually.

    This helps identify which factors have the most impact.
    """
    backtester = PriceActionBacktester(data_dir="data")

    # Common experiment parameters
    common_params = {
        "strategy_dsl": create_baseline_strategy(),
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "5m",
        "start_date": "2023-01-01",
        "end_date": "2025-01-01",
        "initial_capital": 100000.0,
    }

    # Baseline: no filtering
    baseline = BacktestExperiment(
        name="Baseline",
        description="No price action filtering",
        factor_filter=None,
        **common_params,
    )

    # Variant 1: IBS filter (only enter on strong bars)
    variant_ibs = BacktestExperiment(
        name="IBS Filter",
        description="Only enter when IBS > 0.6 (close near high)",
        factor_filter=FactorFilterConfig(
            ibs_long_min=0.6,
        ),
        **common_params,
    )

    # Variant 2: Bar range vs ATR (only enter on momentum bars)
    variant_range = BacktestExperiment(
        name="Range/ATR Filter",
        description="Only enter when bar range > 1.2x ATR",
        factor_filter=FactorFilterConfig(
            range_atr_min=1.2,
        ),
        **common_params,
    )

    # Variant 3: Body ratio (only enter on conviction bars)
    variant_body = BacktestExperiment(
        name="Body Ratio Filter",
        description="Only enter when body > 60% of range",
        factor_filter=FactorFilterConfig(
            body_ratio_min=0.6,
        ),
        **common_params,
    )

    # Variant 4: Consecutive bars (only enter with momentum)
    variant_consecutive = BacktestExperiment(
        name="Consecutive Bars Filter",
        description="Only enter after 2+ consecutive bullish bars",
        factor_filter=FactorFilterConfig(
            consecutive_long_min=2,
        ),
        **common_params,
    )

    # Variant 5: Trend structure (only enter in aligned trend)
    variant_trend = BacktestExperiment(
        name="Trend Structure Filter",
        description="Only enter when trend structure score > 0.3",
        factor_filter=FactorFilterConfig(
            trend_structure_long_min=0.3,
        ),
        **common_params,
    )

    # Variant 6: Breakout quality (only enter on quality breakouts)
    variant_breakout = BacktestExperiment(
        name="Breakout Quality Filter",
        description="Only enter when breakout quality > 0.7",
        factor_filter=FactorFilterConfig(
            breakout_quality_min=0.7,
        ),
        **common_params,
    )

    # Variant 7: Composite trend strength
    variant_composite = BacktestExperiment(
        name="Trend Strength Filter",
        description="Only enter when composite trend strength > 0.4",
        factor_filter=FactorFilterConfig(
            trend_strength_long_min=0.4,
        ),
        **common_params,
    )

    # Run A/B test
    results = backtester.run_ab_test(
        baseline=baseline,
        variants=[
            variant_ibs,
            variant_range,
            variant_body,
            variant_consecutive,
            variant_trend,
            variant_breakout,
            variant_composite,
        ],
    )

    return results


def run_combined_factor_experiments() -> dict:
    """Run experiments with combinations of the best-performing factors.

    After identifying top individual factors, test combinations.
    """
    backtester = PriceActionBacktester(data_dir="data")

    common_params = {
        "strategy_dsl": create_baseline_strategy(),
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "5m",
        "start_date": "2023-01-01",
        "end_date": "2025-01-01",
        "initial_capital": 100000.0,
    }

    baseline = BacktestExperiment(
        name="Baseline",
        description="No price action filtering",
        factor_filter=None,
        **common_params,
    )

    # Combination 1: IBS + Range/ATR (momentum + position)
    variant_combo1 = BacktestExperiment(
        name="IBS + Range/ATR",
        description="Strong bar position and momentum",
        factor_filter=FactorFilterConfig(
            ibs_long_min=0.6,
            range_atr_min=1.2,
        ),
        **common_params,
    )

    # Combination 2: Body ratio + Consecutive (conviction + momentum)
    variant_combo2 = BacktestExperiment(
        name="Body + Consecutive",
        description="Strong conviction with momentum",
        factor_filter=FactorFilterConfig(
            body_ratio_min=0.6,
            consecutive_long_min=2,
        ),
        **common_params,
    )

    # Combination 3: Trend structure + Breakout quality
    variant_combo3 = BacktestExperiment(
        name="Trend + Breakout",
        description="Aligned trend with quality breakout",
        factor_filter=FactorFilterConfig(
            trend_structure_long_min=0.3,
            breakout_quality_min=0.7,
        ),
        **common_params,
    )

    # Combination 4: Conservative (multiple filters)
    variant_conservative = BacktestExperiment(
        name="Conservative Multi-Filter",
        description="Strict filtering for high-quality setups only",
        factor_filter=FactorFilterConfig(
            ibs_long_min=0.65,
            range_atr_min=1.3,
            body_ratio_min=0.65,
            trend_structure_long_min=0.4,
        ),
        **common_params,
    )

    # Combination 5: Aggressive (looser filters)
    variant_aggressive = BacktestExperiment(
        name="Aggressive Multi-Filter",
        description="Moderate filtering to maintain trade frequency",
        factor_filter=FactorFilterConfig(
            ibs_long_min=0.55,
            range_atr_min=1.0,
            body_ratio_min=0.5,
        ),
        **common_params,
    )

    results = backtester.run_ab_test(
        baseline=baseline,
        variants=[
            variant_combo1,
            variant_combo2,
            variant_combo3,
            variant_conservative,
            variant_aggressive,
        ],
    )

    return results


def print_comparison_summary(results: dict) -> None:
    """Print a summary of A/B test results."""
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON RESULTS")
    print("=" * 80)

    baseline = results["baseline"]
    print(f"\nBaseline: {baseline['experiment_name']}")
    print(f"  Total Trades: {baseline['summary']['total_trades']}")
    print(f"  Win Rate: {baseline['summary']['win_rate']:.2f}%")
    print(f"  Total Return: {baseline['summary']['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {baseline['summary']['max_drawdown_pct']:.2f}%")

    print("\n" + "-" * 80)
    print("VARIANTS")
    print("-" * 80)

    for comparison in results["comparison"]:
        print(f"\n{comparison['variant_name']}")
        print(f"  {comparison['variant_description']}")

        metrics = comparison["metrics"]

        print(f"\n  Total Trades:")
        print(f"    Baseline: {metrics['total_trades']['baseline']}")
        print(f"    Variant:  {metrics['total_trades']['variant']}")
        print(f"    Change:   {metrics['total_trades']['change']:+d} ({metrics['total_trades']['change_pct']:+.1f}%)")

        print(f"\n  Win Rate:")
        print(f"    Baseline: {metrics['win_rate']['baseline']:.2f}%")
        print(f"    Variant:  {metrics['win_rate']['variant']:.2f}%")
        print(f"    Change:   {metrics['win_rate']['change']:+.2f}%")

        print(f"\n  Total Return:")
        print(f"    Baseline: {metrics['total_return_pct']['baseline']:.2f}%")
        print(f"    Variant:  {metrics['total_return_pct']['variant']:.2f}%")
        print(f"    Change:   {metrics['total_return_pct']['change']:+.2f}%")

        print(f"\n  Expectancy (avg PnL per trade):")
        print(f"    Baseline: ${metrics['expectancy']['baseline']:.2f}")
        print(f"    Variant:  ${metrics['expectancy']['variant']:.2f}")
        print(f"    Change:   ${metrics['expectancy']['change']:+.2f} ({metrics['expectancy']['change_pct']:+.1f}%)")

        print("\n  " + "-" * 76)


def save_results(results: dict, filename: str) -> None:
    """Save results to JSON file."""
    output_dir = Path("research/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Run all experiments."""
    print("Starting Price Action Factor Research")
    print("=" * 80)

    # Phase 1: Single factor experiments
    print("\nPhase 1: Testing individual factors...")
    single_factor_results = run_single_factor_experiments()
    print_comparison_summary(single_factor_results)
    save_results(single_factor_results, "single_factor_results.json")

    # Phase 2: Combined factor experiments
    print("\n\nPhase 2: Testing factor combinations...")
    combined_factor_results = run_combined_factor_experiments()
    print_comparison_summary(combined_factor_results)
    save_results(combined_factor_results, "combined_factor_results.json")

    print("\n" + "=" * 80)
    print("Research complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
