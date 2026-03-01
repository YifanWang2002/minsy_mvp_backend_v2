"""Comprehensive price action research runner with multiple strategies and out-of-sample validation.

This script:
1. Tests multiple baseline trend-following strategies
2. Applies price action factors to each baseline
3. Uses in-sample (2023) and out-of-sample (2024) validation
4. Generates comprehensive comparison reports
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)
from research.price_action.strategies import get_all_baseline_strategies


class PriceActionResearcher:
    """Autonomous price action research system."""

    def __init__(self, data_dir: str = "data"):
        self.backtester = PriceActionBacktester(data_dir=data_dir)
        self.results_dir = Path("research/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_full_research(self) -> dict[str, Any]:
        """Run complete research workflow with all strategies and validations.

        Returns:
            Complete research results with in-sample and out-of-sample validation
        """
        print("=" * 100)
        print("PRICE ACTION RESEARCH - MULTI-STRATEGY WITH OUT-OF-SAMPLE VALIDATION")
        print("=" * 100)

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "strategies": {},
        }

        strategies = get_all_baseline_strategies()

        for strategy_name, strategy_dsl in strategies.items():
            print(f"\n{'=' * 100}")
            print(f"STRATEGY: {strategy_name.upper()}")
            print(f"{'=' * 100}")

            # Phase 1: In-sample testing (2023)
            print(f"\n[Phase 1] In-sample testing (2023)...")
            insample_results = self._run_strategy_experiments(
                strategy_name=strategy_name,
                strategy_dsl=strategy_dsl,
                start_date="2023-01-01",
                end_date="2023-12-31",
                phase="insample",
            )

            # Phase 2: Out-of-sample validation (2024)
            print(f"\n[Phase 2] Out-of-sample validation (2024)...")
            outofsample_results = self._run_strategy_experiments(
                strategy_name=strategy_name,
                strategy_dsl=strategy_dsl,
                start_date="2024-01-01",
                end_date="2024-12-31",
                phase="outofsample",
            )

            all_results["strategies"][strategy_name] = {
                "insample": insample_results,
                "outofsample": outofsample_results,
            }

            # Print comparison
            self._print_strategy_summary(strategy_name, insample_results, outofsample_results)

        # Save complete results
        self._save_results(all_results, "complete_research_results.json")

        # Generate summary report
        self._generate_summary_report(all_results)

        return all_results

    def _run_strategy_experiments(
        self,
        strategy_name: str,
        strategy_dsl: dict,
        start_date: str,
        end_date: str,
        phase: str,
    ) -> dict[str, Any]:
        """Run experiments for a single strategy in a specific time period.

        Args:
            strategy_name: Name of the strategy
            strategy_dsl: Strategy DSL definition
            start_date: Start date for backtest
            end_date: End date for backtest
            phase: "insample" or "outofsample"

        Returns:
            Experiment results with baseline and variants
        """
        common_params = {
            "strategy_dsl": strategy_dsl,
            "market": "crypto",
            "symbol": "BTCUSD",
            "timeframe": "5m",
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 100000.0,
        }

        # Baseline: no filtering
        baseline = BacktestExperiment(
            name=f"{strategy_name}_baseline_{phase}",
            description=f"Baseline {strategy_name} without price action filtering",
            factor_filter=None,
            **common_params,
        )

        # Define factor variants to test
        variants = self._create_factor_variants(strategy_name, phase, common_params)

        # Run A/B test
        results = self.backtester.run_ab_test(baseline=baseline, variants=variants)

        return results

    def _create_factor_variants(
        self,
        strategy_name: str,
        phase: str,
        common_params: dict,
    ) -> list[BacktestExperiment]:
        """Create factor filter variants for testing.

        Args:
            strategy_name: Name of the strategy
            phase: "insample" or "outofsample"
            common_params: Common experiment parameters

        Returns:
            List of variant experiments
        """
        variants = []

        # Variant 1: IBS filter (strong bar position)
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_ibs_{phase}",
                description="IBS > 0.6 (close near high)",
                factor_filter=FactorFilterConfig(ibs_long_min=0.6),
                **common_params,
            )
        )

        # Variant 2: Range/ATR filter (momentum bars)
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_range_atr_{phase}",
                description="Range > 1.2x ATR (momentum)",
                factor_filter=FactorFilterConfig(range_atr_min=1.2),
                **common_params,
            )
        )

        # Variant 3: Body ratio filter (conviction)
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_body_{phase}",
                description="Body > 60% of range (conviction)",
                factor_filter=FactorFilterConfig(body_ratio_min=0.6),
                **common_params,
            )
        )

        # Variant 4: Trend structure filter
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_trend_structure_{phase}",
                description="Trend structure > 0.3 (aligned trend)",
                factor_filter=FactorFilterConfig(trend_structure_long_min=0.3),
                **common_params,
            )
        )

        # Variant 5: Composite trend strength
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_trend_strength_{phase}",
                description="Composite trend strength > 0.4",
                factor_filter=FactorFilterConfig(trend_strength_long_min=0.4),
                **common_params,
            )
        )

        # Variant 6: Conservative multi-filter
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_conservative_{phase}",
                description="Conservative: IBS + Range + Body + Trend",
                factor_filter=FactorFilterConfig(
                    ibs_long_min=0.65,
                    range_atr_min=1.3,
                    body_ratio_min=0.65,
                    trend_structure_long_min=0.4,
                ),
                **common_params,
            )
        )

        # Variant 7: Moderate multi-filter
        variants.append(
            BacktestExperiment(
                name=f"{strategy_name}_moderate_{phase}",
                description="Moderate: IBS + Range + Trend",
                factor_filter=FactorFilterConfig(
                    ibs_long_min=0.6,
                    range_atr_min=1.2,
                    trend_structure_long_min=0.3,
                ),
                **common_params,
            )
        )

        return variants

    def _print_strategy_summary(
        self,
        strategy_name: str,
        insample: dict,
        outofsample: dict,
    ) -> None:
        """Print summary comparison for a strategy.

        Args:
            strategy_name: Name of the strategy
            insample: In-sample results
            outofsample: Out-of-sample results
        """
        print(f"\n{'=' * 100}")
        print(f"SUMMARY: {strategy_name.upper()}")
        print(f"{'=' * 100}")

        # Baseline comparison
        is_baseline = insample["baseline"]["summary"]
        oos_baseline = outofsample["baseline"]["summary"]

        print(f"\nBaseline Performance:")
        print(f"  In-sample (2023):")
        print(f"    Trades: {is_baseline['total_trades']}, Win Rate: {is_baseline['win_rate']:.2f}%, Return: {is_baseline['total_return_pct']:.2f}%")
        print(f"  Out-of-sample (2024):")
        print(f"    Trades: {oos_baseline['total_trades']}, Win Rate: {oos_baseline['win_rate']:.2f}%, Return: {oos_baseline['total_return_pct']:.2f}%")

        # Find best variant in-sample
        best_is_variant = max(
            insample["comparison"],
            key=lambda x: x["metrics"]["total_return_pct"]["variant"],
        )

        print(f"\nBest In-Sample Variant: {best_is_variant['variant_name']}")
        print(f"  {best_is_variant['variant_description']}")
        print(f"  Return: {best_is_variant['metrics']['total_return_pct']['variant']:.2f}% (vs baseline {best_is_variant['metrics']['total_return_pct']['baseline']:.2f}%)")
        print(f"  Win Rate: {best_is_variant['metrics']['win_rate']['variant']:.2f}% (vs baseline {best_is_variant['metrics']['win_rate']['baseline']:.2f}%)")
        print(f"  Trades: {best_is_variant['metrics']['total_trades']['variant']} (vs baseline {best_is_variant['metrics']['total_trades']['baseline']})")

        # Check if same variant performs well out-of-sample
        variant_name_base = best_is_variant['variant_name'].replace('_insample', '_outofsample')
        oos_variant = next(
            (v for v in outofsample["comparison"] if v["variant_name"] == variant_name_base),
            None,
        )

        if oos_variant:
            print(f"\nSame Variant Out-of-Sample:")
            print(f"  Return: {oos_variant['metrics']['total_return_pct']['variant']:.2f}% (vs baseline {oos_variant['metrics']['total_return_pct']['baseline']:.2f}%)")
            print(f"  Win Rate: {oos_variant['metrics']['win_rate']['variant']:.2f}% (vs baseline {oos_variant['metrics']['win_rate']['baseline']:.2f}%)")
            print(f"  Trades: {oos_variant['metrics']['total_trades']['variant']} (vs baseline {oos_variant['metrics']['total_trades']['baseline']})")

            # Check if improvement holds
            is_improvement = oos_variant['metrics']['total_return_pct']['change']
            if is_improvement > 0:
                print(f"  ✓ IMPROVEMENT VALIDATED OUT-OF-SAMPLE (+{is_improvement:.2f}%)")
            else:
                print(f"  ✗ Improvement did not hold out-of-sample ({is_improvement:.2f}%)")

    def _generate_summary_report(self, all_results: dict) -> None:
        """Generate final summary report across all strategies.

        Args:
            all_results: Complete research results
        """
        print(f"\n{'=' * 100}")
        print("FINAL RESEARCH SUMMARY")
        print(f"{'=' * 100}")

        print("\nKey Findings:")
        print("-" * 100)

        for strategy_name, results in all_results["strategies"].items():
            is_baseline = results["insample"]["baseline"]["summary"]
            oos_baseline = results["outofsample"]["baseline"]["summary"]

            # Find best in-sample variant
            best_is = max(
                results["insample"]["comparison"],
                key=lambda x: x["metrics"]["total_return_pct"]["variant"],
            )

            # Find corresponding out-of-sample variant
            variant_name_base = best_is['variant_name'].replace('_insample', '_outofsample')
            best_oos = next(
                (v for v in results["outofsample"]["comparison"] if v["variant_name"] == variant_name_base),
                None,
            )

            print(f"\n{strategy_name.upper()}:")
            print(f"  Baseline: {is_baseline['total_return_pct']:.2f}% (IS) → {oos_baseline['total_return_pct']:.2f}% (OOS)")

            if best_oos:
                is_return = best_is['metrics']['total_return_pct']['variant']
                oos_return = best_oos['metrics']['total_return_pct']['variant']
                is_improvement = best_is['metrics']['total_return_pct']['change']
                oos_improvement = best_oos['metrics']['total_return_pct']['change']

                print(f"  Best Filter: {best_is['variant_description']}")
                print(f"    Returns: {is_return:.2f}% (IS) → {oos_return:.2f}% (OOS)")
                print(f"    Improvement: +{is_improvement:.2f}% (IS) → {'+' if oos_improvement > 0 else ''}{oos_improvement:.2f}% (OOS)")

                if oos_improvement > 0:
                    print(f"    Status: ✓ VALIDATED")
                else:
                    print(f"    Status: ✗ NOT VALIDATED")

        print(f"\n{'=' * 100}")
        print("Research complete! Results saved to research/results/")
        print(f"{'=' * 100}")

    def _save_results(self, results: dict, filename: str) -> None:
        """Save results to JSON file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = self.results_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    """Run complete price action research."""
    researcher = PriceActionResearcher(data_dir="data")
    results = researcher.run_full_research()

    print("\n" + "=" * 100)
    print("RESEARCH COMPLETE")
    print("=" * 100)
    print("\nNext steps:")
    print("1. Review results in research/results/complete_research_results.json")
    print("2. Identify factors that consistently improve performance across strategies")
    print("3. Validate findings hold out-of-sample")
    print("4. Integrate validated factors into production strategies")


if __name__ == "__main__":
    main()
