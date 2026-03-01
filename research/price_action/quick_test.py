"""Quick test to verify one strategy works before running full research."""

from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)
from research.price_action.strategies import create_ema_crossover_strategy


def main():
    """Run quick test on EMA crossover strategy."""
    backtester = PriceActionBacktester(data_dir="data")

    # Test baseline
    baseline = BacktestExperiment(
        name="EMA Baseline Test",
        description="Quick test of EMA crossover",
        strategy_dsl=create_ema_crossover_strategy(),
        factor_filter=None,
        market="crypto",
        symbol="BTCUSD",
        timeframe="5m",
        start_date="2023-01-01",
        end_date="2023-03-31",  # Just Q1 2023 for speed
        initial_capital=100000.0,
    )

    print("Running baseline test...")
    result = backtester.run_experiment(baseline)

    print(f"\nResults:")
    print(f"  Total Trades: {result['summary']['total_trades']}")
    print(f"  Win Rate: {result['summary']['win_rate']:.2f}%")
    print(f"  Total Return: {result['summary']['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {result['summary']['max_drawdown_pct']:.2f}%")
    print(f"  Final Equity: ${result['summary']['final_equity']:.2f}")

    if result['summary']['total_trades'] == 0:
        print("\n⚠️  No trades generated! Strategy conditions may be too strict.")
    elif result['summary']['total_return_pct'] < -10:
        print("\n⚠️  Strategy is losing money. Price action factors may not help.")
    else:
        print("\n✓ Strategy looks reasonable. Ready for full research.")


if __name__ == "__main__":
    main()
