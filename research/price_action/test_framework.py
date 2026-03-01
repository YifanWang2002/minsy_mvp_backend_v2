"""Quick test to verify the price action research framework works.

This script runs a minimal test to ensure all components are properly integrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from research.price_action.factors import PriceActionFactors
from research.price_action.backtest import (
    BacktestExperiment,
    FactorFilterConfig,
    PriceActionBacktester,
)


def generate_sample_data(bars: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate a trending price series
    base_price = 50000.0
    trend = np.linspace(0, 5000, bars)
    noise = np.random.randn(bars) * 500

    close_prices = base_price + trend + noise

    data = []
    for i, close in enumerate(close_prices):
        # Generate OHLC from close
        volatility = 200 + np.random.rand() * 300
        open_price = close + np.random.randn() * volatility * 0.5
        high = max(open_price, close) + np.random.rand() * volatility
        low = min(open_price, close) - np.random.rand() * volatility
        volume = 1000000 + np.random.rand() * 500000

        timestamp = datetime(2024, 1, 1) + timedelta(minutes=5 * i)

        data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def test_factors():
    """Test factor calculations."""
    print("Testing factor calculations...")

    # Generate sample data
    df = generate_sample_data(500)

    # Calculate all factors
    df_with_factors = PriceActionFactors.calculate_all_factors(df)

    # Check that all factor columns exist
    expected_columns = [
        "pa_ibs",
        "pa_range_atr",
        "pa_body_ratio",
        "pa_consecutive",
        "pa_trend_structure",
        "pa_two_leg",
        "pa_breakout_quality",
        "pa_trend_strength",
    ]

    for col in expected_columns:
        assert col in df_with_factors.columns, f"Missing column: {col}"
        assert not df_with_factors[col].isna().all(), f"Column {col} is all NaN"

    print("✓ All factors calculated successfully")

    # Print sample statistics
    print("\nFactor Statistics:")
    for col in expected_columns:
        values = df_with_factors[col].dropna()
        print(f"  {col}:")
        print(f"    Mean: {values.mean():.3f}")
        print(f"    Std:  {values.std():.3f}")
        print(f"    Min:  {values.min():.3f}")
        print(f"    Max:  {values.max():.3f}")


def test_backtest_framework():
    """Test backtest framework with sample data."""
    print("\n" + "=" * 80)
    print("Testing backtest framework...")

    # Create a simple strategy DSL
    strategy_dsl = {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "Test Strategy",
            "description": "Simple test strategy",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "ema_20": {
                "type": "ema",
                "params": {"source": "close", "period": 20},
            },
            "ema_50": {
                "type": "ema",
                "params": {"source": "close", "period": 50},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_20"},
                            "op": "cross_above",
                            "b": {"ref": "ema_50"},
                        }
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop",
                        "stop": {"kind": "pct", "value": 0.02},
                    },
                    {
                        "type": "take_profit",
                        "name": "target",
                        "take": {"kind": "pct", "value": 0.04},
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }

    print("✓ Strategy DSL created")

    # Note: We can't run a full backtest without real data files
    # But we can verify the framework components are importable and instantiable

    try:
        backtester = PriceActionBacktester(data_dir="data")
        print("✓ PriceActionBacktester instantiated")

        # Create experiment config
        experiment = BacktestExperiment(
            name="Test",
            description="Test experiment",
            strategy_dsl=strategy_dsl,
            factor_filter=FactorFilterConfig(ibs_long_min=0.6),
            market="crypto",
            symbol="BTCUSD",
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
        print("✓ BacktestExperiment created")

        print("\nFramework components verified successfully!")
        print("\nTo run full backtests, ensure you have data files in the data/ directory:")
        print("  data/crypto/BTCUSD_5min_eth_2023.parquet")
        print("  data/crypto/BTCUSD_5min_eth_2024.parquet")

    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 80)
    print("PRICE ACTION RESEARCH FRAMEWORK - QUICK TEST")
    print("=" * 80)

    try:
        test_factors()
        test_backtest_framework()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Prepare your data files (see README.md)")
        print("2. Run: python -m research.price_action.experiments")
        print("3. Analyze results in research/results/")

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
