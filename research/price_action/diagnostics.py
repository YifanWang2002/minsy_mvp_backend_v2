"""Diagnostic script to understand factor distributions and filtering impact."""

import pandas as pd
from research.price_action.factors import PriceActionFactors
from packages.domain.market_data.data.data_loader import DataLoader


def main():
    """Analyze factor distributions to understand filtering behavior."""
    print("=" * 100)
    print("PRICE ACTION FACTOR DIAGNOSTICS")
    print("=" * 100)

    # Load data
    loader = DataLoader(data_dir="data")
    data = loader.load(
        market="crypto",
        symbol="BTCUSD",
        timeframe="5m",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    print(f"\nLoaded {len(data)} bars from 2023")

    # Calculate factors
    data_with_factors = PriceActionFactors.calculate_all_factors(data)

    print("\n" + "=" * 100)
    print("FACTOR STATISTICS")
    print("=" * 100)

    factors = [
        "pa_ibs",
        "pa_range_atr",
        "pa_body_ratio",
        "pa_consecutive",
        "pa_trend_structure",
        "pa_two_leg",
        "pa_breakout_quality",
        "pa_trend_strength",
    ]

    for factor in factors:
        values = data_with_factors[factor].dropna()
        print(f"\n{factor}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  Min: {values.min():.4f}")
        print(f"  25%: {values.quantile(0.25):.4f}")
        print(f"  50%: {values.quantile(0.50):.4f}")
        print(f"  75%: {values.quantile(0.75):.4f}")
        print(f"  Max: {values.max():.4f}")

    print("\n" + "=" * 100)
    print("FILTER IMPACT ANALYSIS")
    print("=" * 100)

    # Test different thresholds
    test_configs = [
        ("IBS > 0.6", lambda df: df["pa_ibs"] >= 0.6),
        ("IBS > 0.5", lambda df: df["pa_ibs"] >= 0.5),
        ("Range/ATR > 1.2", lambda df: df["pa_range_atr"] >= 1.2),
        ("Range/ATR > 1.0", lambda df: df["pa_range_atr"] >= 1.0),
        ("Body > 0.6", lambda df: df["pa_body_ratio"] >= 0.6),
        ("Body > 0.5", lambda df: df["pa_body_ratio"] >= 0.5),
        ("Trend Structure > 0.3", lambda df: df["pa_trend_structure"] >= 0.3),
        ("Trend Structure > 0.2", lambda df: df["pa_trend_structure"] >= 0.2),
        ("Trend Strength > 0.4", lambda df: df["pa_trend_strength"] >= 0.4),
        ("Trend Strength > 0.3", lambda df: df["pa_trend_strength"] >= 0.3),
    ]

    total_bars = len(data_with_factors)

    for name, condition in test_configs:
        passing_bars = condition(data_with_factors).sum()
        pct = (passing_bars / total_bars) * 100
        print(f"\n{name}:")
        print(f"  Passing bars: {passing_bars} / {total_bars} ({pct:.2f}%)")

    # Test combined filters
    print("\n" + "=" * 100)
    print("COMBINED FILTER IMPACT")
    print("=" * 100)

    combined_configs = [
        ("IBS>0.6 + Range>1.2", lambda df: (df["pa_ibs"] >= 0.6) & (df["pa_range_atr"] >= 1.2)),
        ("IBS>0.5 + Range>1.0", lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_range_atr"] >= 1.0)),
        ("IBS>0.6 + Body>0.6", lambda df: (df["pa_ibs"] >= 0.6) & (df["pa_body_ratio"] >= 0.6)),
        ("IBS>0.5 + Body>0.5", lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_body_ratio"] >= 0.5)),
        ("All moderate (IBS>0.5, Range>1.0, Trend>0.2)",
         lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_range_atr"] >= 1.0) & (df["pa_trend_structure"] >= 0.2)),
    ]

    for name, condition in combined_configs:
        passing_bars = condition(data_with_factors).sum()
        pct = (passing_bars / total_bars) * 100
        print(f"\n{name}:")
        print(f"  Passing bars: {passing_bars} / {total_bars} ({pct:.2f}%)")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print("\nBased on the analysis above:")
    print("1. Identify which thresholds allow 20-50% of bars to pass")
    print("2. These thresholds will filter trades without being too restrictive")
    print("3. Use these adjusted thresholds in the next research iteration")


if __name__ == "__main__":
    main()
