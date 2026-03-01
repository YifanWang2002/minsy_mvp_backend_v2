"""Analyze EMA crossover signals and factor filtering impact."""

import pandas as pd
from research.price_action.factors import PriceActionFactors
from packages.domain.market_data.data.data_loader import DataLoader


def main():
    """Analyze EMA crossover signals specifically."""
    print("=" * 100)
    print("EMA CROSSOVER SIGNAL ANALYSIS")
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

    # Calculate EMAs
    ema_20 = data["close"].ewm(span=20, adjust=False).mean()
    ema_50 = data["close"].ewm(span=50, adjust=False).mean()

    # Detect crossovers
    ema_20_above = ema_20 > ema_50
    ema_20_above_prev = ema_20_above.shift(1).fillna(False)
    crossover_up = ema_20_above & ~ema_20_above_prev

    crossover_signals = data[crossover_up].copy()
    print(f"\nTotal EMA crossover signals: {len(crossover_signals)}")

    if len(crossover_signals) == 0:
        print("No crossover signals found!")
        return

    # Calculate factors for crossover bars only
    data_with_factors = PriceActionFactors.calculate_all_factors(data)
    crossover_factors = data_with_factors[crossover_up].copy()

    print("\n" + "=" * 100)
    print("FACTOR STATISTICS AT CROSSOVER SIGNALS")
    print("=" * 100)

    factors = [
        "pa_ibs",
        "pa_range_atr",
        "pa_body_ratio",
        "pa_consecutive",
        "pa_trend_structure",
        "pa_trend_strength",
    ]

    for factor in factors:
        values = crossover_factors[factor].dropna()
        print(f"\n{factor}:")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std: {values.std():.4f}")
        print(f"  Min: {values.min():.4f}")
        print(f"  25%: {values.quantile(0.25):.4f}")
        print(f"  50%: {values.quantile(0.50):.4f}")
        print(f"  75%: {values.quantile(0.75):.4f}")
        print(f"  Max: {values.max():.4f}")

    print("\n" + "=" * 100)
    print("FILTER IMPACT ON CROSSOVER SIGNALS")
    print("=" * 100)

    total_signals = len(crossover_factors)

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

    for name, condition in test_configs:
        passing_signals = condition(crossover_factors).sum()
        pct = (passing_signals / total_signals) * 100 if total_signals > 0 else 0
        print(f"\n{name}:")
        print(f"  Passing signals: {passing_signals} / {total_signals} ({pct:.2f}%)")
        print(f"  Filtered out: {total_signals - passing_signals} signals")

    print("\n" + "=" * 100)
    print("COMBINED FILTERS ON CROSSOVER SIGNALS")
    print("=" * 100)

    combined_configs = [
        ("IBS>0.6 + Range>1.2", lambda df: (df["pa_ibs"] >= 0.6) & (df["pa_range_atr"] >= 1.2)),
        ("IBS>0.5 + Range>1.0", lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_range_atr"] >= 1.0)),
        ("IBS>0.6 + Body>0.6", lambda df: (df["pa_ibs"] >= 0.6) & (df["pa_body_ratio"] >= 0.6)),
        ("IBS>0.5 + Body>0.5", lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_body_ratio"] >= 0.5)),
        ("Moderate (IBS>0.5, Range>1.0, Trend>0.2)",
         lambda df: (df["pa_ibs"] >= 0.5) & (df["pa_range_atr"] >= 1.0) & (df["pa_trend_structure"] >= 0.2)),
    ]

    for name, condition in combined_configs:
        passing_signals = condition(crossover_factors).sum()
        pct = (passing_signals / total_signals) * 100 if total_signals > 0 else 0
        print(f"\n{name}:")
        print(f"  Passing signals: {passing_signals} / {total_signals} ({pct:.2f}%)")
        print(f"  Filtered out: {total_signals - passing_signals} signals")

    print("\n" + "=" * 100)
    print("KEY INSIGHT")
    print("=" * 100)
    print(f"\nThe baseline strategy generates {total_signals} EMA crossover signals in 2023.")
    print("Price action factors should filter these signals, not all bars.")
    print("\nRecommended approach:")
    print("  - Use looser thresholds that keep 30-70% of signals")
    print("  - This maintains reasonable trade frequency while improving quality")


if __name__ == "__main__":
    main()
