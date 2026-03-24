from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.domain.backtest.factors import prepare_backtest_frame
from packages.domain.market_data.regime.family_scoring import score_strategy_families
from packages.domain.market_data.regime.feature_snapshot import build_regime_feature_snapshot
from packages.domain.market_data.runtime import RuntimeBar
from packages.domain.strategy import validate_strategy_payload
from packages.domain.strategy.parser import build_parsed_strategy
from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime


def make_variant(base: dict, name: str, extra_factors: dict) -> dict:
    payload = copy.deepcopy(base)
    payload["strategy"]["name"] = name
    payload["factors"].update(extra_factors)
    return payload


def build_frame(
    rows: int,
    freq_min: int,
    seed: int,
    *,
    regime_profile: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows, freq=f"{freq_min}min", tz="UTC")
    x = np.linspace(0.0, 18.0, rows)

    if regime_profile == "trend":
        drift = np.linspace(0.0, 18.0, rows)
        seasonal = 0.8 * np.sin(x * 0.4)
        noise = rng.normal(0.0, 0.1, rows)
    elif regime_profile == "volatility_shock":
        drift = np.linspace(0.0, 4.0, rows)
        seasonal = 3.2 * np.sin(x * 1.5) + 2.4 * np.cos(x * 1.1)
        noise = rng.normal(0.0, 0.7, rows)
    else:
        drift = np.linspace(0.0, 10.0, rows)
        seasonal = 2.5 * np.sin(x) + 1.5 * np.cos(x * 0.7)
        noise = rng.normal(0.0, 0.2, rows)

    close = 100 + drift + seasonal + noise
    open_ = close + rng.normal(0.0, 0.18 if regime_profile == "volatility_shock" else 0.15, rows)
    spread_base = 1.2 if regime_profile == "volatility_shock" else 0.7
    spread = np.abs(rng.normal(spread_base, 0.16, rows)) + 0.1
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.maximum(
        1000
        + (180 if regime_profile == "volatility_shock" else 80) * np.sin(x * 1.3)
        + rng.normal(0, 40 if regime_profile == "volatility_shock" else 20, rows),
        1,
    )

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def bars_from_frame(frame: pd.DataFrame) -> list[RuntimeBar]:
    bars: list[RuntimeBar] = []
    for ts, row in frame.iterrows():
        bars.append(
            RuntimeBar(
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return bars


def main() -> None:
    base = json.loads(
        Path("packages/domain/strategy/assets/example_strategy.json").read_text(
            encoding="utf-8"
        )
    )

    variants = [
        make_variant(base, "base", {}),
        make_variant(
            base,
            "trend_plus",
            {
                "adx_14": {
                    "type": "adx",
                    "params": {"length": 14},
                    "outputs": ["adx", "dmp", "dmn"],
                },
                "donchian_20": {
                    "type": "donchian",
                    "params": {"lower_length": 20, "upper_length": 20},
                    "outputs": ["DCL", "DCM", "DCU"],
                },
            },
        ),
        make_variant(
            base,
            "mr_plus",
            {
                "bbands_20_2": {
                    "type": "bbands",
                    "params": {"period": 20, "std_dev": 2.0, "source": "close"},
                    "outputs": ["upper", "middle", "lower"],
                },
                "stoch_14_3_3": {
                    "type": "stoch",
                    "params": {"k_period": 14, "k_smooth": 3, "d_period": 3},
                    "outputs": ["k", "d"],
                },
            },
        ),
        make_variant(
            base,
            "regime_plus",
            {
                "eff_ratio_50": {
                    "type": "efficiency_ratio",
                    "params": {"length": 50, "source": "close"},
                },
                "vol_regime_ratio": {
                    "type": "volatility_regime_ratio",
                    "params": {"short_window": 20, "long_window": 60, "source": "close"},
                },
                "sqz_20": {
                    "type": "squeeze_score",
                    "params": {
                        "length": 20,
                        "bb_mult": 2.0,
                        "kc_mult": 1.0,
                        "percentile_window": 120,
                    },
                },
            },
        ),
    ]

    time_cfgs = [
        ("15m", 15, 700),
        ("1h", 60, 900),
        ("1d", 1440, 600),
    ]
    regime_profiles = ("mean_reversion", "trend", "volatility_shock")

    runtime = LiveSignalRuntime()
    total_cases = 0
    max_global_diff = 0.0
    all_regimes: set[str] = set()

    print("# Multi Scenario Consistency Report")
    print(
        "variant,regime_profile,timeframe,rows,factor_cols,max_abs_diff_last80,signal,recommended_regime"
    )

    for variant_index, payload in enumerate(variants):
        validation = validate_strategy_payload(payload)
        if not validation.is_valid:
            raise AssertionError((payload["strategy"]["name"], validation.errors))

        parsed = build_parsed_strategy(payload)
        factor_ids = list(parsed.factors.keys())

        for regime_profile in regime_profiles:
            for timeframe, freq_min, rows in time_cfgs:
                total_cases += 1
                frame = build_frame(
                    rows,
                    freq_min,
                    seed=1000 + variant_index * 100 + freq_min,
                    regime_profile=regime_profile,
                )
                enriched = prepare_backtest_frame(frame, strategy=parsed)

                bars = bars_from_frame(frame)
                _, live_frame, _ = runtime._build_enriched_frame(
                    strategy_payload=payload,
                    bars=bars,
                )
                decision = runtime.evaluate(
                    strategy_payload=payload,
                    bars=bars,
                )

                factor_cols = [
                    col
                    for col in enriched.columns
                    if any(col == fid or col.startswith(f"{fid}.") for fid in factor_ids)
                ]
                sample_cols = factor_cols[: min(len(factor_cols), 80)]

                max_diff = 0.0
                for column in sample_cols:
                    a = pd.to_numeric(enriched[column], errors="coerce").tail(80)
                    b = pd.to_numeric(live_frame[column], errors="coerce").tail(80)
                    diff = (a - b).abs().fillna(0.0)
                    if len(diff):
                        max_diff = max(max_diff, float(diff.max()))

                if max_diff >= 1e-9:
                    raise AssertionError(
                        f"backtest/live mismatch variant={payload['strategy']['name']} regime={regime_profile} timeframe={timeframe} max_diff={max_diff}"
                    )

                max_global_diff = max(max_global_diff, max_diff)

                snapshot = build_regime_feature_snapshot(
                    frame,
                    timeframe=timeframe,
                    lookback_bars=min(500, rows),
                    pivot_window=5,
                )
                scores = score_strategy_families(snapshot)
                recommended = scores.recommended_family
                if recommended not in {
                    "trend_continuation",
                    "mean_reversion",
                    "volatility_regime",
                }:
                    raise AssertionError(f"unexpected recommended family: {recommended}")
                all_regimes.add(recommended)

                print(
                    f"{payload['strategy']['name']},{regime_profile},{timeframe},{rows},{len(factor_cols)},{max_diff:.3e},{decision.signal},{recommended}"
                )

    print()
    print(f"TOTAL_CASES={total_cases}")
    print(f"MAX_GLOBAL_ABS_DIFF={max_global_diff:.3e}")
    print(f"RECOMMENDED_REGIME_SET={sorted(all_regimes)}")
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
