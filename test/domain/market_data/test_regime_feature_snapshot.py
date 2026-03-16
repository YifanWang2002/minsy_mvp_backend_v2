from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.domain.market_data.regime.feature_snapshot import (
    build_regime_feature_snapshot,
)


def _build_sample_ohlcv_frame(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0, 18, rows)
    close = 100 + np.sin(x) * 3 + np.linspace(0, 12, rows)
    open_ = close + np.cos(x) * 0.25
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = 1000 + (np.sin(x * 1.5) * 120) + np.linspace(0, 80, rows)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return frame


def test_feature_snapshot_contains_expected_sections() -> None:
    frame = _build_sample_ohlcv_frame()

    snapshot = build_regime_feature_snapshot(
        frame,
        timeframe="1h",
        lookback_bars=250,
        pivot_window=5,
    )

    expected_sections = {
        "window_stats",
        "price_path_summary",
        "swing_structure",
        "efficiency_noise",
        "volatility_level",
        "volatility_state",
        "volatility_direction_coupling",
        "trend_reversion",
        "volume_participation",
        "meta",
    }
    assert expected_sections.issubset(snapshot.keys())
    assert int(snapshot["meta"]["bars"]) == 250
    assert snapshot["meta"]["timeframe"] == "1h"
    assert isinstance(snapshot["volatility_direction_coupling"]["trend_with_low_vol"], bool)


def test_feature_snapshot_rejects_missing_ohlcv_columns() -> None:
    frame = _build_sample_ohlcv_frame()[["open", "high", "low", "close"]]

    with pytest.raises(ValueError, match="Missing required columns"):
        build_regime_feature_snapshot(
            frame,
            timeframe="1h",
            lookback_bars=200,
            pivot_window=5,
        )
