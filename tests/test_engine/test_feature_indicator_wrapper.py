from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.feature.indicators import calculate


def _build_ohlcv(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    noise = np.sin(np.linspace(0.0, 8.0, rows))
    close = base + noise
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000.0, 3000.0, rows),
        },
        index=index,
    )


def test_sma_and_macd_can_be_calculated_from_migrated_indicators() -> None:
    data = _build_ohlcv()

    sma = calculate("sma", data, length=5)
    assert isinstance(sma, pd.Series)
    assert sma.notna().sum() > 0

    macd = calculate("macd", data, fast=12, slow=26, signal=9)
    assert isinstance(macd, pd.DataFrame)
    assert macd.shape[0] == data.shape[0]
    assert macd.shape[1] >= 3
