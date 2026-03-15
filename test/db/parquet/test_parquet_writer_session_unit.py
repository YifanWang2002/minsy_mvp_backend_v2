from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from packages.domain.market_data.data import DataLoader
from packages.domain.market_data.data.parquet_writer import append_ohlcv_rows


def test_append_ohlcv_rows_respects_explicit_session(tmp_path) -> None:
    loader = DataLoader(data_dir=tmp_path)
    rows = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 13, 14, 30, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [1000.0],
        }
    )

    result = append_ohlcv_rows(
        loader=loader,
        market="stocks",
        symbol="AAPL",
        timeframe="1m",
        session="eth",
        rows=rows,
    )

    assert result.rows_written == 1
    target = tmp_path / "us_stocks" / "AAPL_1min_eth_2026.parquet"
    assert target.exists()
