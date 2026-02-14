from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.engine.data import DataLoader


def _make_ohlcv(
    *,
    start: str,
    periods: int,
    freq: str,
    base_price: float,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    opens = pd.Series([base_price + i for i in range(periods)], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.5,
            "volume": pd.Series([10.0 * (i + 1) for i in range(periods)], dtype=float),
        }
    )


@pytest.fixture
def loader_with_mock_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DataLoader:
    files: dict[str, pd.DataFrame] = {
        "crypto/BTCUSD_5min_eth_2024.parquet": _make_ohlcv(
            start="2024-01-01 00:00:00",
            periods=6,
            freq="5min",
            base_price=100.0,
        ),
        "crypto/BTCUSD_1min_eth_2024.parquet": _make_ohlcv(
            start="2024-01-01 00:00:00",
            periods=10,
            freq="1min",
            base_price=100.0,
        ),
        "crypto/BTCUSD_1min_eth_2025.parquet": _make_ohlcv(
            start="2025-01-01 00:00:00",
            periods=5,
            freq="1min",
            base_price=200.0,
        ),
        "crypto/ETHUSD_1min_eth_2024.parquet": _make_ohlcv(
            start="2024-01-01 00:00:00",
            periods=3,
            freq="1min",
            base_price=300.0,
        ),
        "us_stocks/SPY_5min_rth_2024.parquet": _make_ohlcv(
            start="2024-01-02 14:30:00",
            periods=4,
            freq="5min",
            base_price=400.0,
        ),
    }

    parquet_map: dict[Path, pd.DataFrame] = {}
    for relative_path, dataframe in files.items():
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        parquet_map[file_path.resolve()] = dataframe

    loader = DataLoader(data_dir=tmp_path)

    def fake_read_parquet(
        file_path: Path,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        dataframe = parquet_map[Path(file_path).resolve()]
        if columns is None:
            return dataframe.copy()
        return dataframe.loc[:, columns].copy()

    monkeypatch.setattr(loader, "_read_parquet", fake_read_parquet)
    return loader


def test_load_resample_to_15m(loader_with_mock_parquet: DataLoader) -> None:
    result = loader_with_mock_parquet.load(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        start_date=datetime(2024, 1, 1, 0, 0, 0),
        end_date=datetime(2024, 1, 1, 0, 29, 59),
    )

    assert list(result.index.strftime("%Y-%m-%d %H:%M:%S%z")) == [
        "2024-01-01 00:00:00+0000",
        "2024-01-01 00:15:00+0000",
    ]
    first = result.iloc[0]
    second = result.iloc[1]

    assert first["open"] == pytest.approx(100.0)
    assert first["high"] == pytest.approx(103.0)
    assert first["low"] == pytest.approx(99.0)
    assert first["close"] == pytest.approx(102.5)
    assert first["volume"] == pytest.approx(60.0)

    assert second["open"] == pytest.approx(103.0)
    assert second["high"] == pytest.approx(106.0)
    assert second["low"] == pytest.approx(102.0)
    assert second["close"] == pytest.approx(105.5)
    assert second["volume"] == pytest.approx(150.0)


def test_load_5m_falls_back_to_1m_when_5m_file_missing(
    loader_with_mock_parquet: DataLoader,
) -> None:
    result = loader_with_mock_parquet.load(
        market="crypto",
        symbol="BTCUSD",
        timeframe="5m",
        start_date=datetime(2025, 1, 1, 0, 0, 0),
        end_date=datetime(2025, 1, 1, 0, 4, 59),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["open"] == pytest.approx(200.0)
    assert row["high"] == pytest.approx(205.0)
    assert row["low"] == pytest.approx(199.0)
    assert row["close"] == pytest.approx(204.5)
    assert row["volume"] == pytest.approx(150.0)


def test_symbols_and_metadata(loader_with_mock_parquet: DataLoader) -> None:
    assert loader_with_mock_parquet.get_available_symbols("crypto") == [
        "BTCUSD",
        "ETHUSD",
    ]
    assert loader_with_mock_parquet.get_available_symbols("stock") == ["SPY"]

    crypto_metadata = loader_with_mock_parquet.get_symbol_metadata("crypto", "btcusd")
    assert crypto_metadata["available_timerange"] == {
        "start": "2024-01-01T00:00:00+00:00",
        "end": "2025-01-01T00:04:00+00:00",
    }
    assert crypto_metadata["available_timeframes"] == [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "1d",
    ]

    stock_metadata = loader_with_mock_parquet.get_symbol_metadata("stock", "SPY")
    assert stock_metadata["session"] == "rth"
    assert stock_metadata["available_timeframes"] == [
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "1d",
    ]
