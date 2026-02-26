from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from packages.domain.market_data.data.data_loader import DataLoader


def test_000_accessibility_local_parquet_files_exist() -> None:
    data_dir = Path("data")
    parquet_files = list(data_dir.rglob("*.parquet"))
    assert parquet_files, "Expected local parquet files under backend/data."


def test_010_dataloader_loads_real_btcusd_data() -> None:
    loader = DataLoader(data_dir="data")
    frame = loader.load(
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 5, tzinfo=UTC),
    )
    assert not frame.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(frame.columns)


def test_020_dataloader_symbol_metadata_from_real_files() -> None:
    loader = DataLoader(data_dir="data")
    metadata = loader.get_symbol_metadata(market="stocks", symbol="SPY")
    assert metadata["market"] == "us_stocks"
    assert metadata["symbol"] == "SPY"
    assert metadata["file_count"] > 0
    assert "1m" in metadata["available_timeframes"]
