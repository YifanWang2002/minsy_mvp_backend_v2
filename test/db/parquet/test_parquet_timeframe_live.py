from __future__ import annotations

from datetime import UTC, datetime

from packages.domain.market_data.data.data_loader import DataLoader


def test_000_accessibility_stocks_and_crypto_1m_load() -> None:
    loader = DataLoader(data_dir="data")

    stocks_frame = loader.load(
        market="stocks",
        symbol="SPY",
        timeframe="1m",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 5, tzinfo=UTC),
    )
    crypto_frame = loader.load(
        market="crypto",
        symbol="BTCUSD",
        timeframe="1m",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 5, tzinfo=UTC),
    )

    assert not stocks_frame.empty
    assert not crypto_frame.empty


def test_010_resample_to_15m_is_queryable() -> None:
    loader = DataLoader(data_dir="data")
    frame = loader.load(
        market="stocks",
        symbol="SPY",
        timeframe="15m",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime(2024, 1, 15, tzinfo=UTC),
    )
    assert not frame.empty
    assert set(frame.columns) >= {"open", "high", "low", "close", "volume"}


def test_020_available_timeframes_include_resampled_levels() -> None:
    loader = DataLoader(data_dir="data")
    timeframes = loader.get_available_timeframes(market="stocks", symbol="SPY", include_resampled=True)
    assert "1m" in timeframes
    assert "5m" in timeframes
    assert "15m" in timeframes
