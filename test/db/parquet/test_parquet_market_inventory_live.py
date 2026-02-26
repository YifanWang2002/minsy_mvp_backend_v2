from __future__ import annotations

from pathlib import Path

from packages.domain.market_data.data.data_loader import DataLoader


_EXPECTED_MARKETS = {"crypto", "forex", "futures", "us_stocks"}


def test_000_accessibility_expected_market_dirs_have_parquet() -> None:
    data_dir = Path("data")
    for market in _EXPECTED_MARKETS:
        market_dir = data_dir / market
        assert market_dir.is_dir(), f"missing market dir: {market_dir}"
        files = list(market_dir.glob("*.parquet"))
        assert files, f"no parquet files in {market_dir}"


def test_010_dataloader_market_inventory_non_empty() -> None:
    loader = DataLoader(data_dir="data")
    markets = set(loader.get_available_markets())
    assert _EXPECTED_MARKETS.issubset(markets)


def test_020_dataloader_symbols_non_empty_for_each_market() -> None:
    loader = DataLoader(data_dir="data")
    for market in _EXPECTED_MARKETS:
        symbols = loader.get_available_symbols(market)
        assert len(symbols) > 0, market
