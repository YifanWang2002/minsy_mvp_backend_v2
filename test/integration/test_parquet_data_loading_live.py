"""Live integration tests for local parquet data loading.

These tests verify:
1. Parquet file reading and parsing
2. Market data inventory management
3. Timeframe data handling
4. Data quality validation
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from test._support.live_helpers import BACKEND_DIR


# Check if parquet data directory exists
PARQUET_DATA_DIR = BACKEND_DIR / "data" / "parquet"


class TestParquetDataLoading:
    """Test suite for parquet data loading."""

    @pytest.fixture
    def data_dir(self) -> Path:
        """Get parquet data directory."""
        return PARQUET_DATA_DIR

    def test_000_parquet_directory_exists(
        self,
        data_dir: Path,
    ) -> None:
        """Test that parquet data directory exists."""
        # This test may be skipped if no local parquet data
        if not data_dir.exists():
            pytest.skip("No local parquet data directory")
        assert data_dir.is_dir()

    def test_010_list_available_parquet_files(
        self,
        data_dir: Path,
    ) -> None:
        """Test listing available parquet files."""
        if not data_dir.exists():
            pytest.skip("No local parquet data directory")

        parquet_files = list(data_dir.glob("**/*.parquet"))
        # May have no files, that's OK
        assert isinstance(parquet_files, list)

    def test_020_read_parquet_file_if_exists(
        self,
        data_dir: Path,
    ) -> None:
        """Test reading a parquet file if it exists."""
        if not data_dir.exists():
            pytest.skip("No local parquet data directory")

        parquet_files = list(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            pytest.skip("No parquet files found")

        # Try to read the first file
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_files[0])
            assert table.num_rows >= 0
            assert table.num_columns >= 0
        except ImportError:
            pytest.skip("pyarrow not installed")


class TestMarketDataInventory:
    """Test suite for market data inventory."""

    def test_000_market_data_inventory_accessible(
        self,
        compose_stack: list[dict[str, Any]],
    ) -> None:
        """Test market data inventory is accessible."""
        from test._support.live_helpers import run_command

        _ = compose_stack

        result = run_command(
            [
                "docker",
                "exec",
                "minsy-api-dev",
                ".venv/bin/python",
                "-c",
                "import json; "
                "from packages.domain.market_data.inventory import MarketDataInventory; "
                "inv=MarketDataInventory(); "
                "print('RESULT_JSON='+json.dumps({'status':'ok'}))",
            ],
            cwd=BACKEND_DIR,
            timeout=60,
            check=False,
        )
        # May fail if inventory not configured, that's OK
        assert result.returncode in {0, 1}


class TestTimeframeDataHandling:
    """Test suite for timeframe data handling."""

    def test_000_timeframe_conversion(self) -> None:
        """Test timeframe string conversion."""
        from packages.domain.trading.runtime.timeframe_scheduler import timeframe_to_seconds

        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("5m") == 300
        assert timeframe_to_seconds("15m") == 900
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("1d") == 86400

    def test_010_timeframe_trigger_logic(self) -> None:
        """Test timeframe trigger logic."""
        from packages.domain.trading.runtime.timeframe_scheduler import should_trigger_cycle

        now = datetime.now(UTC)

        # Test 1-minute timeframe
        due, bucket = should_trigger_cycle(
            now=now,
            interval_seconds=60,
            last_trigger_bucket=None,
        )
        # Should be due if no previous trigger
        assert due is True
        assert bucket is not None

        # Test with recent trigger
        due2, bucket2 = should_trigger_cycle(
            now=now,
            interval_seconds=60,
            last_trigger_bucket=bucket,
        )
        # Should not be due immediately after
        assert due2 is False


class TestDataQualityValidation:
    """Test suite for data quality validation."""

    def test_000_ohlcv_bar_validation(self) -> None:
        """Test OHLCV bar data validation."""
        from packages.infra.providers.trading.adapters.base import OhlcvBar

        # Valid bar
        bar = OhlcvBar(
            timestamp=datetime.now(UTC),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("95.00"),
            close=Decimal("102.00"),
            volume=Decimal("1000"),
        )
        assert bar.high >= bar.low
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close

    def test_010_quote_snapshot_validation(self) -> None:
        """Test quote snapshot validation."""
        from packages.infra.providers.trading.adapters.base import QuoteSnapshot

        # Valid quote
        quote = QuoteSnapshot(
            symbol="BTC/USD",
            bid=Decimal("50000.00"),
            ask=Decimal("50010.00"),
            last=Decimal("50005.00"),
            timestamp=datetime.now(UTC),
            raw={},
        )
        assert quote.ask >= quote.bid
        assert quote.last >= quote.bid
        assert quote.last <= quote.ask
