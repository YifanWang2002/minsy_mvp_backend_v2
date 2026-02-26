"""Live integration tests for real Alpaca market data.

These tests use the actual Alpaca API to verify:
1. Real-time crypto market data fetching
2. Historical bars retrieval
3. Quote snapshots
4. Multi-symbol data handling
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

# Clear proxy environment variables to avoid SOCKS proxy issues
for key in (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
):
    os.environ.pop(key, None)

from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.trading.adapters.alpaca_trading import AlpacaTradingAdapter


class TestAlpacaCryptoMarketData:
    """Test suite for real Alpaca crypto market data."""

    @pytest.fixture
    def alpaca_client(self) -> AlpacaMarketDataClient:
        """Create Alpaca market data client with real credentials."""
        # Clear proxy vars again in case they were set
        for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ.pop(key, None)
        return AlpacaMarketDataClient()

    def test_000_fetch_btc_usd_latest_quote(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching latest BTC/USD quote."""
        async def _fetch():
            quote = await alpaca_client.fetch_latest_quote("BTC/USD")
            await alpaca_client.aclose()
            return quote

        quote = asyncio.run(_fetch())
        assert quote is not None, "Expected quote for BTC/USD"
        assert quote.symbol in {"BTCUSD", "BTC/USD"}
        assert quote.last is not None or (quote.bid is not None and quote.ask is not None)
        if quote.last is not None:
            assert quote.last > Decimal("0")
        if quote.bid is not None:
            assert quote.bid > Decimal("0")
        if quote.ask is not None:
            assert quote.ask > Decimal("0")

    def test_010_fetch_eth_usd_latest_quote(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching latest ETH/USD quote."""
        async def _fetch():
            quote = await alpaca_client.fetch_latest_quote("ETH/USD")
            await alpaca_client.aclose()
            return quote

        quote = asyncio.run(_fetch())
        assert quote is not None, "Expected quote for ETH/USD"
        assert quote.last is not None or (quote.bid is not None and quote.ask is not None)

    def test_020_fetch_btc_usd_ohlcv_1min(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching BTC/USD 1-minute OHLCV bars."""
        async def _fetch():
            since = datetime.now(UTC) - timedelta(hours=1)
            bars = await alpaca_client.fetch_ohlcv(
                "BTC/USD",
                since=since,
                timeframe="1Min",
                limit=60,
            )
            await alpaca_client.aclose()
            return bars

        bars = asyncio.run(_fetch())
        assert len(bars) > 0, "Expected at least one bar"
        for bar in bars:
            assert bar.open > Decimal("0")
            assert bar.high >= bar.low
            assert bar.close > Decimal("0")
            assert bar.volume >= Decimal("0")

    def test_030_fetch_btc_usd_ohlcv_5min(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching BTC/USD 5-minute OHLCV bars."""
        async def _fetch():
            since = datetime.now(UTC) - timedelta(hours=6)
            bars = await alpaca_client.fetch_ohlcv(
                "BTC/USD",
                since=since,
                timeframe="5Min",
                limit=72,
            )
            await alpaca_client.aclose()
            return bars

        bars = asyncio.run(_fetch())
        assert len(bars) > 0, "Expected at least one bar"

    def test_040_fetch_multiple_crypto_quotes(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching quotes for multiple crypto pairs."""
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]

        async def _fetch_all():
            quotes = {}
            for symbol in symbols:
                try:
                    quote = await alpaca_client.fetch_latest_quote(symbol)
                    if quote is not None:
                        quotes[symbol] = quote
                except Exception:
                    pass
            await alpaca_client.aclose()
            return quotes

        quotes = asyncio.run(_fetch_all())
        # At least BTC and ETH should be available
        assert len(quotes) >= 2, f"Expected at least 2 quotes, got {len(quotes)}"
        assert "BTC/USD" in quotes or "BTCUSD" in [q.symbol for q in quotes.values()]

    def test_050_fetch_latest_bar(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test fetching latest single bar."""
        async def _fetch():
            bar = await alpaca_client.fetch_latest_bar("BTC/USD")
            await alpaca_client.aclose()
            return bar

        bar = asyncio.run(_fetch())
        assert bar is not None, "Expected latest bar for BTC/USD"
        assert bar.open > Decimal("0")
        assert bar.close > Decimal("0")


class TestAlpacaTradingAdapter:
    """Test suite for real Alpaca trading adapter (paper mode)."""

    @pytest.fixture
    def trading_adapter(self) -> AlpacaTradingAdapter:
        """Create Alpaca trading adapter with real credentials."""
        # Clear proxy vars
        for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ.pop(key, None)
        return AlpacaTradingAdapter()

    def test_000_fetch_account_state(
        self,
        trading_adapter: AlpacaTradingAdapter,
    ) -> None:
        """Test fetching account state from Alpaca paper trading."""
        async def _fetch():
            state = await trading_adapter.fetch_account_state()
            await trading_adapter.aclose()
            return state

        state = asyncio.run(_fetch())
        assert state is not None
        assert state.equity >= Decimal("0")
        assert state.cash >= Decimal("0")
        assert state.buying_power >= Decimal("0")

    def test_010_fetch_positions(
        self,
        trading_adapter: AlpacaTradingAdapter,
    ) -> None:
        """Test fetching positions from Alpaca paper trading."""
        async def _fetch():
            positions = await trading_adapter.fetch_positions()
            await trading_adapter.aclose()
            return positions

        positions = asyncio.run(_fetch())
        assert isinstance(positions, list)
        # Positions may be empty, that's OK
        for position in positions:
            assert position.symbol
            assert position.qty >= Decimal("0")

    def test_020_fetch_ohlcv_via_adapter(
        self,
        trading_adapter: AlpacaTradingAdapter,
    ) -> None:
        """Test fetching OHLCV via trading adapter."""
        async def _fetch():
            bars = await trading_adapter.fetch_ohlcv_1m(
                "BTC/USD",
                limit=30,
            )
            await trading_adapter.aclose()
            return bars

        bars = asyncio.run(_fetch())
        assert len(bars) > 0

    def test_030_fetch_latest_quote_via_adapter(
        self,
        trading_adapter: AlpacaTradingAdapter,
    ) -> None:
        """Test fetching latest quote via trading adapter."""
        async def _fetch():
            quote = await trading_adapter.fetch_latest_quote("ETH/USD")
            await trading_adapter.aclose()
            return quote

        quote = asyncio.run(_fetch())
        assert quote is not None


class TestAlpacaMarketDataTimeframes:
    """Test suite for different timeframe data fetching."""

    @pytest.fixture
    def alpaca_client(self) -> AlpacaMarketDataClient:
        return AlpacaMarketDataClient()

    @pytest.mark.parametrize("timeframe,expected_bars", [
        ("1Min", 30),
        ("5Min", 12),
        ("15Min", 4),
        ("1Hour", 2),
    ])
    def test_various_timeframes(
        self,
        alpaca_client: AlpacaMarketDataClient,
        timeframe: str,
        expected_bars: int,
    ) -> None:
        """Test fetching bars for various timeframes."""
        async def _fetch():
            since = datetime.now(UTC) - timedelta(hours=24)
            bars = await alpaca_client.fetch_ohlcv(
                "BTC/USD",
                since=since,
                timeframe=timeframe,
                limit=expected_bars * 2,
            )
            await alpaca_client.aclose()
            return bars

        bars = asyncio.run(_fetch())
        # Should have at least some bars
        assert len(bars) >= 1, f"Expected bars for {timeframe}"


class TestAlpacaDataQuality:
    """Test suite for data quality validation."""

    @pytest.fixture
    def alpaca_client(self) -> AlpacaMarketDataClient:
        return AlpacaMarketDataClient()

    def test_000_bar_data_consistency(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test that bar data is consistent (high >= low, etc.)."""
        async def _fetch():
            bars = await alpaca_client.fetch_ohlcv(
                "BTC/USD",
                timeframe="1Min",
                limit=100,
            )
            await alpaca_client.aclose()
            return bars

        bars = asyncio.run(_fetch())
        for bar in bars:
            assert bar.high >= bar.low, f"High {bar.high} < Low {bar.low}"
            assert bar.high >= bar.open, f"High {bar.high} < Open {bar.open}"
            assert bar.high >= bar.close, f"High {bar.high} < Close {bar.close}"
            assert bar.low <= bar.open, f"Low {bar.low} > Open {bar.open}"
            assert bar.low <= bar.close, f"Low {bar.low} > Close {bar.close}"

    def test_010_timestamp_ordering(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test that bars are ordered by timestamp."""
        async def _fetch():
            bars = await alpaca_client.fetch_ohlcv(
                "BTC/USD",
                timeframe="1Min",
                limit=50,
            )
            await alpaca_client.aclose()
            return bars

        bars = asyncio.run(_fetch())
        if len(bars) > 1:
            for i in range(1, len(bars)):
                assert bars[i].timestamp >= bars[i - 1].timestamp, (
                    f"Bars not ordered: {bars[i - 1].timestamp} > {bars[i].timestamp}"
                )

    def test_020_quote_spread_reasonable(
        self,
        alpaca_client: AlpacaMarketDataClient,
    ) -> None:
        """Test that bid-ask spread is reasonable."""
        async def _fetch():
            quote = await alpaca_client.fetch_latest_quote("BTC/USD")
            await alpaca_client.aclose()
            return quote

        quote = asyncio.run(_fetch())
        if quote is not None and quote.bid is not None and quote.ask is not None:
            spread = quote.ask - quote.bid
            spread_pct = (spread / quote.bid) * Decimal("100")
            # Spread should be less than 1% for major pairs
            assert spread_pct < Decimal("1"), f"Spread too wide: {spread_pct}%"
