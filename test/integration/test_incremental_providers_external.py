"""External integration tests for incremental data providers.

These tests intentionally hit real services:
- Alpaca market data REST (crypto + US equity, 1m bars)
- Local IBKR Gateway on port 4002 (forex, 1m bars)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from packages.infra.providers.market_data.alpaca_client import AlpacaMarketDataClient
from packages.infra.providers.market_data.ibkr_async import IbkrAsyncMarketDataProvider
from packages.shared_settings.schema.settings import settings


async def _with_retry(
    call: Callable[[], Awaitable[object]],
    *,
    attempts: int = 3,
) -> object:
    last_error: Exception | None = None
    for idx in range(max(1, attempts)):
        try:
            return await call()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 429 or idx >= attempts - 1:
                raise
            await asyncio.sleep(1.5 * (2**idx))
        except (httpx.TimeoutException, OSError) as exc:
            last_error = exc
            if idx >= attempts - 1:
                raise
            await asyncio.sleep(1.0 * (2**idx))
    assert last_error is not None
    raise last_error


def _last_weekday_window_utc(*, minutes: int = 30) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    end = now.replace(hour=15, minute=0, second=0, microsecond=0)
    if end >= now:
        end -= timedelta(days=1)
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    start = end - timedelta(minutes=max(1, int(minutes)))
    return start, end


@pytest.mark.external
@pytest.mark.asyncio
async def test_external_alpaca_real_1m_fetch_for_equity_and_crypto() -> None:
    client = AlpacaMarketDataClient()
    try:
        equity_start, equity_end = _last_weekday_window_utc(minutes=20)

        async def _fetch_equity():
            return await client.fetch_ohlcv(
                "AAPL",
                market="stocks",
                timeframe="1Min",
                since=equity_start,
                until=equity_end,
                limit=40,
            )

        equity_bars = await _with_retry(_fetch_equity)
        assert len(equity_bars) > 0, "Expected real Alpaca 1m bars for AAPL."

        async def _fetch_crypto():
            return await client.fetch_ohlcv(
                "ETH/USD",
                market="crypto",
                timeframe="1Min",
                since=datetime.now(UTC) - timedelta(hours=4),
                limit=30,
            )

        crypto_bars = await _with_retry(_fetch_crypto)
        assert len(crypto_bars) > 0, "Expected real Alpaca 1m bars for ETH/USD."
    finally:
        await client.aclose()


@pytest.mark.external
@pytest.mark.asyncio
async def test_external_ibkr_real_1m_fetch_on_local_gateway_4002(
    monkeypatch,
) -> None:
    # Keep provider hard gate while allowing this local external test to initialize.
    monkeypatch.setattr(
        settings,
        "market_data_incremental_execution_mode",
        "local_collector",
    )

    host = os.getenv("IBKR_GATEWAY_TEST_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("IBKR_GATEWAY_TEST_PORT", "4002"))
    client_id = int(os.getenv("IBKR_GATEWAY_TEST_CLIENT_ID", "491"))
    start, end = _last_weekday_window_utc(minutes=45)

    provider = IbkrAsyncMarketDataProvider(
        host=host,
        port=port,
        client_id=client_id,
    )
    try:
        bars = await _with_retry(
            lambda: provider.fetch_ohlcv(
                symbol="EURUSD",
                market="forex",
                timeframe="1m",
                since=start,
                until=end,
                limit=120,
            ),
            attempts=2,
        )
        assert len(bars) > 0, (
            "Expected real IBKR 1m bars for EURUSD from local gateway "
            f"{host}:{port}. "
            "Check IBKR Gateway is running and API access is enabled."
        )
    finally:
        await provider.aclose()
