from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from apps.worker.io.tasks.market_data import (
    _aggregate_bars_from_source,
    _bars_are_ready,
    _fallback_source_request,
    _provider_window_bar_cap,
)
from packages.infra.providers.trading.adapters.base import OhlcvBar


def _bar(ts: datetime, price: str, volume: str = "1") -> OhlcvBar:
    value = Decimal(price)
    return OhlcvBar(
        timestamp=ts,
        open=value,
        high=value + Decimal("1"),
        low=value - Decimal("1"),
        close=value + Decimal("0.5"),
        volume=Decimal(volume),
    )


def test_crypto_history_requires_fresh_latest_bar() -> None:
    stale_end = datetime.now(UTC) - timedelta(days=14)
    bars = [
        _bar(stale_end - timedelta(hours=2), "100"),
        _bar(stale_end - timedelta(hours=1), "101"),
        _bar(stale_end, "102"),
    ]

    assert not _bars_are_ready(
        market="crypto",
        timeframe="1h",
        bars=bars,
        target_bars=3,
    )


def test_aggregate_hourly_bars_into_4h_buckets() -> None:
    start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    source_bars = [_bar(start + timedelta(hours=offset), str(100 + offset)) for offset in range(8)]

    rows = _aggregate_bars_from_source(
        source_bars=source_bars,
        source_timeframe="1h",
        target_timeframe="4h",
        target_bars=10,
    )

    assert len(rows) == 2
    assert rows[0].timestamp == start
    assert rows[1].timestamp == start + timedelta(hours=4)
    assert rows[0].open == Decimal("100")
    assert rows[0].close == Decimal("103.5")
    assert rows[0].volume == Decimal("4")
    assert rows[1].open == Decimal("104")
    assert rows[1].close == Decimal("107.5")
    assert rows[1].volume == Decimal("4")


def test_crypto_4h_fallback_uses_1h_source_depth() -> None:
    assert _fallback_source_request(
        market="crypto",
        timeframe="4h",
        target_bars=2500,
    ) == ("1h", 10000)


def test_crypto_hourly_history_uses_provider_window_cap() -> None:
    assert _provider_window_bar_cap("crypto", "1h") == 167
    assert _provider_window_bar_cap("crypto", "4h") == 41
