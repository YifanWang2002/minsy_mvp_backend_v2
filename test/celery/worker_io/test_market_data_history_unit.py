from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from apps.worker.io.tasks import market_data as market_data_tasks
from apps.worker.io.tasks.market_data import (
    _aggregate_bars_from_source,
    _bars_are_ready,
    _fallback_source_request,
    _merge_history_rows,
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


def test_merge_history_rows_dedupes_and_keeps_latest_window() -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    rows = [
        _bar(now - timedelta(minutes=3), "100"),
        _bar(now - timedelta(minutes=2), "101"),
        _bar(now - timedelta(minutes=2), "102"),
        _bar(now - timedelta(minutes=1), "103"),
    ]
    merged = _merge_history_rows(target_bars=3, groups=[rows[:2], rows[2:]])
    assert len(merged) == 3
    # duplicate timestamp should keep the latest row ("102")
    assert merged[1].open == Decimal("102")


class _StubRuntime:
    def __init__(self) -> None:
        self._bars: dict[str, list[OhlcvBar]] = {}
        self.last_quote = None
        self.last_ingested_bar = None

    def get_recent_bars(self, *, market: str, symbol: str, timeframe: str, limit: int):
        _ = (market, symbol)
        return list(self._bars.get(timeframe, []))[-limit:]

    def restore_bars(self, *, market: str, symbol: str, timeframe: str, bars):
        _ = (market, symbol, timeframe)
        return list(bars)

    def hydrate_bars(self, *, market: str, symbol: str, timeframe: str, bars):
        _ = (market, symbol)
        normalized = list(bars)
        self._bars[timeframe] = normalized
        return normalized

    def ingest_1m_bar(self, *, market: str, symbol: str, bar: OhlcvBar):
        _ = (market, symbol)
        self.last_ingested_bar = bar
        return {}

    def upsert_quote(self, *, market: str, symbol: str, quote):
        _ = (market, symbol)
        self.last_quote = quote
        return None


class _StubHistoryProvider:
    def __init__(self, dataset: dict[str, list[OhlcvBar]]) -> None:
        self._dataset = dataset
        self.calls: list[str] = []

    async def fetch_recent_bars(
        self,
        *,
        symbol: str,
        market: str,
        timeframe: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[OhlcvBar]:
        _ = (symbol, market)
        self.calls.append(timeframe)
        rows = list(self._dataset.get(timeframe, []))
        if since is not None:
            rows = [row for row in rows if row.timestamp >= since]
        if until is not None:
            rows = [row for row in rows if row.timestamp <= until]
        rows = sorted(rows, key=lambda row: row.timestamp)
        if len(rows) > limit:
            rows = rows[-limit:]
        return rows

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str):
        _ = (symbol, market)
        return None

    async def fetch_quote(self, *, symbol: str, market: str):
        _ = (symbol, market)
        return None

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_history_warmup_prefers_1m_and_direct_fills_only_requested_timeframe(
    monkeypatch,
) -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    dataset = {
        "1m": [
            _bar(now - timedelta(minutes=4), "100"),
            _bar(now - timedelta(minutes=3), "101"),
            _bar(now - timedelta(minutes=2), "102"),
            _bar(now - timedelta(minutes=1), "103"),
        ],
        "1d": [
            _bar(now - timedelta(days=4), "90"),
            _bar(now - timedelta(days=3), "91"),
            _bar(now - timedelta(days=2), "92"),
            _bar(now - timedelta(days=1), "93"),
        ],
    }
    runtime = _StubRuntime()
    provider = _StubHistoryProvider(dataset)

    monkeypatch.setattr(market_data_tasks, "market_data_runtime", runtime)
    monkeypatch.setattr(market_data_tasks, "AlpacaRestProvider", lambda: provider)
    monkeypatch.setattr(market_data_tasks.settings, "market_data_history_target_bars", 4)
    monkeypatch.setattr(market_data_tasks.settings, "market_data_history_warmup_chunk_bars", 4)
    monkeypatch.setattr(market_data_tasks.settings, "market_data_aggregate_timeframes_csv", "5m,1d")

    summary = await market_data_tasks._ensure_symbol_history_once(
        market="crypto",
        symbol="BTCUSD",
        requested_timeframe="1d",
        min_bars=4,
    )

    assert summary["1m"]["source"] == "direct"
    assert summary["1d"]["direct_bars"] > 0
    assert summary["1d"]["status"] in {"hydrated_direct", "partial_direct"}
    # 5m should be derived from 1m warmup without direct REST call.
    assert "5m" in summary
    assert "5m" not in provider.calls


class _StubRefreshProvider:
    def __init__(self, *, bar: OhlcvBar | None) -> None:
        self._bar = bar
        self.fetch_quote_calls = 0

    async def fetch_latest_1m_bar(self, *, symbol: str, market: str):
        _ = (symbol, market)
        return self._bar

    async def fetch_quote(self, *, symbol: str, market: str):
        _ = (symbol, market)
        self.fetch_quote_calls += 1
        return None

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_refresh_symbol_uses_latest_bar_to_seed_quote_without_extra_quote_call(
    monkeypatch,
) -> None:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    runtime = _StubRuntime()
    provider = _StubRefreshProvider(bar=_bar(now, "123"))

    async def _fake_ensure_history(**kwargs):
        _ = kwargs
        return {"1m": {"status": "ready", "bars": 0, "target_bars": 0}}

    monkeypatch.setattr(market_data_tasks, "_ensure_symbol_history_once", _fake_ensure_history)
    monkeypatch.setattr(market_data_tasks, "market_data_runtime", runtime)
    monkeypatch.setattr(market_data_tasks, "AlpacaRestProvider", lambda: provider)

    payload = await market_data_tasks._refresh_symbol_once(
        market="crypto",
        symbol="BTCUSD",
    )

    assert payload["status"] == "ok"
    assert provider.fetch_quote_calls == 0
    assert runtime.last_ingested_bar is not None
    assert runtime.last_quote is not None
    assert runtime.last_quote.raw == {"source": "latest_1m_bar"}
