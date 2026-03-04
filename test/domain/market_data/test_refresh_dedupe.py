from __future__ import annotations

import asyncio

from datetime import UTC, datetime, timedelta

from apps.api.routes import market_data as market_data_route
from packages.domain.market_data import refresh_dedupe
from packages.infra.providers.trading.adapters.base import QuoteSnapshot


class _BrokenRedisClient:
    def set(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("redis unavailable")


def test_refresh_dedupe_falls_back_locally(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_dedupe,
        "get_sync_redis_client",
        lambda: _BrokenRedisClient(),
    )
    monkeypatch.setattr(
        refresh_dedupe.settings,
        "market_data_refresh_dedupe_enabled",
        True,
    )
    monkeypatch.setattr(
        refresh_dedupe.settings,
        "market_data_refresh_active_subscriptions_interval_seconds",
        1,
    )
    monkeypatch.setattr(
        refresh_dedupe.settings,
        "market_data_refresh_dedupe_window_seconds",
        1,
    )
    refresh_dedupe._REFRESH_DEDUPE_FALLBACK.clear()

    try:
        assert refresh_dedupe.reserve_market_data_refresh_slot("stocks", "AAPL") is True
        assert refresh_dedupe.reserve_market_data_refresh_slot("stocks", "AAPL") is False
        assert refresh_dedupe.reserve_market_data_refresh_slot("stocks", "MSFT") is True
    finally:
        refresh_dedupe._REFRESH_DEDUPE_FALLBACK.clear()


def test_market_data_bars_path_uses_cache_only_for_live_quote(monkeypatch) -> None:
    def _unexpected_provider_fetch(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("bars route should not fetch provider quote inline")

    stale_quote = QuoteSnapshot(
        symbol="AAPL",
        bid=None,
        ask=None,
        last=101.25,
        timestamp=datetime.now(UTC) - timedelta(minutes=5),
    )
    monkeypatch.setattr(
        market_data_route.market_data_runtime,
        "get_latest_quote",
        lambda **_kwargs: stale_quote,
    )
    monkeypatch.setattr(
        market_data_route.AlpacaRestProvider,
        "__init__",
        _unexpected_provider_fetch,
    )

    result = asyncio.run(
        market_data_route._resolve_live_quote(
            market="stocks",
            symbol="AAPL",
        )
    )
    assert result is None
