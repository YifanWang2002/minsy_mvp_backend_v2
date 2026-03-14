from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from packages.domain.market_data.incremental.provider_router import (
    normalize_incremental_market,
    resolve_provider_for_market,
)
from packages.domain.market_data.incremental.session_gate import (
    market_is_open_for_incremental,
)
from packages.infra.providers.market_data import ibkr_async as ibkr_async_provider

_NY = ZoneInfo("America/New_York")


def test_provider_router_matches_required_market_mapping() -> None:
    assert resolve_provider_for_market("crypto") == "alpaca"
    assert resolve_provider_for_market("us_stocks") == "alpaca"
    assert resolve_provider_for_market("us_equity") == "alpaca"
    assert resolve_provider_for_market("forex") == "ibkr"
    assert resolve_provider_for_market("futures") == "ibkr"


def test_provider_router_rejects_unknown_market() -> None:
    with pytest.raises(ValueError, match="Unsupported incremental market"):
        normalize_incremental_market("hk_equity")


def test_session_gate_keeps_crypto_open_on_weekend() -> None:
    saturday_utc = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
    assert market_is_open_for_incremental(market="crypto", now=saturday_utc)


def test_session_gate_closes_us_stocks_on_weekend() -> None:
    sunday_utc = datetime(2026, 3, 15, 15, 0, tzinfo=UTC)
    assert not market_is_open_for_incremental(market="us_stocks", now=sunday_utc)


def test_session_gate_closes_forex_after_friday_close_until_sunday_open() -> None:
    friday_after_close = datetime(2026, 3, 13, 18, 0, tzinfo=_NY).astimezone(UTC)
    sunday_before_open = datetime(2026, 3, 15, 16, 0, tzinfo=_NY).astimezone(UTC)
    sunday_at_open = datetime(2026, 3, 15, 17, 0, tzinfo=_NY).astimezone(UTC)

    assert not market_is_open_for_incremental(market="forex", now=friday_after_close)
    assert not market_is_open_for_incremental(market="forex", now=sunday_before_open)
    assert market_is_open_for_incremental(market="forex", now=sunday_at_open)


def test_ibkr_provider_rejects_non_local_execution_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        ibkr_async_provider.settings,
        "market_data_incremental_execution_mode",
        "remote_importer",
    )
    with pytest.raises(RuntimeError, match="local_collector"):
        ibkr_async_provider.IbkrAsyncMarketDataProvider()
