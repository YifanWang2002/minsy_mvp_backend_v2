from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from apps.api.routes import trading_ws
from packages.domain.market_data.runtime import RuntimeBar


def _make_bars(count: int) -> list[RuntimeBar]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[RuntimeBar] = []
    for index in range(count):
        price = 100.0 + float(index)
        rows.append(
            RuntimeBar(
                timestamp=start + timedelta(minutes=index),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1000.0,
            )
        )
    return rows


def test_load_bar_snapshot_uses_inline_hydration_for_short_cache(monkeypatch) -> None:
    full_snapshot = _make_bars(3)
    state = {"calls": 0, "hydrates": 0}

    def _fake_get_recent_bars(**_kwargs):  # type: ignore[no-untyped-def]
        state["calls"] += 1
        if state["calls"] == 1:
            return []
        return full_snapshot

    async def _fake_hydrate(**_kwargs):  # type: ignore[no-untyped-def]
        state["hydrates"] += 1
        return full_snapshot

    monkeypatch.setattr(
        trading_ws.market_data_runtime,
        "get_recent_bars",
        _fake_get_recent_bars,
    )
    monkeypatch.setattr(trading_ws, "reserve_market_data_refresh_slot", lambda *_args: True)
    monkeypatch.setattr(
        trading_ws,
        "_hydrate_bar_snapshot_from_provider",
        _fake_hydrate,
    )
    monkeypatch.setattr(trading_ws, "enqueue_market_data_refresh", lambda **_kwargs: "queued")

    result = asyncio.run(
        trading_ws._load_bar_snapshot(
            market="stocks",
            symbol="AAPL",
            timeframe="1m",
            limit=3,
        )
    )

    assert len(result) == 3
    assert state["hydrates"] == 1


def test_load_bar_snapshot_uses_inline_hydration_for_stale_full_cache(monkeypatch) -> None:
    stale_snapshot = _make_bars(3)
    now = datetime.now(UTC)
    fresh_snapshot = [
        RuntimeBar(
            timestamp=now - timedelta(hours=8),
            open=110.0,
            high=111.0,
            low=109.0,
            close=110.5,
            volume=1000.0,
        ),
        RuntimeBar(
            timestamp=now - timedelta(hours=4),
            open=111.0,
            high=112.0,
            low=110.0,
            close=111.5,
            volume=1000.0,
        ),
        RuntimeBar(
            timestamp=now,
            open=112.0,
            high=113.0,
            low=111.0,
            close=112.5,
            volume=1000.0,
        ),
    ]
    state = {"reads": 0, "hydrates": 0}

    async def _fake_read_bar_window(**_kwargs):  # type: ignore[no-untyped-def]
        state["reads"] += 1
        if state["reads"] == 1:
            return stale_snapshot
        return fresh_snapshot

    async def _fake_finalize_bar_window(**_kwargs):  # type: ignore[no-untyped-def]
        bars = _kwargs.get("bars")
        if isinstance(bars, list):
            return bars
        return []

    async def _fake_hydrate(**_kwargs):  # type: ignore[no-untyped-def]
        state["hydrates"] += 1
        return fresh_snapshot

    monkeypatch.setattr(trading_ws, "_read_bar_window", _fake_read_bar_window)
    monkeypatch.setattr(trading_ws, "_finalize_bar_window", _fake_finalize_bar_window)
    monkeypatch.setattr(trading_ws, "reserve_market_data_refresh_slot", lambda *_args: True)
    monkeypatch.setattr(
        trading_ws,
        "_hydrate_bar_snapshot_from_provider",
        _fake_hydrate,
    )
    monkeypatch.setattr(trading_ws, "enqueue_market_data_refresh", lambda **_kwargs: "queued")

    result = asyncio.run(
        trading_ws._load_bar_snapshot(
            market="crypto",
            symbol="BTCUSD",
            timeframe="4h",
            limit=3,
        )
    )

    assert result == fresh_snapshot
    assert state["hydrates"] == 1
