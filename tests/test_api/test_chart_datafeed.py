"""Unit coverage for TradingView chart datafeed routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.routes import chart_datafeed
from packages.domain.market_data.runtime import RuntimeBar


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _FakeDbSession:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        return _FakeResult(self.rows)


@asynccontextmanager
async def _noop_lifespan(_):
    yield


@pytest.fixture()
def app(monkeypatch):
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr("apps.api.main.lifespan", _noop_lifespan)
        test_app = create_app()
        yield test_app


@pytest.fixture()
def client(app):
    return TestClient(app)


def _override_dependencies(*, app, user, db):
    async def _override_user():
        return user

    async def _override_db():
        yield db

    app.dependency_overrides[chart_datafeed.get_current_user] = _override_user
    app.dependency_overrides[chart_datafeed.get_db] = _override_db


def _make_bar(*, minutes: int, price: float) -> RuntimeBar:
    return RuntimeBar(
        timestamp=datetime(2026, 1, 1, 0, minutes, tzinfo=UTC),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=1000 + minutes,
    )


def test_resolve_symbol_returns_featured_descriptor(client, app):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id="user-1"),
        db=_FakeDbSession(),
    )

    response = client.get(
        "/api/v1/chart-datafeed/resolveSymbol",
        params={"symbolName": "CRYPTO:BTCUSD"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "BTCUSD"
    assert payload["full_name"] == "CRYPTO:BTCUSD"
    assert payload["minsy_market"] == "crypto"
    assert payload["data_status"] == "streaming"


def test_resolve_symbol_returns_404_for_unknown_symbol(client, app):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id="user-1"),
        db=_FakeDbSession(rows=[]),
    )

    response = client.get(
        "/api/v1/chart-datafeed/resolveSymbol",
        params={"symbolName": "UNKNOWN"},
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["code"] == "UNKNOWN_SYMBOL"


def test_resolution_mapping_helpers_cover_intraday_and_daily():
    assert chart_datafeed._timeframe_to_resolution("1m") == "1"
    assert chart_datafeed._timeframe_to_resolution("4h") == "240"
    assert chart_datafeed._timeframe_to_resolution("1d") == "1D"

    assert chart_datafeed._resolution_to_timeframe("5") == "5m"
    assert chart_datafeed._resolution_to_timeframe("60") == "1h"
    assert chart_datafeed._resolution_to_timeframe("1D") == "1d"


def test_slice_history_window_backfills_countback_and_sets_next_time():
    bars = [_make_bar(minutes=index, price=100 + index) for index in range(6)]
    from_dt = datetime(2026, 1, 1, 0, 3, tzinfo=UTC)
    to_dt = datetime(2026, 1, 1, 0, 6, tzinfo=UTC)

    sliced, next_time = chart_datafeed._slice_history_window(
        bars=bars,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=4,
    )

    assert [bar.timestamp.minute for bar in sliced] == [2, 3, 4, 5]
    assert next_time == int(bars[2].timestamp.timestamp() * 1000)


def test_slice_history_window_keeps_full_requested_range_when_it_exceeds_countback():
    bars = [_make_bar(minutes=index, price=100 + index) for index in range(10)]
    from_dt = datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
    to_dt = datetime(2026, 1, 1, 0, 8, tzinfo=UTC)

    sliced, next_time = chart_datafeed._slice_history_window(
        bars=bars,
        from_dt=from_dt,
        to_dt=to_dt,
        count_back=3,
    )

    assert [bar.timestamp.minute for bar in sliced] == [2, 3, 4, 5, 6, 7]
    assert next_time is None


def test_provider_history_since_looks_back_for_countback_not_just_visible_window():
    step = timedelta(minutes=15)
    from_dt = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    to_dt = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)

    since = chart_datafeed._provider_history_since(
        from_dt=from_dt,
        to_dt=to_dt,
        step=step,
        limit=300,
    )

    assert since == datetime(2025, 12, 29, 8, 0, tzinfo=UTC)


def test_merge_history_groups_keeps_older_local_bars_and_newer_runtime_bars():
    local = [_make_bar(minutes=index, price=100 + index) for index in range(4)]
    runtime = [_make_bar(minutes=index, price=200 + index) for index in range(3, 6)]

    merged = chart_datafeed._merge_history_groups(
        groups=[local, runtime],
        limit=10,
    )

    assert [bar.timestamp.minute for bar in merged] == [0, 1, 2, 3, 4, 5]
    assert merged[3].open == 203


def test_get_bars_returns_nodata_for_empty_history_window(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id="user-1"),
        db=_FakeDbSession(),
    )
    async def _fake_load_chart_history(**kwargs):
        del kwargs
        return []

    monkeypatch.setattr(chart_datafeed, "_load_chart_history", _fake_load_chart_history)

    response = client.get(
        "/api/v1/chart-datafeed/getBars",
        params={
          "symbol": "CRYPTO:BTCUSD",
          "resolution": "15",
          "from": int(datetime(2026, 1, 1, 0, 15, tzinfo=UTC).timestamp()),
          "to": int(datetime(2026, 1, 1, 0, 30, tzinfo=UTC).timestamp()),
          "countBack": 50,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bars"] == []
    assert payload["meta"]["noData"] is True
    assert "nextTime" not in payload["meta"]


def test_get_bars_respects_countback_and_backfills_when_range_is_shorter(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id="user-1"),
        db=_FakeDbSession(),
    )
    bars = [_make_bar(minutes=index, price=200 + index) for index in range(8)]

    async def _fake_load_chart_history(**kwargs):
        del kwargs
        return bars

    monkeypatch.setattr(chart_datafeed, "_load_chart_history", _fake_load_chart_history)

    response = client.get(
        "/api/v1/chart-datafeed/getBars",
        params={
          "symbol": "NASDAQ:AAPL",
          "resolution": "5",
          "from": int(datetime(2026, 1, 1, 0, 6, tzinfo=UTC).timestamp()),
          "to": int(datetime(2026, 1, 1, 0, 8, tzinfo=UTC).timestamp()),
          "countBack": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [bar["time"] for bar in payload["bars"]] == [
        int(datetime(2026, 1, 1, 0, 5, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 6, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 7, tzinfo=UTC).timestamp() * 1000),
    ]
    assert payload["meta"]["noData"] is False
    assert "nextTime" not in payload["meta"]


def test_get_bars_returns_full_requested_range_even_when_range_exceeds_countback(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id="user-1"),
        db=_FakeDbSession(),
    )
    bars = [_make_bar(minutes=index, price=300 + index) for index in range(10)]

    async def _fake_load_chart_history(**kwargs):
        del kwargs
        return bars

    monkeypatch.setattr(chart_datafeed, "_load_chart_history", _fake_load_chart_history)

    response = client.get(
        "/api/v1/chart-datafeed/getBars",
        params={
          "symbol": "NASDAQ:AAPL",
          "resolution": "1",
          "from": int(datetime(2026, 1, 1, 0, 2, tzinfo=UTC).timestamp()),
          "to": int(datetime(2026, 1, 1, 0, 8, tzinfo=UTC).timestamp()),
          "countBack": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [bar["time"] for bar in payload["bars"]] == [
        int(datetime(2026, 1, 1, 0, 2, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 3, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 4, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 5, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 6, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2026, 1, 1, 0, 7, tzinfo=UTC).timestamp() * 1000),
    ]
    assert payload["meta"]["noData"] is False
    assert "nextTime" not in payload["meta"]
