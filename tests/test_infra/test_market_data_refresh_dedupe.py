from __future__ import annotations

from uuid import uuid4

import pytest

from src.config import settings
from src.workers import market_data_tasks


def test_refresh_active_subscriptions_task_dedupes_same_symbol_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    token = uuid4().hex

    monkeypatch.setattr(settings, "market_data_redis_subs_enabled", True)
    monkeypatch.setattr(settings, "market_data_refresh_dedupe_enabled", True)
    monkeypatch.setattr(settings, "market_data_refresh_dedupe_window_seconds", 60)
    monkeypatch.setattr(
        market_data_tasks,
        "_refresh_dedupe_key",
        lambda market, symbol: f"md:test:{token}:dedupe:{market}:{symbol}",
    )
    monkeypatch.setattr(
        market_data_tasks.market_data_runtime,
        "active_subscriptions",
        lambda: (("stocks", "AAPL"),),
    )
    monkeypatch.setattr(
        market_data_tasks.market_data_runtime,
        "record_refresh_scheduler_metrics",
        lambda **_: None,
    )
    monkeypatch.setattr(
        market_data_tasks.refresh_symbol_task,
        "apply_async",
        lambda *, args, queue: calls.append((str(args[0]), str(args[1]))),
    )
    market_data_tasks._REFRESH_DEDUPE_FALLBACK.clear()

    first = market_data_tasks.refresh_active_subscriptions_task()
    second = market_data_tasks.refresh_active_subscriptions_task()

    assert first["scheduled"] == 1
    assert first["deduped"] == 0
    assert first["total"] == 1
    assert second["scheduled"] == 0
    assert second["deduped"] == 1
    assert second["total"] == 1
    assert calls == [("stocks", "AAPL")]
