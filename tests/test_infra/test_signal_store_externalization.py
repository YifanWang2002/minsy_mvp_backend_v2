from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.engine.execution.signal_store import SignalRecord, SignalStore
from src.models.redis import close_redis, init_redis


@pytest.mark.asyncio
async def test_signal_store_persists_to_redis_and_can_be_reloaded() -> None:
    await init_redis()
    deployment_id = uuid4()
    prefix = f"test:signals:{uuid4().hex}"
    index_key = f"{prefix}:index"
    store = SignalStore(max_per_deployment=10, redis_prefix=prefix, redis_index_key=index_key)

    await store.add_async(
        SignalRecord(
            deployment_id=deployment_id,
            signal="OPEN_LONG",
            symbol="BTCUSD",
            timeframe="1m",
            bar_time=datetime(2026, 2, 21, 5, 33, tzinfo=UTC),
            reason="entry",
            metadata={"x": 1},
        )
    )
    await store.add_async(
        SignalRecord(
            deployment_id=deployment_id,
            signal="NOOP",
            symbol="BTCUSD",
            timeframe="1m",
            bar_time=datetime(2026, 2, 21, 5, 34, tzinfo=UTC),
            reason="hold",
            metadata={"x": 2},
        )
    )

    reloaded_store = SignalStore(max_per_deployment=10, redis_prefix=prefix, redis_index_key=index_key)
    rows = await reloaded_store.list_recent_async(deployment_id, limit=10)
    assert len(rows) == 2
    assert rows[0].signal == "OPEN_LONG"
    assert rows[1].signal == "NOOP"

    await reloaded_store.clear_async()
    await close_redis()
