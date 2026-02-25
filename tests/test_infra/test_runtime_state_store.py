from __future__ import annotations

from uuid import uuid4

import pytest

from src.engine.execution.runtime_state_store import RuntimeStateStore
from src.models.redis import close_redis, init_redis


@pytest.mark.asyncio
async def test_runtime_state_store_upsert_and_get() -> None:
    await init_redis()
    deployment_id = uuid4()
    store = RuntimeStateStore(key_prefix=f"test:runtime_state:{uuid4().hex}", ttl_seconds=3600)

    payload = {
        "runtime_status": "running",
        "runtime_reason": "ok",
        "runtime_signal": "OPEN_LONG",
    }
    await store.upsert(deployment_id, payload)
    loaded = await store.get(deployment_id)
    assert isinstance(loaded, dict)
    assert loaded["runtime_status"] == "running"
    assert loaded["runtime_reason"] == "ok"

    await store.clear()
    assert await store.get(deployment_id) is None
    await close_redis()


@pytest.mark.asyncio
async def test_runtime_state_store_upsert_merges_existing_payload() -> None:
    await init_redis()
    deployment_id = uuid4()
    store = RuntimeStateStore(key_prefix=f"test:runtime_state:{uuid4().hex}", ttl_seconds=3600)

    await store.upsert(
        deployment_id,
        {
            "scheduler": {"timeframe_seconds": 60, "last_trigger_bucket": 123},
            "runtime_status": "running",
        },
    )
    await store.upsert(
        deployment_id,
        {"runtime_reason": "ok"},
    )

    loaded = await store.get(deployment_id)
    assert isinstance(loaded, dict)
    assert loaded["runtime_status"] == "running"
    assert loaded["runtime_reason"] == "ok"
    assert loaded["scheduler"]["timeframe_seconds"] == 60
    await store.clear()
    await close_redis()
