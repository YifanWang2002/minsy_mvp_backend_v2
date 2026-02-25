from __future__ import annotations

from uuid import uuid4

import pytest

from src.engine.execution.deployment_lock import DeploymentLockLease, DeploymentRuntimeLock
from src.models.redis import close_redis, init_redis


@pytest.mark.asyncio
async def test_deployment_runtime_lock_blocks_double_acquire() -> None:
    await init_redis()
    lock = DeploymentRuntimeLock(prefix=f"test:deployment_lock:{uuid4().hex}")
    deployment_id = uuid4()

    first = await lock.acquire(deployment_id)
    assert first is not None

    second = await lock.acquire(deployment_id)
    assert second is None

    released = await lock.release(first)
    assert released is True

    third = await lock.acquire(deployment_id)
    assert third is not None
    assert await lock.release(third) is True
    await close_redis()


@pytest.mark.asyncio
async def test_deployment_runtime_lock_refresh_requires_matching_token() -> None:
    await init_redis()
    lock = DeploymentRuntimeLock(prefix=f"test:deployment_lock:{uuid4().hex}")
    deployment_id = uuid4()

    lease = await lock.acquire(deployment_id)
    assert lease is not None
    assert await lock.refresh(lease) is True

    wrong_lease = DeploymentLockLease(key=lease.key, token=uuid4().hex, via_fallback=False)
    assert await lock.refresh(wrong_lease) is False

    assert await lock.release(lease) is True
    await close_redis()
