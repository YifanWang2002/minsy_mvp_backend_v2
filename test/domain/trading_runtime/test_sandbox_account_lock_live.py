from __future__ import annotations

from packages.infra.redis.locks.sandbox_account_lock import SandboxAccountRuntimeLock


async def test_000_accessibility_sandbox_account_lock_fallback_serializes_same_account(
    monkeypatch,
) -> None:
    lock = SandboxAccountRuntimeLock(prefix="pytest:sandbox-lock")

    async def _raise_no_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(lock, "_get_ready_redis", _raise_no_redis)

    first = await lock.acquire("acct-1")
    second = await lock.acquire("acct-1")

    assert first is not None
    assert first.via_fallback is True
    assert second is None

    released = await lock.release(first)
    third = await lock.acquire("acct-1")

    assert released is True
    assert third is not None
