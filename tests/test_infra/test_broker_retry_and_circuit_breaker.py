from __future__ import annotations

import pytest

from src.engine.execution.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerOpenError,
    execute_with_retry,
)


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure() -> None:
    breaker = AsyncCircuitBreaker(
        name="test-breaker",
        failure_threshold=3,
        recovery_timeout_seconds=60,
    )
    attempts = 0

    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("temporary error")
        return "ok"

    result = await execute_with_retry(
        flaky_operation,
        breaker=breaker,
        max_attempts=3,
        base_backoff_seconds=0,
    )
    assert result == "ok"
    assert attempts == 2

    snapshot = await breaker.snapshot()
    assert snapshot.state == "closed"
    assert snapshot.consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_blocks_new_calls() -> None:
    breaker = AsyncCircuitBreaker(
        name="test-breaker",
        failure_threshold=2,
        recovery_timeout_seconds=60,
    )

    async def fail_operation() -> str:
        raise RuntimeError("permanent error")

    with pytest.raises(RuntimeError):
        await execute_with_retry(
            fail_operation,
            breaker=breaker,
            max_attempts=1,
            base_backoff_seconds=0,
        )

    with pytest.raises(RuntimeError):
        await execute_with_retry(
            fail_operation,
            breaker=breaker,
            max_attempts=1,
            base_backoff_seconds=0,
        )

    snapshot = await breaker.snapshot()
    assert snapshot.state == "open"
    assert snapshot.consecutive_failures >= 2

    with pytest.raises(CircuitBreakerOpenError):
        await execute_with_retry(
            fail_operation,
            breaker=breaker,
            max_attempts=1,
            base_backoff_seconds=0,
        )
