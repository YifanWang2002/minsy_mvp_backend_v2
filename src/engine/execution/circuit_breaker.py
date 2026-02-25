"""Async circuit-breaker and retry helpers for broker requests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable, Generic, Literal, TypeVar

from src.config import settings

T = TypeVar("T")


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker is open."""


@dataclass(frozen=True, slots=True)
class CircuitBreakerSnapshot:
    name: str
    state: Literal["closed", "open"]
    consecutive_failures: int
    opened_until: datetime | None


class AsyncCircuitBreaker:
    """Lightweight async circuit breaker with open/close states."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, int(failure_threshold))
        self._recovery_timeout_seconds = max(0.1, float(recovery_timeout_seconds))
        self._consecutive_failures = 0
        self._opened_until: datetime | None = None
        self._lock = asyncio.Lock()

    async def ensure_can_execute(self) -> None:
        async with self._lock:
            if self._opened_until is None:
                return
            now = datetime.now(UTC)
            if now >= self._opened_until:
                self._opened_until = None
                self._consecutive_failures = 0
                return
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self._name}' is open until {self._opened_until.isoformat()}.",
            )

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._opened_until = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._opened_until = datetime.now(UTC) + timedelta(
                    seconds=self._recovery_timeout_seconds,
                )

    async def snapshot(self) -> CircuitBreakerSnapshot:
        async with self._lock:
            state: Literal["closed", "open"] = (
                "open" if self._opened_until is not None else "closed"
            )
            return CircuitBreakerSnapshot(
                name=self._name,
                state=state,
                consecutive_failures=self._consecutive_failures,
                opened_until=self._opened_until,
            )


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    breaker: AsyncCircuitBreaker,
    max_attempts: int,
    base_backoff_seconds: float,
) -> T:
    attempts = max(1, int(max_attempts))
    backoff_base = max(0.0, float(base_backoff_seconds))

    for attempt in range(1, attempts + 1):
        await breaker.ensure_can_execute()
        try:
            result = await operation()
        except Exception:
            await breaker.record_failure()
            if attempt >= attempts:
                raise
            if backoff_base > 0:
                await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
            continue
        await breaker.record_success()
        return result

    raise RuntimeError("Retry loop exhausted unexpectedly.")


_broker_request_circuit_breaker = AsyncCircuitBreaker(
    name="broker_request",
    failure_threshold=settings.paper_trading_circuit_breaker_failure_threshold,
    recovery_timeout_seconds=settings.paper_trading_circuit_breaker_recovery_seconds,
)


def get_broker_request_circuit_breaker() -> AsyncCircuitBreaker:
    return _broker_request_circuit_breaker


def reset_broker_request_circuit_breaker() -> None:
    global _broker_request_circuit_breaker
    _broker_request_circuit_breaker = AsyncCircuitBreaker(
        name="broker_request",
        failure_threshold=settings.paper_trading_circuit_breaker_failure_threshold,
        recovery_timeout_seconds=settings.paper_trading_circuit_breaker_recovery_seconds,
    )
