"""Alpaca stream provider with reconnect loop semantics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from packages.shared_settings.schema.settings import settings


class StreamReconnectError(RuntimeError):
    """Raised when stream reconnect attempts exceed configured threshold."""


StreamContextFactory = Callable[[], AsyncIterator[AsyncIterator[dict[str, Any]]]]
OnEventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ShouldStopCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class StreamStats:
    reconnects: int = 0
    events: int = 0


@asynccontextmanager
async def _empty_stream_context() -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
    async def _iterator() -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}

    yield _iterator()


class AlpacaStreamProvider:
    """Reconnect-capable stream wrapper with testable connection factory."""

    def __init__(
        self,
        *,
        connect_factory: StreamContextFactory | None = None,
        reconnect_base_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._connect_factory = connect_factory or (lambda: _empty_stream_context())
        self._reconnect_base_seconds = (
            reconnect_base_seconds
            if reconnect_base_seconds is not None
            else settings.alpaca_stream_reconnect_base_seconds
        )
        self._max_retries = max_retries if max_retries is not None else settings.alpaca_stream_max_retries
        self._stats = StreamStats()

    @property
    def stats(self) -> StreamStats:
        return self._stats

    async def run(
        self,
        *,
        symbols: list[str],
        on_event: OnEventCallback,
        should_stop: ShouldStopCallback | None = None,
    ) -> None:
        """Run stream forever with reconnects until `should_stop` or retries exhausted."""
        _ = symbols
        retries = 0
        reconnects = 0
        events = 0

        while True:
            if should_stop is not None and should_stop():
                self._stats = StreamStats(reconnects=reconnects, events=events)
                return

            try:
                async with self._connect_factory() as stream:
                    retries = 0
                    async for message in stream:
                        if should_stop is not None and should_stop():
                            self._stats = StreamStats(reconnects=reconnects, events=events)
                            return
                        await on_event(message)
                        events += 1
            except asyncio.CancelledError:
                self._stats = StreamStats(reconnects=reconnects, events=events)
                raise
            except Exception as exc:  # noqa: BLE001
                retries += 1
                reconnects += 1
                if retries > self._max_retries:
                    self._stats = StreamStats(reconnects=reconnects, events=events)
                    raise StreamReconnectError(
                        f"Stream reconnect exceeded max retries={self._max_retries}"
                    ) from exc
                await asyncio.sleep(self._reconnect_base_seconds * retries)
