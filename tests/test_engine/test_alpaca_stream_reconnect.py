from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from src.engine.market_data.providers.alpaca_stream import (
    AlpacaStreamProvider,
    StreamReconnectError,
)


def _build_connect_factory(
    plan: list[Exception | list[dict[str, str]]],
):
    state = {"attempt": 0}

    @asynccontextmanager
    async def _connect() -> AsyncIterator[AsyncIterator[dict[str, str]]]:
        if state["attempt"] >= len(plan):
            item: Exception | list[dict[str, str]] = RuntimeError("unexpected extra connect")
        else:
            item = plan[state["attempt"]]
        state["attempt"] += 1

        if isinstance(item, Exception):
            raise item

        async def _stream() -> AsyncIterator[dict[str, str]]:
            for event in item:
                yield event

        yield _stream()

    return _connect


@pytest.mark.asyncio
async def test_stream_provider_reconnects_then_recovers() -> None:
    received: list[dict[str, str]] = []
    provider = AlpacaStreamProvider(
        connect_factory=_build_connect_factory(
            [
                RuntimeError("disconnect"),
                [{"symbol": "AAPL"}, {"symbol": "TSLA"}],
            ]
        ),
        reconnect_base_seconds=0.001,
        max_retries=3,
    )

    async def _on_event(message: dict[str, str]) -> None:
        received.append(message)

    await provider.run(
        symbols=["AAPL", "TSLA"],
        on_event=_on_event,
        should_stop=lambda: len(received) >= 2,
    )

    assert [item["symbol"] for item in received] == ["AAPL", "TSLA"]
    assert provider.stats.reconnects == 1
    assert provider.stats.events == 2


@pytest.mark.asyncio
async def test_stream_provider_raises_when_retries_exhausted() -> None:
    provider = AlpacaStreamProvider(
        connect_factory=_build_connect_factory(
            [
                RuntimeError("disconnect-1"),
                RuntimeError("disconnect-2"),
                RuntimeError("disconnect-3"),
            ]
        ),
        reconnect_base_seconds=0.001,
        max_retries=1,
    )

    async def _on_event(_: dict[str, str]) -> None:
        return None

    with pytest.raises(StreamReconnectError):
        await provider.run(
            symbols=["AAPL"],
            on_event=_on_event,
            should_stop=lambda: False,
        )
