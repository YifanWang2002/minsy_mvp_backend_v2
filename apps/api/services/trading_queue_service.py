"""Queue dispatch helpers shared by API/MCP/IM entrypoints.

This module centralizes worker enqueue calls so entrypoint layers do not
import worker task modules directly.
"""

from __future__ import annotations

from uuid import UUID


def enqueue_paper_trading_runtime(deployment_id: UUID | str) -> str:
    from packages.infra.queue.publishers import enqueue_paper_trading_runtime as _enqueue

    return _enqueue(deployment_id)


def enqueue_market_data_refresh(*, market: str, symbol: str) -> str:
    from packages.infra.queue.publishers import enqueue_market_data_refresh as _enqueue

    return _enqueue(market=market, symbol=symbol)


def enqueue_execute_approved_open(request_id: UUID | str) -> str:
    from packages.infra.queue.publishers import enqueue_execute_approved_open as _enqueue

    return _enqueue(request_id)
