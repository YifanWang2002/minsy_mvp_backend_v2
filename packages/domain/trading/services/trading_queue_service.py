"""Queue dispatch helpers shared by API/MCP/IM entrypoints."""

from __future__ import annotations

from uuid import UUID

from packages.infra.queue.publishers import (
    enqueue_execute_approved_open as _enqueue_execute_approved_open,
)
from packages.infra.queue.publishers import (
    enqueue_market_data_refresh as _enqueue_market_data_refresh,
)
from packages.infra.queue.publishers import (
    enqueue_paper_trading_runtime as _enqueue_paper_trading_runtime,
)


def enqueue_paper_trading_runtime(deployment_id: UUID | str) -> str:
    return _enqueue_paper_trading_runtime(deployment_id)


def enqueue_market_data_refresh(*, market: str, symbol: str) -> str:
    return _enqueue_market_data_refresh(market=market, symbol=symbol)


def enqueue_execute_approved_open(request_id: UUID | str) -> str | None:
    return _enqueue_execute_approved_open(request_id)
