"""Event-driven backtest engine."""

from packages.domain.backtest.engine import EventDrivenBacktestEngine
from packages.domain.backtest.types import (
    BacktestConfig,
    BacktestEvent,
    BacktestEventType,
    BacktestResult,
    BacktestSummary,
    BacktestTrade,
    EquityPoint,
    PositionSide,
)

_SERVICE_EXPORTS: frozenset[str] = frozenset(
    {
        "BacktestBarLimitExceededError",
        "BacktestJobNotFoundError",
        "BacktestJobReceipt",
        "BacktestJobView",
        "BacktestStrategyNotFoundError",
        "create_backtest_job",
        "execute_backtest_job",
        "execute_backtest_job_with_fresh_session",
        "get_backtest_job_view",
        "schedule_backtest_job",
    }
)


def __getattr__(name: str):
    if name in _SERVICE_EXPORTS:
        from packages.domain.backtest import service as _service

        return getattr(_service, name)
    raise AttributeError(name)

__all__ = [
    "BacktestConfig",
    "BacktestEvent",
    "BacktestEventType",
    "BacktestBarLimitExceededError",
    "BacktestJobNotFoundError",
    "BacktestJobReceipt",
    "BacktestJobView",
    "BacktestResult",
    "BacktestStrategyNotFoundError",
    "BacktestSummary",
    "BacktestTrade",
    "EquityPoint",
    "EventDrivenBacktestEngine",
    "PositionSide",
    "create_backtest_job",
    "execute_backtest_job",
    "execute_backtest_job_with_fresh_session",
    "get_backtest_job_view",
    "schedule_backtest_job",
]
