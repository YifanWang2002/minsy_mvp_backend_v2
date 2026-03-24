"""Event-driven backtest engine."""

from packages.domain.backtest.condition import evaluate_condition_trace_at
from packages.domain.backtest.decision_trace import build_trade_decision_trace
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
        "BacktestJobNotReadyError",
        "BacktestJobNotFoundError",
        "BacktestJobReceipt",
        "BacktestTradeSnapshotInputError",
        "BacktestTradesTruncatedForAllModeError",
        "BacktestJobView",
        "BacktestStrategyNotFoundError",
        "build_backtest_trade_snapshots",
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
    "build_trade_decision_trace",
    "BacktestConfig",
    "BacktestEvent",
    "BacktestEventType",
    "BacktestBarLimitExceededError",
    "BacktestJobNotReadyError",
    "BacktestJobNotFoundError",
    "BacktestJobReceipt",
    "BacktestTradeSnapshotInputError",
    "BacktestTradesTruncatedForAllModeError",
    "BacktestJobView",
    "BacktestResult",
    "BacktestStrategyNotFoundError",
    "BacktestSummary",
    "BacktestTrade",
    "EquityPoint",
    "EventDrivenBacktestEngine",
    "PositionSide",
    "evaluate_condition_trace_at",
    "build_backtest_trade_snapshots",
    "create_backtest_job",
    "execute_backtest_job",
    "execute_backtest_job_with_fresh_session",
    "get_backtest_job_view",
    "schedule_backtest_job",
]
