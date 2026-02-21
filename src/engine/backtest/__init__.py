"""Event-driven backtest engine."""

from src.engine.backtest.engine import EventDrivenBacktestEngine
from src.engine.backtest.service import (
    BacktestBarLimitExceededError,
    BacktestJobNotFoundError,
    BacktestJobReceipt,
    BacktestJobView,
    BacktestStrategyNotFoundError,
    create_backtest_job,
    execute_backtest_job,
    execute_backtest_job_with_fresh_session,
    get_backtest_job_view,
    schedule_backtest_job,
)
from src.engine.backtest.types import (
    BacktestConfig,
    BacktestEvent,
    BacktestEventType,
    BacktestResult,
    BacktestSummary,
    BacktestTrade,
    EquityPoint,
    PositionSide,
)

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
