"""Typed models for event-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class BacktestEventType(StrEnum):
    """Runtime event stream types emitted by the backtest engine."""

    BAR = "bar"
    ENTRY_SIGNAL = "entry_signal"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"


class PositionSide(StrEnum):
    """Position direction."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution controls for one backtest run."""

    initial_capital: float = 100_000.0
    commission_rate: float = 0.0
    slippage_bps: float = 0.0
    record_bar_events: bool = False
    performance_series_max_points: int = 5_000


@dataclass(frozen=True, slots=True)
class BacktestEvent:
    """A single event in the event-driven runtime."""

    type: BacktestEventType
    timestamp: datetime
    bar_index: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One completed trade."""

    side: PositionSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    bars_held: int
    exit_reason: str
    pnl: float
    pnl_pct: float
    commission: float


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Equity snapshot on one bar."""

    timestamp: datetime
    equity: float


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Basic deterministic metrics computed by the core engine."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    final_equity: float
    max_drawdown_pct: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Event-driven backtest result payload."""

    config: BacktestConfig
    summary: BacktestSummary
    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    returns: tuple[float, ...]
    events: tuple[BacktestEvent, ...]
    performance: dict[str, Any]
    started_at: datetime
    finished_at: datetime


def utc_now() -> datetime:
    """UTC timestamp helper."""

    return datetime.now(UTC)
