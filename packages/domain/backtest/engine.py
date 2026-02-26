"""Event-driven backtest engine for strategy DSL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from packages.domain.backtest.condition import (
    ConditionEvaluator,
    compile_condition,
    evaluate_condition_series,
)
from packages.domain.backtest.factors import prepare_backtest_frame
from packages.domain.backtest.types import (
    BacktestConfig,
    BacktestEvent,
    BacktestEventType,
    BacktestResult,
    BacktestSummary,
    BacktestTrade,
    EquityPoint,
    PositionSide,
    utc_now,
)
from packages.domain.backtest.performance import build_quantstats_performance
from packages.domain.strategy.models import ParsedStrategyDsl

_LOOKBACK_KEYS = {
    "period",
    "length",
    "fast",
    "slow",
    "signal",
    "k_period",
    "k_smooth",
    "d_period",
    "fastk_period",
    "slowk_period",
    "slowd_period",
}


@dataclass(slots=True)
class _OpenPosition:
    side: PositionSide
    entry_index: int
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_price: float | None
    take_price: float | None


@dataclass(frozen=True, slots=True)
class _CompiledSideRules:
    entry: ConditionEvaluator | None
    entry_signal: np.ndarray[Any, np.dtype[np.bool_]] | None
    signal_exits: tuple[
        tuple[str, ConditionEvaluator, np.ndarray[Any, np.dtype[np.bool_]]],
        ...,
    ]


class EventDrivenBacktestEngine:
    """Run DSL strategy rules in an event-driven bar loop."""

    def __init__(
        self,
        *,
        strategy: ParsedStrategyDsl,
        data: pd.DataFrame,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.frame = prepare_backtest_frame(data, strategy=strategy)

        self._events: list[BacktestEvent] = []
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[EquityPoint] = []

        self._cash = float(self.config.initial_capital)
        self._position: _OpenPosition | None = None
        self._timestamps = self._build_timestamp_cache()
        self._compiled_side_rules = self._build_compiled_side_rules()

    def run(self) -> BacktestResult:
        """Execute the backtest and return deterministic runtime artifacts."""

        started_at = utc_now()
        warmup = self._infer_warmup_bars()

        for bar_index in range(warmup, len(self.frame)):
            timestamp = self._timestamp_at(bar_index)
            if self.config.record_bar_events:
                self._emit(
                    BacktestEventType.BAR,
                    timestamp=timestamp,
                    bar_index=bar_index,
                    payload={},
                )

            if self._position is not None:
                exit_payload = self._check_exit(bar_index)
                if exit_payload is not None:
                    reason, exit_price = exit_payload
                    self._close_position(
                        bar_index=bar_index,
                        reason=reason,
                        exit_price=exit_price,
                    )

            if self._position is None:
                self._maybe_open_position(bar_index)
            elif self._equity_value(bar_index) <= 0:
                close_price = float(self.frame.iloc[bar_index]["close"])
                self._close_position(
                    bar_index=bar_index,
                    reason="liquidation",
                    exit_price=close_price,
                )

            self._record_equity(bar_index)

        if self._position is not None:
            last_index = len(self.frame) - 1
            close_price = float(self.frame.iloc[last_index]["close"])
            self._close_position(
                bar_index=last_index,
                reason="end_of_data",
                exit_price=close_price,
            )
            self._record_equity(last_index)

        summary = self._build_summary()
        returns = self._equity_returns()
        return_timestamps = [point.timestamp for point in self._equity_curve[1:]]
        performance = build_quantstats_performance(
            returns=returns,
            timestamps=return_timestamps,
            max_series_points=self.config.performance_series_max_points,
        )
        finished_at = utc_now()
        return BacktestResult(
            config=self.config,
            summary=summary,
            trades=tuple(self._trades),
            equity_curve=tuple(self._equity_curve),
            returns=tuple(returns),
            events=tuple(self._events),
            performance=performance,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _emit(
        self,
        event_type: BacktestEventType,
        *,
        timestamp: datetime,
        bar_index: int,
        payload: dict[str, Any],
    ) -> None:
        self._events.append(
            BacktestEvent(
                type=event_type,
                timestamp=timestamp,
                bar_index=bar_index,
                payload=payload,
            )
        )

    def _infer_warmup_bars(self) -> int:
        max_lookback = 1
        for factor in self.strategy.factors.values():
            for key, value in factor.params.items():
                if key not in _LOOKBACK_KEYS:
                    continue
                if isinstance(value, int | float) and not isinstance(value, bool):
                    max_lookback = max(max_lookback, int(value))
        return max(2, max_lookback + 2)

    def _maybe_open_position(self, bar_index: int) -> None:
        for side in (PositionSide.LONG, PositionSide.SHORT):
            side_payload = self._side_payload(side)
            if side_payload is None:
                continue

            compiled_rules = self._compiled_side_rules.get(side)
            if (
                compiled_rules is None
                or compiled_rules.entry is None
                or compiled_rules.entry_signal is None
                or bar_index >= len(compiled_rules.entry_signal)
            ):
                continue

            if not bool(compiled_rules.entry_signal[bar_index]):
                continue

            row = self.frame.iloc[bar_index]
            close_price = float(row["close"])
            entry_price = self._apply_entry_slippage(close_price, side)
            quantity = self._resolve_quantity(
                side_payload=side_payload,
                entry_price=entry_price,
            )
            if quantity <= 0:
                continue

            stop_price, take_price = self._resolve_position_stops(
                side_payload=side_payload,
                side=side,
                entry_price=entry_price,
                bar_index=bar_index,
            )

            timestamp = self._timestamp_at(bar_index)
            self._emit(
                BacktestEventType.ENTRY_SIGNAL,
                timestamp=timestamp,
                bar_index=bar_index,
                payload={"side": side.value},
            )
            self._position = _OpenPosition(
                side=side,
                entry_index=bar_index,
                entry_time=timestamp,
                entry_price=entry_price,
                quantity=quantity,
                stop_price=stop_price,
                take_price=take_price,
            )
            self._emit(
                BacktestEventType.POSITION_OPENED,
                timestamp=timestamp,
                bar_index=bar_index,
                payload={
                    "side": side.value,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "stop_price": stop_price,
                    "take_price": take_price,
                },
            )
            return

    def _check_exit(self, bar_index: int) -> tuple[str, float] | None:
        if self._position is None:
            return None

        side_payload = self._side_payload(self._position.side)
        if side_payload is None:
            return None

        row = self.frame.iloc[bar_index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        stop_price = self._position.stop_price
        if stop_price is not None:
            stop_hit = (
                (self._position.side == PositionSide.LONG and low <= stop_price)
                or (self._position.side == PositionSide.SHORT and high >= stop_price)
            )
            if stop_hit:
                return ("stop_loss", stop_price)

        take_price = self._position.take_price
        if take_price is not None:
            take_hit = (
                (self._position.side == PositionSide.LONG and high >= take_price)
                or (self._position.side == PositionSide.SHORT and low <= take_price)
            )
            if take_hit:
                return ("take_profit", take_price)

        exits = side_payload.get("exits", [])
        if not isinstance(exits, list):
            return None

        compiled_rules = self._compiled_side_rules.get(self._position.side)
        if compiled_rules is None:
            return None

        for reason, _condition_eval, signal in compiled_rules.signal_exits:
            if bar_index < len(signal) and bool(signal[bar_index]):
                return (reason, close)

        return None

    def _close_position(
        self,
        *,
        bar_index: int,
        reason: str,
        exit_price: float,
    ) -> None:
        if self._position is None:
            return

        position = self._position
        adjusted_exit_price = self._apply_exit_slippage(exit_price, position.side)
        quantity = position.quantity

        if position.side == PositionSide.LONG:
            gross = (adjusted_exit_price - position.entry_price) * quantity
        else:
            gross = (position.entry_price - adjusted_exit_price) * quantity

        commission = (position.entry_price + adjusted_exit_price) * quantity * self.config.commission_rate
        pnl = gross - commission
        notional = position.entry_price * quantity
        pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
        self._cash += pnl
        if self._cash < 0:
            self._cash = 0.0

        timestamp = self._timestamp_at(bar_index)
        trade = BacktestTrade(
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=adjusted_exit_price,
            quantity=quantity,
            bars_held=max(0, bar_index - position.entry_index),
            exit_reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
        )
        self._trades.append(trade)
        self._position = None

        self._emit(
            BacktestEventType.POSITION_CLOSED,
            timestamp=timestamp,
            bar_index=bar_index,
            payload={
                "side": trade.side.value,
                "exit_reason": trade.exit_reason,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
            },
        )

    def _record_equity(self, bar_index: int) -> None:
        timestamp = self._timestamp_at(bar_index)
        equity = max(0.0, self._equity_value(bar_index))
        point = EquityPoint(timestamp=timestamp, equity=equity)
        if self._equity_curve and self._equity_curve[-1].timestamp == timestamp:
            self._equity_curve[-1] = point
            return
        self._equity_curve.append(point)

    def _resolve_quantity(self, *, side_payload: dict[str, Any], entry_price: float) -> float:
        sizing = side_payload.get("position_sizing", {"mode": "fixed_qty", "qty": 1.0})
        if not isinstance(sizing, dict):
            sizing = {"mode": "fixed_qty", "qty": 1.0}

        mode = str(sizing.get("mode", "fixed_qty")).strip().lower()
        max_affordable_qty = self._max_affordable_qty(entry_price)
        if max_affordable_qty <= 0:
            return 0.0

        if mode == "fixed_qty":
            raw = sizing.get("qty", 1.0)
            if not isinstance(raw, int | float):
                return 0.0
            quantity = float(raw)
            if quantity <= 0:
                return 0.0
            return min(quantity, max_affordable_qty)
        if mode == "fixed_cash":
            raw = sizing.get("cash", 0.0)
            cash = float(raw) if isinstance(raw, int | float) else 0.0
            if cash <= 0:
                return 0.0
            return min(cash, self._cash) / entry_price
        if mode == "pct_equity":
            raw = sizing.get("pct", 0.0)
            pct = float(raw) if isinstance(raw, int | float) else 0.0
            if pct <= 0:
                return 0.0
            notional = self._cash * pct
            if notional <= 0:
                return 0.0
            return min(notional / entry_price, max_affordable_qty)
        return 0.0

    def _max_affordable_qty(self, entry_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        if self._cash <= 0:
            return 0.0
        return self._cash / entry_price

    def _equity_value(self, bar_index: int) -> float:
        equity = self._cash
        if self._position is None:
            return equity

        close_price = float(self.frame.iloc[bar_index]["close"])
        if self._position.side == PositionSide.LONG:
            unrealized = (close_price - self._position.entry_price) * self._position.quantity
        else:
            unrealized = (self._position.entry_price - close_price) * self._position.quantity
        return equity + unrealized

    def _resolve_position_stops(
        self,
        *,
        side_payload: dict[str, Any],
        side: PositionSide,
        entry_price: float,
        bar_index: int,
    ) -> tuple[float | None, float | None]:
        exits = side_payload.get("exits", [])
        if not isinstance(exits, list):
            return (None, None)

        stop_distance: float | None = None
        take_distance: float | None = None

        for exit_rule in exits:
            if not isinstance(exit_rule, dict):
                continue
            rule_type = str(exit_rule.get("type", "")).strip().lower()
            if rule_type == "stop_loss" and stop_distance is None:
                stop_distance = self._distance_from_stop_spec(
                    exit_rule.get("stop"),
                    entry_price=entry_price,
                    bar_index=bar_index,
                )
            elif rule_type == "take_profit" and take_distance is None:
                take_distance = self._distance_from_stop_spec(
                    exit_rule.get("take"),
                    entry_price=entry_price,
                    bar_index=bar_index,
                )
            elif rule_type == "bracket_rr":
                rr_raw = exit_rule.get("risk_reward")
                rr = float(rr_raw) if isinstance(rr_raw, int | float) else 0.0
                bracket_stop = self._distance_from_stop_spec(
                    exit_rule.get("stop"),
                    entry_price=entry_price,
                    bar_index=bar_index,
                )
                bracket_take = self._distance_from_stop_spec(
                    exit_rule.get("take"),
                    entry_price=entry_price,
                    bar_index=bar_index,
                )

                if stop_distance is None and bracket_stop is not None:
                    stop_distance = bracket_stop
                if take_distance is None and bracket_take is not None:
                    take_distance = bracket_take

                if rr > 0:
                    if stop_distance is not None and take_distance is None:
                        take_distance = stop_distance * rr
                    if take_distance is not None and stop_distance is None:
                        stop_distance = take_distance / rr

        stop_price = None
        take_price = None
        if stop_distance is not None:
            stop_price = (
                entry_price - stop_distance
                if side == PositionSide.LONG
                else entry_price + stop_distance
            )
        if take_distance is not None:
            take_price = (
                entry_price + take_distance
                if side == PositionSide.LONG
                else entry_price - take_distance
            )
        return (stop_price, take_price)

    def _distance_from_stop_spec(
        self,
        stop_spec: Any,
        *,
        entry_price: float,
        bar_index: int,
    ) -> float | None:
        if not isinstance(stop_spec, dict):
            return None

        kind = str(stop_spec.get("kind", "")).strip().lower()
        if kind == "points":
            raw = stop_spec.get("value")
            if isinstance(raw, int | float) and raw > 0:
                return float(raw)
            return None
        if kind == "pct":
            raw = stop_spec.get("value")
            if isinstance(raw, int | float) and raw > 0:
                return float(entry_price) * float(raw)
            return None
        if kind == "atr_multiple":
            atr_ref = stop_spec.get("atr_ref")
            multiple = stop_spec.get("multiple")
            if not isinstance(atr_ref, str):
                return None
            if not isinstance(multiple, int | float) or multiple <= 0:
                return None
            if atr_ref not in self.frame.columns:
                return None
            atr_value = self.frame.iloc[bar_index][atr_ref]
            if not isinstance(atr_value, int | float):
                return None
            return float(atr_value) * float(multiple)
        return None

    def _build_summary(self) -> BacktestSummary:
        total = len(self._trades)
        winners = sum(1 for trade in self._trades if trade.pnl > 0)
        losers = sum(1 for trade in self._trades if trade.pnl < 0)
        win_rate = (winners / total * 100.0) if total else 0.0

        final_equity = self._equity_curve[-1].equity if self._equity_curve else self._cash
        total_pnl = final_equity - self.config.initial_capital
        total_return_pct = (
            total_pnl / self.config.initial_capital * 100.0
            if self.config.initial_capital > 0
            else 0.0
        )
        max_drawdown_pct = self._max_drawdown_pct()

        return BacktestSummary(
            total_trades=total,
            winning_trades=winners,
            losing_trades=losers,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            final_equity=final_equity,
            max_drawdown_pct=max_drawdown_pct,
        )

    def _max_drawdown_pct(self) -> float:
        if not self._equity_curve:
            return 0.0
        values = [point.equity for point in self._equity_curve]
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            if peak <= 0:
                continue
            drawdown = (peak - value) / peak * 100.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def _equity_returns(self) -> list[float]:
        if len(self._equity_curve) < 2:
            return []
        values = [point.equity for point in self._equity_curve]
        returns: list[float] = []
        for index in range(1, len(values)):
            prev = values[index - 1]
            curr = values[index]
            if prev == 0:
                returns.append(0.0)
            else:
                returns.append((curr - prev) / prev)
        return returns

    def _side_payload(self, side: PositionSide) -> dict[str, Any] | None:
        trade = self.strategy.trade
        raw = trade.get(side.value)
        if not isinstance(raw, dict):
            return None
        return raw

    def _build_compiled_side_rules(self) -> dict[PositionSide, _CompiledSideRules]:
        compiled: dict[PositionSide, _CompiledSideRules] = {}
        for side in (PositionSide.LONG, PositionSide.SHORT):
            side_payload = self._side_payload(side)
            if side_payload is None:
                compiled[side] = _CompiledSideRules(
                    entry=None,
                    entry_signal=None,
                    signal_exits=(),
                )
                continue

            entry_eval: ConditionEvaluator | None = None
            entry_signal: np.ndarray[Any, np.dtype[np.bool_]] | None = None
            entry = side_payload.get("entry")
            if isinstance(entry, dict):
                condition = entry.get("condition")
                if isinstance(condition, dict):
                    entry_eval = compile_condition(condition)
                    entry_signal = evaluate_condition_series(condition, frame=self.frame)

            signal_exits: list[
                tuple[str, ConditionEvaluator, np.ndarray[Any, np.dtype[np.bool_]]]
            ] = []
            exits = side_payload.get("exits", [])
            if isinstance(exits, list):
                for exit_rule in exits:
                    if not isinstance(exit_rule, dict):
                        continue
                    if str(exit_rule.get("type", "")).strip().lower() != "signal_exit":
                        continue
                    condition = exit_rule.get("condition")
                    if not isinstance(condition, dict):
                        continue
                    reason = str(exit_rule.get("name", "signal_exit")).strip() or "signal_exit"
                    condition_eval = compile_condition(condition)
                    signal = evaluate_condition_series(condition, frame=self.frame)
                    signal_exits.append((reason, condition_eval, signal))

            compiled[side] = _CompiledSideRules(
                entry=entry_eval,
                entry_signal=entry_signal,
                signal_exits=tuple(signal_exits),
            )
        return compiled

    def _build_timestamp_cache(self) -> tuple[datetime, ...]:
        index = self.frame.index
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is None:
                utc_index = index.tz_localize("UTC")
            else:
                utc_index = index.tz_convert("UTC")
            return tuple(timestamp.to_pydatetime() for timestamp in utc_index)

        timestamps: list[datetime] = []
        for index_value in index:
            if isinstance(index_value, pd.Timestamp):
                if index_value.tzinfo is None:
                    timestamps.append(index_value.to_pydatetime().replace(tzinfo=UTC))
                else:
                    timestamps.append(index_value.tz_convert("UTC").to_pydatetime())
            elif isinstance(index_value, datetime):
                if index_value.tzinfo is None:
                    timestamps.append(index_value.replace(tzinfo=UTC))
                else:
                    timestamps.append(index_value.astimezone(UTC))
            else:
                timestamps.append(utc_now())
        return tuple(timestamps)

    def _timestamp_at(self, bar_index: int) -> datetime:
        if 0 <= bar_index < len(self._timestamps):
            return self._timestamps[bar_index]
        return utc_now()

    def _apply_entry_slippage(self, price: float, side: PositionSide) -> float:
        slip = self.config.slippage_bps / 10_000.0
        if side == PositionSide.LONG:
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _apply_exit_slippage(self, price: float, side: PositionSide) -> float:
        slip = self.config.slippage_bps / 10_000.0
        if side == PositionSide.LONG:
            return price * (1.0 - slip)
        return price * (1.0 + slip)
