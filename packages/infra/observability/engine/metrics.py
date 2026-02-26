"""Runtime metric snapshots for paper-trading health and alerts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    bar_lag_seconds: float
    signal_lag_seconds: float
    order_reject_rate: float
    queue_lag_seconds: float
    pnl_staleness_seconds: float


def build_runtime_metrics(
    *,
    bar_lag_seconds: float,
    signal_lag_seconds: float,
    order_reject_rate: float,
    queue_lag_seconds: float,
    pnl_staleness_seconds: float,
) -> RuntimeMetrics:
    return RuntimeMetrics(
        bar_lag_seconds=max(0.0, float(bar_lag_seconds)),
        signal_lag_seconds=max(0.0, float(signal_lag_seconds)),
        order_reject_rate=max(0.0, min(1.0, float(order_reject_rate))),
        queue_lag_seconds=max(0.0, float(queue_lag_seconds)),
        pnl_staleness_seconds=max(0.0, float(pnl_staleness_seconds)),
    )
