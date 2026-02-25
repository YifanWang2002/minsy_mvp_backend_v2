"""Alert evaluation for runtime metric thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.engine.observability.metrics import RuntimeMetrics

AlertLevel = Literal["info", "warning", "critical"]


@dataclass(frozen=True, slots=True)
class AlertEvent:
    code: str
    level: AlertLevel
    message: str


def evaluate_alerts(
    *,
    metrics: RuntimeMetrics,
    max_bar_lag_seconds: float = 10.0,
    max_signal_lag_seconds: float = 10.0,
    max_order_reject_rate: float = 0.25,
    max_queue_lag_seconds: float = 30.0,
    max_pnl_staleness_seconds: float = 60.0,
) -> list[AlertEvent]:
    alerts: list[AlertEvent] = []
    if metrics.bar_lag_seconds > max_bar_lag_seconds:
        alerts.append(
            AlertEvent(
                code="BAR_LAG_HIGH",
                level="warning",
                message=f"Bar lag is {metrics.bar_lag_seconds:.2f}s.",
            )
        )
    if metrics.signal_lag_seconds > max_signal_lag_seconds:
        alerts.append(
            AlertEvent(
                code="SIGNAL_LAG_HIGH",
                level="warning",
                message=f"Signal lag is {metrics.signal_lag_seconds:.2f}s.",
            )
        )
    if metrics.order_reject_rate > max_order_reject_rate:
        alerts.append(
            AlertEvent(
                code="ORDER_REJECT_RATE_HIGH",
                level="critical",
                message=f"Order reject rate is {metrics.order_reject_rate:.2%}.",
            )
        )
    if metrics.queue_lag_seconds > max_queue_lag_seconds:
        alerts.append(
            AlertEvent(
                code="QUEUE_LAG_HIGH",
                level="warning",
                message=f"Queue lag is {metrics.queue_lag_seconds:.2f}s.",
            )
        )
    if metrics.pnl_staleness_seconds > max_pnl_staleness_seconds:
        alerts.append(
            AlertEvent(
                code="PNL_STALE",
                level="warning",
                message=f"PnL staleness is {metrics.pnl_staleness_seconds:.2f}s.",
            )
        )
    return alerts
