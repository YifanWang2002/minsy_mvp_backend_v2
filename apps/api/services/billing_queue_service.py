"""Queue dispatch helpers for billing reliability workflows."""

from __future__ import annotations

from typing import Any


def enqueue_reconcile_billing_usage_event(payload: dict[str, Any]) -> str | None:
    from packages.infra.queue.publishers import (
        enqueue_reconcile_billing_usage_event as _enqueue,
    )

    return _enqueue(payload)
