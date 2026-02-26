"""Order status transition guardrails for runtime execution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "new": {"pending_new", "accepted", "partially_filled", "filled", "canceled", "rejected", "expired"},
    "pending_new": {"accepted", "partially_filled", "filled", "canceled", "rejected", "expired"},
    "accepted": {"partially_filled", "filled", "canceled", "rejected", "expired"},
    "partially_filled": {"partially_filled", "filled", "canceled", "rejected", "expired"},
    "filled": set(),
    "canceled": set(),
    "rejected": set(),
    "expired": set(),
}


class _OrderLike(Protocol):
    status: str
    metadata_: dict[str, Any]


def normalize_order_status(status: str) -> str:
    value = status.strip().lower()
    if value == "cancelled":
        return "canceled"
    if value in {"pending", "pending_new"}:
        return "pending_new"
    if value in {"done_for_day", "expired"}:
        return "expired"
    if value in _ALLOWED_TRANSITIONS:
        return value
    return "new"


def can_transition(current_status: str, target_status: str) -> bool:
    current = normalize_order_status(current_status)
    target = normalize_order_status(target_status)
    if current == target:
        return True
    return target in _ALLOWED_TRANSITIONS.get(current, set())


def apply_order_status_transition(
    order: _OrderLike,
    *,
    target_status: str,
    reason: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transition one order to target status and append transition history metadata."""
    current = normalize_order_status(order.status)
    target = normalize_order_status(target_status)
    if not can_transition(current, target):
        raise ValueError(f"Invalid order status transition: {current} -> {target}")

    metadata = dict(order.metadata_) if isinstance(order.metadata_, dict) else {}
    transitions = metadata.get("state_transitions")
    transition_rows = list(transitions) if isinstance(transitions, list) else []
    transition_record = {
        "from": current,
        "to": target,
        "reason": reason,
        "ts": datetime.now(UTC).isoformat(),
    }
    transition_rows.append(
        transition_record
    )
    metadata["state_transitions"] = transition_rows[-100:]
    metadata["last_state_reason"] = reason
    metadata["last_state_updated_at"] = datetime.now(UTC).isoformat()
    if extra_metadata:
        metadata.update(extra_metadata)

    order.status = target
    order.metadata_ = metadata
    return transition_record
