from __future__ import annotations

from dataclasses import dataclass, field

from packages.domain.trading.runtime.order_state_machine import (
    apply_order_status_transition,
    can_transition,
    normalize_order_status,
)


@dataclass
class _OrderStub:
    status: str
    metadata_: dict[str, object] = field(default_factory=dict)


def test_000_accessibility_normalize_status_aliases() -> None:
    assert normalize_order_status("cancelled") == "canceled"
    assert normalize_order_status("pending") == "pending_new"
    assert normalize_order_status("done_for_day") == "expired"


def test_010_order_state_machine_happy_path_transition() -> None:
    order = _OrderStub(status="new")
    transition = apply_order_status_transition(
        order,
        target_status="accepted",
        reason="provider_ack",
        extra_metadata={"provider_status": "accepted"},
    )
    assert order.status == "accepted"
    assert transition["from"] == "new"
    assert transition["to"] == "accepted"
    assert order.metadata_["provider_status"] == "accepted"


def test_020_order_state_machine_rejects_invalid_transition() -> None:
    assert can_transition("filled", "accepted") is False
    order = _OrderStub(status="filled")
    try:
        apply_order_status_transition(order, target_status="accepted", reason="invalid")
    except ValueError as exc:
        assert "Invalid order status transition" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid transition")
