from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.engine.execution.order_state_machine import (
    apply_order_status_transition,
    can_transition,
    normalize_order_status,
)


@dataclass
class _OrderStub:
    status: str
    metadata_: dict[str, Any] = field(default_factory=dict)


def test_order_state_machine_happy_path_transitions() -> None:
    order = _OrderStub(status="new", metadata_={})
    apply_order_status_transition(order, target_status="accepted", reason="submit_ok")
    assert order.status == "accepted"
    apply_order_status_transition(order, target_status="filled", reason="fill_ok")
    assert order.status == "filled"
    transitions = order.metadata_.get("state_transitions")
    assert isinstance(transitions, list)
    assert transitions[0]["from"] == "new"
    assert transitions[0]["to"] == "accepted"
    assert transitions[1]["to"] == "filled"


def test_order_state_machine_rejects_invalid_transition() -> None:
    order = _OrderStub(status="filled", metadata_={})
    assert can_transition("filled", "accepted") is False
    with pytest.raises(ValueError):
        apply_order_status_transition(order, target_status="accepted", reason="should_fail")


def test_order_state_machine_normalizes_pending_and_expired_status() -> None:
    assert normalize_order_status("pending") == "pending_new"
    assert normalize_order_status("done_for_day") == "expired"
