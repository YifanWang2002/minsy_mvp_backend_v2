"""Unit tests for realtime WS transport policy state transitions."""

from __future__ import annotations

from packages.domain.trading.services.realtime_transport_policy import (
    RealtimeTransportPolicy,
)


def _make_policy() -> RealtimeTransportPolicy:
    return RealtimeTransportPolicy(
        fallback_poll_seconds=0.5,
        reconcile_seconds=2.0,
        pubsub_probe_base_seconds=1.0,
        pubsub_probe_max_seconds=15.0,
    )


def test_initial_state_prefers_pubsub_when_available() -> None:
    policy = _make_policy()
    state = policy.initial_state(pubsub_available=True, now=10.0)

    assert state.mode == "pubsub"
    assert state.consecutive_pubsub_failures == 0


def test_failure_switches_to_fallback_and_applies_backoff() -> None:
    policy = _make_policy()
    state = policy.initial_state(pubsub_available=True, now=10.0)

    changed = policy.record_pubsub_failure(state, now=20.0, error="pubsub_read_error:RuntimeError")
    assert changed is True
    assert state.mode == "polling_fallback"
    assert state.consecutive_pubsub_failures == 1
    assert state.next_pubsub_probe_at == 21.0

    changed_again = policy.record_pubsub_failure(
        state,
        now=21.0,
        error="pubsub_probe_error:RuntimeError",
    )
    assert changed_again is False
    assert state.consecutive_pubsub_failures == 2
    assert state.next_pubsub_probe_at == 23.0


def test_success_recovers_pubsub_mode_and_resets_failure_count() -> None:
    policy = _make_policy()
    state = policy.initial_state(pubsub_available=False, now=0.0)
    assert state.mode == "polling_fallback"

    changed = policy.record_pubsub_success(state, now=5.0)

    assert changed is True
    assert state.mode == "pubsub"
    assert state.consecutive_pubsub_failures == 0
    assert state.next_pubsub_probe_at == 6.0
    assert state.last_error is None


def test_poll_and_reconcile_gates_follow_intervals() -> None:
    policy = _make_policy()
    state = policy.initial_state(pubsub_available=False, now=0.0)

    assert policy.should_poll_fallback(state, now=0.2) is False
    assert policy.should_poll_fallback(state, now=0.6) is True
    policy.mark_fallback_poll(state, now=0.6)
    assert policy.should_poll_fallback(state, now=0.9) is False
    assert policy.should_poll_fallback(state, now=1.2) is True

    assert policy.should_reconcile(state, now=1.0) is False
    assert policy.should_reconcile(state, now=2.1) is True
