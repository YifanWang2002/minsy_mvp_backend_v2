"""Unit tests for trading WS transport tuning settings."""

from __future__ import annotations

from packages.shared_settings.schema.settings import settings


def test_trading_ws_transport_defaults_are_positive() -> None:
    assert settings.trading_ws_pubsub_wait_seconds > 0
    assert settings.trading_ws_fallback_poll_seconds > 0
    assert settings.trading_ws_reconcile_seconds > 0
    assert settings.trading_ws_pubsub_probe_base_seconds > 0
    assert settings.trading_ws_pubsub_probe_max_seconds >= settings.trading_ws_pubsub_probe_base_seconds


def test_trading_ws_transport_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setattr(settings, "trading_ws_pubsub_wait_seconds", 0.3)
    monkeypatch.setattr(settings, "trading_ws_fallback_poll_seconds", 0.4)
    monkeypatch.setattr(settings, "trading_ws_reconcile_seconds", 1.7)
    monkeypatch.setattr(settings, "trading_ws_pubsub_probe_base_seconds", 0.8)
    monkeypatch.setattr(settings, "trading_ws_pubsub_probe_max_seconds", 12.0)

    assert settings.trading_ws_pubsub_wait_seconds == 0.3
    assert settings.trading_ws_fallback_poll_seconds == 0.4
    assert settings.trading_ws_reconcile_seconds == 1.7
    assert settings.trading_ws_pubsub_probe_base_seconds == 0.8
    assert settings.trading_ws_pubsub_probe_max_seconds == 12.0
