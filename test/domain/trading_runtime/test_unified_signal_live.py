from __future__ import annotations

from decimal import Decimal

from packages.domain.trading.runtime.unified_signal import (
    build_manual_unified_signal,
    build_runtime_unified_signal,
    reconcile_unified_signal_capabilities,
    unified_signal_to_order_intent,
)


def test_runtime_unified_signal_uses_strategy_entry_order_config() -> None:
    signal = build_runtime_unified_signal(
        signal="OPEN_LONG",
        reason="long_entry_condition",
        symbol="AAPL",
        market="us_stocks",
        timeframe="1m",
        qty=Decimal("2"),
        position_side="flat",
        strategy_payload={
            "trade": {
                "long": {
                    "entry": {
                        "order": {
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 123.45,
                        }
                    }
                }
            }
        },
        mark_price=Decimal("124"),
    )

    assert signal.order_type == "limit"
    assert signal.time_in_force == "day"
    assert signal.limit_price == Decimal("123.45")
    assert signal.side == "buy"


def test_runtime_unified_signal_falls_back_to_mark_price_when_limit_missing() -> None:
    signal = build_runtime_unified_signal(
        signal="OPEN_SHORT",
        reason="short_entry_condition",
        symbol="BTC/USD",
        market="crypto",
        timeframe="1m",
        qty=Decimal("1"),
        position_side="flat",
        strategy_payload={
            "trade": {
                "short": {
                    "entry": {
                        "order": {
                            "type": "limit",
                        }
                    }
                }
            }
        },
        mark_price=Decimal("50000"),
    )

    assert signal.order_type == "limit"
    assert signal.limit_price == Decimal("50000")
    assert signal.metadata.get("limit_price_fallback") == "mark_price"


def test_reconcile_unified_signal_capabilities_downgrades_unsupported_order_semantics() -> None:
    signal = build_manual_unified_signal(
        action="open",
        signal="OPEN_LONG",
        reason="manual_open",
        symbol="AAPL",
        market="us_stocks",
        timeframe="1m",
        qty=Decimal("1"),
        current_position_side="flat",
        mark_price=Decimal("200"),
        payload={
            "order_type": "stop_limit",
            "time_in_force": "day",
            "stop_price": 199,
            "limit_price": 198,
        },
    )

    reconciled, error = reconcile_unified_signal_capabilities(
        unified_signal=signal,
        capabilities={
            "order_types": ["market", "limit"],
            "time_in_force": ["gtc"],
        },
        strict=False,
    )

    assert error is None
    assert reconciled is not None
    assert reconciled.order_type == "market"
    assert reconciled.time_in_force == "gtc"
    assert reconciled.metadata.get("order_type_downgraded_from") == "stop_limit"


def test_manual_unified_signal_supports_flat_payload_fields() -> None:
    signal = build_manual_unified_signal(
        action="open",
        signal="OPEN_SHORT",
        reason="manual_open",
        symbol="ETH/USD",
        market="crypto",
        timeframe="5m",
        qty=Decimal("0.5"),
        current_position_side="flat",
        mark_price=Decimal("1800"),
        payload={
            "order_type": "stop",
            "time_in_force": "ioc",
            "stop_price": "1810",
        },
    )

    assert signal.side == "sell"
    assert signal.order_type == "stop"
    assert signal.time_in_force == "ioc"
    assert signal.stop_price == Decimal("1810")


def test_unified_signal_to_order_intent_preserves_compiled_order_fields() -> None:
    signal = build_manual_unified_signal(
        action="open",
        signal="OPEN_LONG",
        reason="manual_open",
        symbol="SOL/USD",
        market="crypto",
        timeframe="1m",
        qty=Decimal("3"),
        current_position_side="flat",
        mark_price=Decimal("100"),
        payload={
            "order": {
                "type": "limit",
                "time_in_force": "gtc",
                "limit_price": 99.5,
            }
        },
    )
    intent = unified_signal_to_order_intent(
        unified_signal=signal,
        client_order_id="manual-123",
    )

    assert intent.order_type == "limit"
    assert intent.time_in_force == "gtc"
    assert intent.limit_price == Decimal("99.5")
    assert intent.side == "buy"
    assert intent.client_order_id == "manual-123"

