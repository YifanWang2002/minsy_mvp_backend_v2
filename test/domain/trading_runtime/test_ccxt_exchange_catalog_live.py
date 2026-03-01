from __future__ import annotations

from packages.domain.trading.services.ccxt_exchange_catalog import (
    list_supported_ccxt_exchanges,
    resolve_ccxt_exchange_metadata,
)


def test_000_accessibility_list_supported_ccxt_exchanges_contains_mainstream_entries() -> (
    None
):
    rows = list_supported_ccxt_exchanges()
    assert rows
    ids = {str(item.get("exchange_id", "")).strip() for item in rows}
    assert "binance" in ids
    assert "okx" in ids
    assert "bybit" in ids


def test_010_accessibility_exchange_password_requirement_mapping() -> None:
    okx = resolve_ccxt_exchange_metadata("okx")
    bitget = resolve_ccxt_exchange_metadata("bitget")
    binance = resolve_ccxt_exchange_metadata("binance")

    okx_required = okx.get("required_fields") if isinstance(okx, dict) else []
    bitget_required = bitget.get("required_fields") if isinstance(bitget, dict) else []
    binance_required = (
        binance.get("required_fields") if isinstance(binance, dict) else []
    )

    assert isinstance(okx_required, list)
    assert isinstance(bitget_required, list)
    assert isinstance(binance_required, list)

    assert "password" in okx_required
    assert "password" in bitget_required
    assert "password" not in binance_required


def test_020_accessibility_dynamic_fallback_keeps_core_contract() -> None:
    dynamic = resolve_ccxt_exchange_metadata("nonexistent_exchange_x")
    assert dynamic["exchange_id"] == "nonexistent_exchange_x"
    assert "api_key" in dynamic["required_fields"]
    assert "api_secret" in dynamic["required_fields"]
    assert dynamic["paper_trading_status"] in {
        "supported",
        "in_progress",
        "not_supported",
    }
    assert isinstance(dynamic["paper_trading_message"], str)
