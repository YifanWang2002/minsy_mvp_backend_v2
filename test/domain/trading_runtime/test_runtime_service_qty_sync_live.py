from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from packages.domain.trading.runtime.runtime_service import (
    _resolve_close_qty_against_broker,
    _resolve_provider_fill_fee,
    _resolve_provider_filled_qty,
    _symbols_match,
    _sync_order_provider_state,
)
from packages.infra.providers.trading.adapters.base import PositionRecord


@dataclass
class _OrderStub:
    metadata_: dict[str, object]
    last_sync_at: datetime | None = None
    provider_updated_at: datetime | None = None
    reject_reason: str | None = None


class _AdapterStub:
    async def fetch_positions(self) -> list[PositionRecord]:
        return [
            PositionRecord(
                symbol="BTCUSD",
                side="long",
                qty=Decimal("0.0009978"),
                avg_entry_price=Decimal("100000"),
                mark_price=Decimal("100100"),
                unrealized_pnl=Decimal("0"),
            )
        ]


async def test_000_accessibility_resolve_close_qty_uses_broker_position_qty() -> None:
    qty, metadata = await _resolve_close_qty_against_broker(
        requested_qty=Decimal("0.001"),
        symbol="BTC/USD",
        adapter=_AdapterStub(),
    )
    assert qty == Decimal("0.0009978")
    assert metadata.get("close_qty_source") == "broker_positions"
    assert metadata.get("close_qty_broker") == "0.0009978"


def test_010_accessibility_sync_order_provider_state_persists_filled_qty() -> None:
    order = _OrderStub(metadata_={})
    _sync_order_provider_state(
        order,  # type: ignore[arg-type]
        provider_status="filled",
        filled_qty=Decimal("0.0009978"),
    )
    assert order.metadata_["provider_status"] == "filled"
    assert order.metadata_["provider_filled_qty"] == "0.0009978"


def test_020_accessibility_resolve_provider_filled_qty_reads_metadata() -> None:
    order = _OrderStub(metadata_={"provider_filled_qty": "0.12345678"})
    resolved = _resolve_provider_filled_qty(order)  # type: ignore[arg-type]
    assert resolved == Decimal("0.12345678")


def test_030_accessibility_symbols_match_crypto_asset_and_pair() -> None:
    assert _symbols_match("BTC", "BTC/USDT") is True
    assert _symbols_match("BTC/USDT", "BTC") is True


def test_040_accessibility_resolve_provider_fill_fee_reads_metadata() -> None:
    order = _OrderStub(metadata_={"provider_fee": "0.00123"})
    resolved = _resolve_provider_fill_fee(order)  # type: ignore[arg-type]
    assert resolved == Decimal("0.00123")
