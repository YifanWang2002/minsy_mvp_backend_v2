from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from packages.domain.trading.runtime.runtime_service import (
    RuntimeAccountBudget,
    _resolve_scope,
    _resolve_close_qty_against_broker,
    _resolve_runtime_account_budget_from_run,
    _resolve_provider_fill_fee,
    _resolve_provider_filled_qty,
    _sync_pending_orders_for_deployment,
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


@dataclass
class _RunStub:
    runtime_state: dict[str, object]


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


def test_050_accessibility_resolve_runtime_account_budget_prefers_broker_snapshot_cash() -> None:
    budget = _resolve_runtime_account_budget_from_run(
        _RunStub(
            runtime_state={
                "broker_account": {
                    "cash": 2500,
                    "equity": 3000,
                    "buying_power": 5000,
                }
            }
        )  # type: ignore[arg-type]
    )

    assert budget == RuntimeAccountBudget(
        cash=Decimal("2500"),
        equity=Decimal("3000"),
        source="runtime_state.broker_account.cash",
    )


def test_060_accessibility_resolve_scope_normalizes_invalid_timeframe_to_dsl_default() -> None:
    from types import SimpleNamespace

    deployment = SimpleNamespace(
        strategy=SimpleNamespace(
            dsl_payload={
                "universe": {"market": "crypto", "tickers": ["BTCUSD"]},
                "timeframe": "7m",
            },
            symbols=["BTCUSD"],
            timeframe="7m",
        )
    )

    market, symbol, timeframe = _resolve_scope(deployment)  # type: ignore[arg-type]

    assert market == "crypto"
    assert symbol == "BTCUSD"
    assert timeframe == "1m"


async def test_070_accessibility_pending_order_sync_degrades_to_partial_error(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    class _ScalarResultStub:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def all(self) -> list[object]:
            return list(self._rows)

    class _DbStub:
        async def scalars(self, _query) -> _ScalarResultStub:
            return _ScalarResultStub(
                [
                    SimpleNamespace(
                        id="order-1",
                        deployment_id="dep-1",
                        provider_order_id="prov-1",
                        status="pending_new",
                        symbol="BTCUSD",
                    )
                ]
            )

    async def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "packages.domain.trading.runtime.runtime_service.sync_order_status_from_adapter",
        _boom,
    )

    result = await _sync_pending_orders_for_deployment(
        _DbStub(),  # type: ignore[arg-type]
        deployment=SimpleNamespace(id="dep-1"),  # type: ignore[arg-type]
        adapter=object(),  # type: ignore[arg-type]
    )

    assert result["pending_order_sync"] == "partial_error"
    assert result["pending_orders_checked"] == 1
    assert result["pending_order_sync_errors"] == 1
    assert result["pending_order_sync_last_error"] == "RuntimeError"
