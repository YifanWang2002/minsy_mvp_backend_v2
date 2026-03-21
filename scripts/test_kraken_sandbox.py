"""Comprehensive Kraken Futures demo/sandbox test.

Tests:
  1. Account state (balance, equity)
  2. Positions
  3. Quote (ticker)
  4. OHLCV candles
  5. Broker validation service
  6. Order lifecycle: submit limit → fetch → cancel
  7. Recent fills
  8. Exchange catalog & capability profile
"""

import asyncio
import os
import sys
import uuid
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.infra.providers.trading.adapters.base import OrderIntent
from packages.infra.providers.trading.adapters.ccxt_trading import CcxtTradingAdapter
from packages.domain.trading.services.broker_validation_service import BrokerValidationService
from packages.domain.trading.services.ccxt_exchange_catalog import (
    list_supported_ccxt_exchanges,
    resolve_ccxt_exchange_metadata,
)
from packages.domain.trading.services.broker_capability_registry import (
    resolve_broker_capability_profile,
)


API_KEY = os.environ.get(
    "KRAKEN_DEMO_API_KEY",
    "qYKzO+QEGxI7oJCRIlwMvw2mVu0pjM/4taWS6Q2KbwHZPqV8MQZmRuwu",
)
API_SECRET = os.environ.get(
    "KRAKEN_DEMO_API_SECRET",
    "Np4pvZDR/BZ1vHcU0AqbfHrbieIC4KXVYq+SHMoYn5oQFBhsjB2X7dVl821iksFRAmyt8FdpXLvxJ4j0DCA00tVq",
)
EXCHANGE_ID = "krakenfutures"
SYMBOL = "BTC/USD:USD"  # perpetual contract


passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


async def test_exchange_catalog() -> None:
    print("\n== Exchange Catalog & Capability ==")

    # krakenfutures in catalog
    exchanges = list_supported_ccxt_exchanges()
    ids = {item.get("exchange_id") for item in exchanges}
    report("krakenfutures in catalog", "krakenfutures" in ids)

    # metadata
    meta = resolve_ccxt_exchange_metadata("krakenfutures")
    report("metadata.supports_demo", bool(meta.get("supports_demo")))
    report("metadata.supports_sandbox", bool(meta.get("supports_sandbox")))
    report(
        "metadata.paper_trading_status == 'supported'",
        meta.get("paper_trading_status") == "supported",
    )
    report(
        "required_fields include api_key & api_secret",
        "api_key" in meta.get("required_fields", [])
        and "api_secret" in meta.get("required_fields", []),
    )

    # capability profile
    profile = resolve_broker_capability_profile(
        provider="ccxt", exchange_id="krakenfutures", is_sandbox=True
    )
    report("capability profile provider == ccxt", profile.provider == "ccxt")
    report(
        "capability profile exchange_id == krakenfutures",
        profile.exchange_id == "krakenfutures",
    )
    report("capability profile sandbox_supported", profile.sandbox_supported)
    report(
        "capability profile asset_classes includes crypto",
        "crypto" in profile.asset_classes,
    )


async def test_broker_validation_service() -> None:
    print("\n== Broker Validation Service ==")

    svc = BrokerValidationService()

    # Missing credentials should fail fast
    result = await svc.validate_credentials(
        provider="ccxt",
        credentials={"exchange_id": "krakenfutures"},
    )
    report("missing api_key → credentials_missing", result.status == "credentials_missing")

    # Invalid credentials should fail with probe error
    result = await svc.validate_credentials(
        provider="ccxt",
        credentials={
            "exchange_id": "krakenfutures",
            "api_key": "bad-key",
            "api_secret": "bad-secret",
            "sandbox": True,
        },
    )
    report("bad credentials → probe failed", not result.ok)
    report(
        "bad credentials → status is ccxt_probe_failed",
        result.status == "ccxt_probe_failed",
    )

    # Valid credentials should succeed
    result = await svc.validate_credentials(
        provider="ccxt",
        credentials={
            "exchange_id": "krakenfutures",
            "api_key": API_KEY,
            "api_secret": API_SECRET,
            "sandbox": True,
        },
    )
    report("valid credentials → ok", result.ok)
    report("valid credentials → ccxt_probe_ok", result.status == "ccxt_probe_ok")
    equity = result.metadata.get("equity", 0)
    report(f"valid credentials → equity > 0 (got {equity})", equity > 0)
    report(
        "valid credentials → sandbox=True in metadata",
        result.metadata.get("sandbox") is True,
    )


async def test_adapter_account_and_market_data() -> None:
    print("\n== Adapter: Account & Market Data ==")

    adapter = CcxtTradingAdapter(
        exchange_id=EXCHANGE_ID,
        api_key=API_KEY,
        api_secret=API_SECRET,
        sandbox=True,
        timeout_seconds=15.0,
    )

    try:
        # Account state
        state = await adapter.fetch_account_state()
        report(f"account equity > 0 (got {state.equity})", state.equity > 0)
        report(f"account cash > 0 (got {state.cash})", state.cash > 0)
        report(
            f"account buying_power > 0 (got {state.buying_power})",
            state.buying_power > 0,
        )

        # Positions
        positions = await adapter.fetch_positions()
        report(
            f"positions returned (count={len(positions)})",
            isinstance(positions, list),
        )

        # Quote
        quote = await adapter.fetch_latest_quote(SYMBOL)
        report("quote returned", quote is not None)
        if quote:
            report(f"quote.bid > 0 (got {quote.bid})", (quote.bid or 0) > 0)
            report(f"quote.ask > 0 (got {quote.ask})", (quote.ask or 0) > 0)
            report(
                "quote.ask >= quote.bid (spread >= 0)",
                (quote.ask or 0) >= (quote.bid or 0),
            )

        # OHLCV
        bars = await adapter.fetch_ohlcv_1m(SYMBOL, limit=5)
        report(f"ohlcv returned bars (count={len(bars)})", len(bars) > 0)
        if bars:
            b = bars[-1]
            report(f"last bar close > 0 (got {b.close})", b.close > 0)

        # Recent fills (may be empty on fresh demo account)
        fills = await adapter.fetch_recent_fills()
        report(
            f"recent fills returned (count={len(fills)})",
            isinstance(fills, list),
        )

    finally:
        await adapter.aclose()


async def test_order_lifecycle() -> None:
    print("\n== Adapter: Order Lifecycle (Limit Order) ==")

    adapter = CcxtTradingAdapter(
        exchange_id=EXCHANGE_ID,
        api_key=API_KEY,
        api_secret=API_SECRET,
        sandbox=True,
        timeout_seconds=15.0,
    )

    order_id = None
    try:
        # Get current price to place a far-from-market limit order
        quote = await adapter.fetch_latest_quote(SYMBOL)
        if not quote or not quote.bid:
            report("SKIP: no quote available for order test", False)
            return

        # Place limit buy well below market so it won't fill
        far_price = round(float(quote.bid) * 0.80, 1)  # 20% below bid
        client_id = f"test-{uuid.uuid4().hex[:12]}"

        intent = OrderIntent(
            client_order_id=client_id,
            symbol=SYMBOL,
            side="buy",
            qty=Decimal("0.001"),  # minimum size
            order_type="limit",
            limit_price=Decimal(str(far_price)),
            time_in_force="gtc",
        )

        # Submit
        order_state = await adapter.submit_order(intent)
        order_id = order_state.provider_order_id
        report(
            f"order submitted (id={order_id})",
            bool(order_id),
        )
        report(
            f"order status is open/new (got '{order_state.status}')",
            order_state.status in ("open", "new", "accepted"),
        )
        report(
            f"order side == buy (got '{order_state.side}')",
            order_state.side == "buy",
        )
        report(
            f"order filled_qty == 0 (got {order_state.filled_qty})",
            order_state.filled_qty == 0,
        )

        # Fetch order
        fetched = await adapter.fetch_order(order_id, symbol=SYMBOL)
        report("fetch_order returned result", fetched is not None)
        if fetched:
            report(
                f"fetched order id matches (got {fetched.provider_order_id})",
                fetched.provider_order_id == order_id,
            )
            report(
                f"fetched order still open (got '{fetched.status}')",
                fetched.status in ("open", "new", "accepted"),
            )

        # Cancel
        cancelled = await adapter.cancel_order(order_id)
        report("cancel_order returned True", cancelled is True)

        # Verify cancelled
        after_cancel = await adapter.fetch_order(order_id, symbol=SYMBOL)
        if after_cancel:
            report(
                f"order status after cancel (got '{after_cancel.status}')",
                after_cancel.status in ("canceled", "cancelled", "closed", "expired"),
            )
        else:
            report("order not found after cancel (acceptable)", True)

    except Exception as e:
        report(f"order lifecycle error: {type(e).__name__}: {e}", False)
        # Try to clean up
        if order_id:
            try:
                await adapter.cancel_order(order_id)
            except Exception:
                pass
    finally:
        await adapter.aclose()


async def main() -> None:
    print("=" * 60)
    print("  Kraken Futures Demo — Comprehensive Test Suite")
    print("=" * 60)

    await test_exchange_catalog()
    await test_broker_validation_service()
    await test_adapter_account_and_market_data()
    await test_order_lifecycle()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
