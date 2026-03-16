"""Pure helper functions for broker capability matching."""

from __future__ import annotations

from typing import Any

from packages.domain.trading.services.broker_capability_registry import (
    resolve_broker_capability_profile,
)

_MARKET_ALIASES: dict[str, str] = {
    "stock": "us_stocks",
    "stocks": "us_stocks",
    "us_stock": "us_stocks",
    "us_stocks": "us_stocks",
    "us_equity": "us_stocks",
    "us_equities": "us_stocks",
    "equity": "us_stocks",
    "equities": "us_stocks",
    "crypto": "crypto",
    "cryptos": "crypto",
    "cryptocurrency": "crypto",
    "cryptocurrencies": "crypto",
    "fx": "forex",
    "forex": "forex",
    "futures": "futures",
    "future": "futures",
}

_MARKET_TO_ASSET_CLASS: dict[str, str] = {
    "us_stocks": "us_equity",
    "crypto": "crypto",
    "forex": "forex",
    "futures": "futures",
}

_ASSET_CLASS_TO_MARKETS: dict[str, tuple[str, ...]] = {
    "us_equity": ("us_stocks",),
    "crypto": ("crypto",),
    "forex": ("forex",),
    "futures": ("futures",),
}


def normalize_market(value: Any) -> str:
    """Normalize a market identifier into product-facing form."""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _MARKET_ALIASES.get(text, text)


def build_broker_capabilities(
    *,
    provider: str,
    exchange_id: str,
    is_sandbox: bool,
) -> dict[str, Any]:
    """Build canonical persisted capability metadata for one broker account."""

    profile = resolve_broker_capability_profile(
        provider=provider,
        exchange_id=exchange_id,
        is_sandbox=is_sandbox,
    )
    return profile.to_capabilities()


def derive_supported_markets(capabilities: dict[str, Any] | None) -> list[str]:
    """Derive normalized supported markets from stored capability metadata."""

    data = capabilities if isinstance(capabilities, dict) else {}
    explicit = data.get("supported_markets")
    output: list[str] = []
    seen: set[str] = set()

    if isinstance(explicit, list):
        for item in explicit:
            normalized = normalize_market(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        if output:
            return output

    raw_asset_classes = data.get("asset_classes")
    if not isinstance(raw_asset_classes, list):
        return output

    for item in raw_asset_classes:
        asset_class = str(item or "").strip().lower()
        if not asset_class:
            continue
        for market in _ASSET_CLASS_TO_MARKETS.get(asset_class, ()):
            if market in seen:
                continue
            seen.add(market)
            output.append(market)
    return output


def capability_supports_market(
    *,
    capabilities: dict[str, Any] | None,
    market: str | None,
) -> bool:
    """Return True when the broker capabilities support the requested market."""

    normalized_market = normalize_market(market)
    if not normalized_market:
        return False

    supported_markets = derive_supported_markets(capabilities)
    if normalized_market in supported_markets:
        return True

    asset_class = _MARKET_TO_ASSET_CLASS.get(normalized_market)
    if asset_class is None:
        return False

    raw_asset_classes = (
        capabilities.get("asset_classes")
        if isinstance(capabilities, dict)
        else []
    )
    if not isinstance(raw_asset_classes, list):
        return False
    return asset_class in {
        str(item or "").strip().lower()
        for item in raw_asset_classes
        if str(item or "").strip()
    }


def capability_supports_order_type(
    *,
    capabilities: dict[str, Any] | None,
    order_type: str | None,
) -> bool:
    """Return True when broker capabilities support the requested order type."""

    normalized_order_type = str(order_type or "").strip().lower()
    if not normalized_order_type:
        return False
    data = capabilities if isinstance(capabilities, dict) else {}
    raw_order_types = data.get("order_types")
    if not isinstance(raw_order_types, list):
        return False
    supported = {
        str(item or "").strip().lower()
        for item in raw_order_types
        if str(item or "").strip()
    }
    return normalized_order_type in supported


def capability_supports_time_in_force(
    *,
    capabilities: dict[str, Any] | None,
    time_in_force: str | None,
) -> bool:
    """Return True when broker capabilities support the requested TIF."""

    normalized_tif = str(time_in_force or "").strip().lower()
    if not normalized_tif:
        return False
    data = capabilities if isinstance(capabilities, dict) else {}
    raw_values = data.get("time_in_force")
    if not isinstance(raw_values, list):
        return False
    supported = {
        str(item or "").strip().lower()
        for item in raw_values
        if str(item or "").strip()
    }
    return normalized_tif in supported


def evaluate_broker_compatibility(
    *,
    strategy_market: str | None,
    accounts: list[dict[str, Any]],
    explicit_broker_account_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate compatibility of a user's broker accounts with one strategy."""

    normalized_market = normalize_market(strategy_market)
    active_accounts = [
        account
        for account in accounts
        if str(account.get("status", "")).strip().lower() == "active"
    ]
    if not active_accounts:
        return {
            "status": "no_broker",
            "strategy_market": normalized_market,
            "matched_broker_account_ids": [],
            "preferred_broker_account_id": None,
            "blockers": ["No active paper broker account is connected."],
        }

    matched_accounts = [
        account
        for account in active_accounts
        if capability_supports_market(
            capabilities=account.get("capabilities"),
            market=normalized_market,
        )
    ]
    matched_ids = [
        str(account.get("broker_account_id"))
        for account in matched_accounts
        if str(account.get("broker_account_id", "")).strip()
    ]

    explicit_account_id = str(explicit_broker_account_id or "").strip()
    if explicit_account_id:
        if explicit_account_id in matched_ids:
            return {
                "status": "ready",
                "strategy_market": normalized_market,
                "matched_broker_account_ids": matched_ids,
                "preferred_broker_account_id": explicit_account_id,
                "blockers": [],
            }
        return {
            "status": "blocked",
            "strategy_market": normalized_market,
            "matched_broker_account_ids": matched_ids,
            "preferred_broker_account_id": None,
            "blockers": [
                "The selected broker account does not support this strategy market."
            ],
        }

    if not matched_accounts:
        return {
            "status": "blocked",
            "strategy_market": normalized_market,
            "matched_broker_account_ids": [],
            "preferred_broker_account_id": None,
            "blockers": [
                "No connected broker account supports the current strategy market."
            ],
        }

    if len(matched_accounts) == 1:
        account_id = str(matched_accounts[0].get("broker_account_id", "")).strip()
        return {
            "status": "ready",
            "strategy_market": normalized_market,
            "matched_broker_account_ids": matched_ids,
            "preferred_broker_account_id": account_id or None,
            "blockers": [],
        }

    default_id = None
    for account in matched_accounts:
        if bool(account.get("is_default")):
            candidate = str(account.get("broker_account_id", "")).strip()
            if candidate:
                default_id = candidate
                break

    return {
        "status": "needs_choice",
        "strategy_market": normalized_market,
        "matched_broker_account_ids": matched_ids,
        "preferred_broker_account_id": default_id,
        "blockers": [],
    }
