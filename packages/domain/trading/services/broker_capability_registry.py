"""Unified broker capability profile registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from packages.domain.trading.services.ccxt_exchange_catalog import (
    list_supported_ccxt_exchanges,
    resolve_ccxt_exchange_metadata,
)


@dataclass(frozen=True, slots=True)
class BrokerCapabilityProfile:
    """Canonical capability profile for one broker/provider target."""

    provider: str
    exchange_id: str = ""
    display_name: str = ""
    integration_status: str = "available"
    connectivity_mode: str = "api"
    requires_api_credentials: bool = True
    requires_local_port: bool = False
    requires_desktop_app: bool = False
    desktop_app: str | None = None
    data_source_mode: str = "user_byo_credentials"
    asset_classes: tuple[str, ...] = ()
    supported_markets: tuple[str, ...] = ()
    order_types: tuple[str, ...] = ("market",)
    time_in_force: tuple[str, ...] = ("gtc",)
    execution_routes_supported: tuple[str, ...] = ("server",)
    sandbox_supported: bool = False
    supports_demo: bool = False
    supports_testnet: bool = False
    supports_paper: bool = False
    supports_live: bool = False
    region_policy: str = "unknown"
    region_notes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_capabilities(self) -> dict[str, Any]:
        """Convert profile to persisted broker_account capabilities payload."""

        payload = self.to_public_dict()
        payload["asset_classes"] = list(self.asset_classes)
        payload["supported_markets"] = list(self.supported_markets)
        payload["order_types"] = list(self.order_types)
        payload["time_in_force"] = list(self.time_in_force)
        payload["execution_routes_supported"] = list(self.execution_routes_supported)
        payload["region_notes"] = list(self.region_notes)
        payload["notes"] = list(self.notes)
        return payload

    def to_public_dict(self) -> dict[str, Any]:
        """Convert profile to API-facing payload."""

        return {
            "provider": self.provider,
            "exchange_id": self.exchange_id,
            "display_name": self.display_name or self.provider.upper(),
            "integration_status": self.integration_status,
            "connectivity_mode": self.connectivity_mode,
            "requires_api_credentials": self.requires_api_credentials,
            "requires_local_port": self.requires_local_port,
            "requires_desktop_app": self.requires_desktop_app,
            "desktop_app": self.desktop_app,
            "data_source_mode": self.data_source_mode,
            "asset_classes": list(self.asset_classes),
            "supported_markets": list(self.supported_markets),
            "order_types": list(self.order_types),
            "time_in_force": list(self.time_in_force),
            "execution_routes_supported": list(self.execution_routes_supported),
            "sandbox_supported": self.sandbox_supported,
            "supports_demo": self.supports_demo,
            "supports_testnet": self.supports_testnet,
            "supports_paper": self.supports_paper,
            "supports_live": self.supports_live,
            "region_policy": self.region_policy,
            "region_notes": list(self.region_notes),
            "notes": list(self.notes),
        }


_CCXT_REGION_HINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "binance": (
        "restricted_by_jurisdiction",
        (
            "Service availability and legal entity vary by user region.",
            "US users typically require Binance.US instead of global Binance.",
        ),
    ),
    "okx": (
        "restricted_by_jurisdiction",
        (
            "Availability differs by region and legal entity.",
            "Some jurisdictions require region-specific domains and account onboarding.",
        ),
    ),
    "coinbase": (
        "restricted_by_jurisdiction",
        (
            "Product availability depends on region and regulatory approvals.",
        ),
    ),
}

_PLANNED_LOCAL_PROFILES: tuple[BrokerCapabilityProfile, ...] = (
    BrokerCapabilityProfile(
        provider="ibkr",
        display_name="IBKR",
        integration_status="planned",
        connectivity_mode="local_gateway",
        requires_local_port=True,
        requires_desktop_app=True,
        desktop_app="IB Gateway / Trader Workstation",
        data_source_mode="edge_collector",
        asset_classes=("us_equity", "forex", "futures"),
        supported_markets=("us_stocks", "forex", "futures"),
        order_types=("market", "limit", "stop", "stop_limit"),
        time_in_force=("day", "gtc", "ioc", "fok"),
        execution_routes_supported=("client",),
        supports_paper=True,
        supports_live=True,
        region_policy="restricted_by_jurisdiction",
        region_notes=("Account availability varies by country and legal entity.",),
        notes=("Desktop gateway dependency: cloud direct execution not always possible.",),
    ),
    BrokerCapabilityProfile(
        provider="mt4",
        display_name="MetaTrader 4",
        integration_status="planned",
        connectivity_mode="desktop_port",
        requires_local_port=True,
        requires_desktop_app=True,
        desktop_app="MetaTrader 4 terminal",
        data_source_mode="edge_collector",
        asset_classes=("forex",),
        supported_markets=("forex",),
        order_types=("market", "limit", "stop", "stop_limit"),
        time_in_force=("day", "gtc"),
        execution_routes_supported=("client",),
        supports_paper=False,
        supports_live=True,
        region_policy="broker_dependent",
    ),
    BrokerCapabilityProfile(
        provider="mt5",
        display_name="MetaTrader 5",
        integration_status="planned",
        connectivity_mode="desktop_port",
        requires_local_port=True,
        requires_desktop_app=True,
        desktop_app="MetaTrader 5 terminal",
        data_source_mode="edge_collector",
        asset_classes=("forex", "futures"),
        supported_markets=("forex", "futures"),
        order_types=("market", "limit", "stop", "stop_limit"),
        time_in_force=("day", "gtc"),
        execution_routes_supported=("client",),
        supports_paper=False,
        supports_live=True,
        region_policy="broker_dependent",
    ),
    BrokerCapabilityProfile(
        provider="ninjatrader",
        display_name="NinjaTrader",
        integration_status="planned",
        connectivity_mode="desktop_port",
        requires_local_port=True,
        requires_desktop_app=True,
        desktop_app="NinjaTrader desktop",
        data_source_mode="edge_collector",
        asset_classes=("futures", "forex"),
        supported_markets=("futures", "forex"),
        order_types=("market", "limit", "stop", "stop_limit"),
        time_in_force=("day", "gtc"),
        execution_routes_supported=("client",),
        supports_paper=True,
        supports_live=True,
        region_policy="restricted_by_jurisdiction",
    ),
    BrokerCapabilityProfile(
        provider="tradestation",
        display_name="TradeStation",
        integration_status="planned",
        connectivity_mode="api",
        requires_local_port=False,
        requires_desktop_app=False,
        data_source_mode="user_byo_credentials",
        asset_classes=("us_equity", "futures", "forex"),
        supported_markets=("us_stocks", "futures", "forex"),
        order_types=("market", "limit", "stop", "stop_limit"),
        time_in_force=("day", "gtc", "ioc", "fok"),
        execution_routes_supported=("server", "client"),
        supports_paper=True,
        supports_live=True,
        region_policy="restricted_by_jurisdiction",
    ),
)


def _normalize_provider(provider: str) -> str:
    return str(provider or "").strip().lower()


def _normalize_exchange_id(exchange_id: str) -> str:
    return str(exchange_id or "").strip().lower()


def _resolve_ccxt_region_profile(exchange_id: str) -> tuple[str, tuple[str, ...]]:
    return _CCXT_REGION_HINTS.get(
        exchange_id,
        ("restricted_by_jurisdiction", ("Exchange availability may vary by country.",)),
    )


def resolve_broker_capability_profile(
    *,
    provider: str,
    exchange_id: str = "",
    is_sandbox: bool = False,
) -> BrokerCapabilityProfile:
    """Resolve one canonical capability profile."""

    provider_key = _normalize_provider(provider)
    exchange_key = _normalize_exchange_id(exchange_id)
    sandbox_enabled = bool(is_sandbox)

    if provider_key == "alpaca":
        return BrokerCapabilityProfile(
            provider="alpaca",
            display_name="Alpaca",
            connectivity_mode="api",
            data_source_mode="user_byo_credentials",
            asset_classes=("us_equity", "crypto"),
            supported_markets=("us_stocks", "crypto"),
            order_types=("market", "limit", "stop", "stop_limit"),
            time_in_force=("day", "gtc", "ioc", "fok"),
            execution_routes_supported=("server", "client"),
            sandbox_supported=True,
            supports_demo=False,
            supports_testnet=False,
            supports_paper=True,
            supports_live=True,
            region_policy="restricted_by_jurisdiction",
            region_notes=(
                "Availability depends on country and account eligibility.",
            ),
        )

    if provider_key == "sandbox":
        return BrokerCapabilityProfile(
            provider="sandbox",
            display_name="Built-in Sandbox",
            connectivity_mode="api",
            requires_api_credentials=False,
            data_source_mode="platform_shared",
            asset_classes=("us_equity", "crypto"),
            supported_markets=("us_stocks", "crypto"),
            order_types=("market", "limit"),
            time_in_force=("gtc",),
            execution_routes_supported=("server",),
            sandbox_supported=True,
            supports_demo=True,
            supports_testnet=False,
            supports_paper=True,
            supports_live=False,
            region_policy="global",
            notes=("Execution is internally simulated with platform-owned runtime.",),
        )

    if provider_key == "ccxt":
        metadata = resolve_ccxt_exchange_metadata(exchange_key)
        profile_exchange_id = _normalize_exchange_id(metadata.get("exchange_id", exchange_key))
        display_name = str(metadata.get("name") or profile_exchange_id.upper())
        region_policy, region_notes = _resolve_ccxt_region_profile(profile_exchange_id)
        metadata_sandbox = bool(metadata.get("supports_sandbox"))
        metadata_demo = bool(metadata.get("supports_demo"))
        metadata_testnet = bool(metadata.get("supports_testnet"))
        metadata_paper = bool(metadata.get("supports_paper"))
        metadata_live = bool(metadata.get("supports_live"))
        sandbox_supported = sandbox_enabled or metadata_sandbox or metadata_demo or metadata_testnet
        return BrokerCapabilityProfile(
            provider="ccxt",
            exchange_id=profile_exchange_id,
            display_name=f"CCXT / {display_name}",
            connectivity_mode="api",
            data_source_mode="user_byo_credentials",
            asset_classes=("crypto",),
            supported_markets=("crypto",),
            order_types=("market", "limit"),
            time_in_force=("gtc", "ioc", "fok"),
            execution_routes_supported=("server", "client"),
            sandbox_supported=sandbox_supported,
            supports_demo=metadata_demo,
            supports_testnet=metadata_testnet,
            supports_paper=metadata_paper or sandbox_supported,
            supports_live=metadata_live,
            region_policy=region_policy,
            region_notes=region_notes,
            notes=(
                "Exchange-specific precision and order-parameter rules still apply.",
            ),
        )

    for planned in _PLANNED_LOCAL_PROFILES:
        if planned.provider == provider_key:
            return planned

    return BrokerCapabilityProfile(
        provider=provider_key or "unknown",
        display_name=(provider_key or "unknown").upper(),
        integration_status="unknown",
        supports_paper=False,
        supports_live=False,
        notes=("No capability profile is registered for this provider.",),
    )


@lru_cache(maxsize=1)
def list_supported_broker_capability_profiles() -> tuple[dict[str, Any], ...]:
    """Return all discoverable broker capability profiles."""

    profiles: list[BrokerCapabilityProfile] = [
        resolve_broker_capability_profile(provider="alpaca"),
        resolve_broker_capability_profile(provider="sandbox"),
    ]

    for item in list_supported_ccxt_exchanges():
        exchange_id = _normalize_exchange_id(item.get("exchange_id", ""))
        if not exchange_id:
            continue
        profiles.append(
            resolve_broker_capability_profile(
                provider="ccxt",
                exchange_id=exchange_id,
            )
        )

    profiles.extend(_PLANNED_LOCAL_PROFILES)
    return tuple(profile.to_public_dict() for profile in profiles)

