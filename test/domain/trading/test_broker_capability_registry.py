from __future__ import annotations

from packages.domain.trading.services.broker_capability_registry import (
    list_supported_broker_capability_profiles,
    resolve_broker_capability_profile,
)


def test_resolve_alpaca_profile_exposes_region_and_connectivity_metadata() -> None:
    profile = resolve_broker_capability_profile(provider="alpaca")
    payload = profile.to_public_dict()

    assert payload["provider"] == "alpaca"
    assert payload["connectivity_mode"] == "api"
    assert payload["supported_markets"] == ["us_stocks", "crypto"]
    assert payload["requires_local_port"] is False
    assert payload["region_policy"] == "restricted_by_jurisdiction"
    assert payload["supports_paper"] is True


def test_resolve_ccxt_profile_includes_exchange_specific_fields() -> None:
    profile = resolve_broker_capability_profile(
        provider="ccxt",
        exchange_id="okx",
    )
    capabilities = profile.to_capabilities()

    assert capabilities["provider"] == "ccxt"
    assert capabilities["exchange_id"] == "okx"
    assert capabilities["supported_markets"] == ["crypto"]
    assert capabilities["order_types"] == ["market", "limit"]
    assert capabilities["execution_routes_supported"] == ["server", "client"]


def test_list_supported_profiles_includes_planned_local_connectors() -> None:
    profiles = list_supported_broker_capability_profiles()
    provider_keys = {(row["provider"], row.get("exchange_id", "")) for row in profiles}

    assert ("alpaca", "") in provider_keys
    assert ("sandbox", "") in provider_keys
    assert ("ibkr", "") in provider_keys
    assert any(provider == "ccxt" for provider, _ in provider_keys)

