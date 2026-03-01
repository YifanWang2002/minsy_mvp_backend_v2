from __future__ import annotations

import sys
from types import SimpleNamespace

import packages.domain.trading.services.broker_provider_service as broker_provider_module
from packages.domain.trading.services.broker_provider_service import (
    BrokerProviderService,
)


def _account(
    *,
    provider: str,
    encrypted_credentials: str = "enc",
    exchange_id: str = "",
    account_uid: str = "",
    metadata_: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        encrypted_credentials=encrypted_credentials,
        exchange_id=exchange_id,
        account_uid=account_uid,
        metadata_=metadata_ or {},
    )


def test_000_accessibility_returns_unknown_when_account_missing() -> None:
    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(None)
    assert binding.provider == "unknown"
    assert binding.adapter is None


def test_010_accessibility_returns_no_adapter_for_unimplemented_provider() -> None:
    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(_account(provider="ccxt"))
    assert binding.provider == "ccxt"
    assert binding.adapter is None


def test_020_accessibility_builds_alpaca_adapter_when_credentials_present(monkeypatch) -> None:
    def _fake_decrypt(_self: object, _payload: str) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": "alpaca-key",
            "APCA-API-SECRET-KEY": "alpaca-secret",
            "trading_base_url": "https://paper-api.alpaca.markets",
        }

    monkeypatch.setattr(broker_provider_module.CredentialCipher, "decrypt", _fake_decrypt)

    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(_account(provider="alpaca"))

    assert binding.provider == "alpaca"
    assert binding.adapter is not None
    assert binding.adapter.provider == "alpaca"


def test_030_accessibility_returns_none_when_alpaca_credentials_missing(monkeypatch) -> None:
    def _fake_decrypt(_self: object, _payload: str) -> dict[str, str]:
        return {"APCA-API-KEY-ID": ""}

    monkeypatch.setattr(broker_provider_module.CredentialCipher, "decrypt", _fake_decrypt)

    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(_account(provider="alpaca"))

    assert binding.provider == "alpaca"
    assert binding.adapter is None


def test_040_accessibility_builds_ccxt_adapter_when_credentials_present(monkeypatch) -> None:
    class _CcxtAdapterStub:
        provider = "ccxt"

        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

    fake_module = SimpleNamespace(CcxtTradingAdapter=_CcxtAdapterStub)
    monkeypatch.setitem(sys.modules, "packages.infra.providers.trading.adapters.ccxt_trading", fake_module)

    def _fake_decrypt(_self: object, _payload: str) -> dict[str, str]:
        return {
            "exchange_id": "binance",
            "api_key": "ccxt-key",
            "api_secret": "ccxt-secret",
        }

    monkeypatch.setattr(broker_provider_module.CredentialCipher, "decrypt", _fake_decrypt)

    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(_account(provider="ccxt"))

    assert binding.provider == "ccxt"
    assert binding.adapter is not None
    assert binding.adapter.provider == "ccxt"


def test_050_accessibility_okx_requires_passphrase(monkeypatch) -> None:
    class _CcxtAdapterStub:
        provider = "ccxt"

        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

    fake_module = SimpleNamespace(CcxtTradingAdapter=_CcxtAdapterStub)
    monkeypatch.setitem(sys.modules, "packages.infra.providers.trading.adapters.ccxt_trading", fake_module)

    def _fake_decrypt(_self: object, _payload: str) -> dict[str, str]:
        return {
            "exchange_id": "okx",
            "api_key": "ccxt-key",
            "api_secret": "ccxt-secret",
        }

    monkeypatch.setattr(broker_provider_module.CredentialCipher, "decrypt", _fake_decrypt)

    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(_account(provider="ccxt"))

    assert binding.provider == "ccxt"
    assert binding.adapter is None


def test_060_accessibility_builds_sandbox_adapter() -> None:
    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(
        _account(
            provider="sandbox",
            exchange_id="sandbox",
            account_uid="sandbox-account-1",
            metadata_={"starting_cash": "5000"},
        )
    )
    assert binding.provider == "sandbox"
    assert binding.adapter is not None
    assert binding.adapter.provider == "sandbox"


def test_070_accessibility_sandbox_adapter_reads_fee_slippage_profiles() -> None:
    service = BrokerProviderService()
    binding = service.build_adapter_binding_from_account(
        _account(
            provider="sandbox",
            exchange_id="sandbox",
            account_uid="sandbox-account-2",
            metadata_={
                "starting_cash": "2500",
                "slippage_bps": "1",
                "fee_bps": "2",
                "slippage_bps_by_asset_class": {"crypto": "9"},
                "fee_bps_by_asset_class": {"crypto": "12"},
            },
        )
    )
    assert binding.provider == "sandbox"
    assert binding.adapter is not None
    assert binding.adapter.provider == "sandbox"
    assert binding.adapter._starting_cash == 2500  # type: ignore[attr-defined]
    assert binding.adapter._slippage_bps_profile["crypto"] == 9  # type: ignore[attr-defined]
    assert binding.adapter._fee_bps_profile["crypto"] == 12  # type: ignore[attr-defined]
