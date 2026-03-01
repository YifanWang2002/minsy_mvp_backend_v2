from __future__ import annotations

from types import SimpleNamespace

import packages.domain.trading.services.broker_validation_service as validation_module
from packages.domain.trading.services.broker_validation_service import (
    BrokerValidationService,
)


async def test_000_accessibility_sandbox_validation_succeeds() -> None:
    service = BrokerValidationService()
    result = await service.validate_credentials(provider="sandbox", credentials={})
    assert result.ok is True
    assert result.status == "sandbox_ready"
    assert result.metadata.get("provider") == "sandbox"


async def test_010_accessibility_ccxt_validation_requires_core_fields() -> None:
    service = BrokerValidationService()
    result = await service.validate_credentials(provider="ccxt", credentials={"exchange_id": "binance"})
    assert result.ok is False
    assert result.status == "credentials_missing"


async def test_020_accessibility_ccxt_validation_probe_success(monkeypatch) -> None:
    class _CcxtAdapterStub:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def fetch_account_state(self) -> SimpleNamespace:
            return SimpleNamespace(equity=1000.0, cash=1000.0, buying_power=1000.0)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(validation_module, "CcxtTradingAdapter", _CcxtAdapterStub)

    service = BrokerValidationService()
    result = await service.validate_credentials(
        provider="ccxt",
        credentials={
            "exchange_id": "binance",
            "api_key": "demo-key",
            "api_secret": "demo-secret",
        },
    )
    assert result.ok is True
    assert result.status == "ccxt_probe_ok"
    assert result.metadata.get("exchange_id") == "binance"


async def test_030_accessibility_okx_requires_passphrase() -> None:
    service = BrokerValidationService()
    result = await service.validate_credentials(
        provider="ccxt",
        credentials={
            "exchange_id": "okx",
            "api_key": "demo-key",
            "api_secret": "demo-secret",
        },
    )
    assert result.ok is False
    assert result.status == "credentials_missing"
    assert result.metadata.get("missing_fields") == ["passphrase"]


async def test_035_accessibility_bitget_requires_passphrase() -> None:
    service = BrokerValidationService()
    result = await service.validate_credentials(
        provider="ccxt",
        credentials={
            "exchange_id": "bitget",
            "api_key": "demo-key",
            "api_secret": "demo-secret",
        },
    )
    assert result.ok is False
    assert result.status == "credentials_missing"
    assert result.metadata.get("missing_fields") == ["passphrase"]


async def test_040_accessibility_alpaca_missing_credentials_fails_fast() -> None:
    service = BrokerValidationService()
    result = await service.validate_credentials(provider="alpaca", credentials={})
    assert result.ok is False
    assert result.status == "credentials_missing"
