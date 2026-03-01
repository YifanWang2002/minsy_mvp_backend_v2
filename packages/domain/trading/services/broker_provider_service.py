"""Broker provider resolution and adapter construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.trading.services.ccxt_exchange_catalog import (
    resolve_ccxt_exchange_metadata,
)
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.deployment_run import DeploymentRun
from packages.infra.providers.trading.adapters.alpaca_trading import (
    AlpacaTradingAdapter,
)
from packages.infra.providers.trading.adapters.base import BrokerAdapter
from packages.infra.providers.trading.adapters.sandbox_trading import (
    SandboxTradingAdapter,
)
from packages.infra.providers.trading.credentials import CredentialCipher
from packages.shared_settings.schema.settings import settings


@dataclass(frozen=True, slots=True)
class BrokerAdapterBinding:
    """Resolved provider name and instantiated adapter."""

    provider: str
    adapter: BrokerAdapter | None


def _normalize_provider(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


class BrokerProviderService:
    """Translate broker account config into runtime adapter instances."""

    async def load_account_for_run(
        self,
        db: AsyncSession,
        *,
        run: DeploymentRun,
    ) -> BrokerAccount | None:
        return await db.scalar(select(BrokerAccount).where(BrokerAccount.id == run.broker_account_id))

    async def build_adapter_binding_for_run(
        self,
        db: AsyncSession,
        *,
        run: DeploymentRun,
    ) -> BrokerAdapterBinding:
        account = await self.load_account_for_run(db=db, run=run)
        return self.build_adapter_binding_from_account(account)

    def build_adapter_binding_from_account(self, account: BrokerAccount | None) -> BrokerAdapterBinding:
        if account is None:
            return BrokerAdapterBinding(provider="unknown", adapter=None)

        provider = _normalize_provider(account.provider)
        if provider == "alpaca":
            return BrokerAdapterBinding(provider=provider, adapter=self._build_alpaca_adapter(account))
        if provider == "ccxt":
            return BrokerAdapterBinding(provider=provider, adapter=self._build_ccxt_adapter(account))
        if provider == "sandbox":
            return BrokerAdapterBinding(provider=provider, adapter=self._build_sandbox_adapter(account))

        return BrokerAdapterBinding(provider=provider, adapter=None)

    def _build_alpaca_adapter(self, account: BrokerAccount) -> BrokerAdapter | None:
        try:
            credentials = CredentialCipher().decrypt(account.encrypted_credentials)
        except Exception:  # noqa: BLE001
            return None

        api_key = _extract_credential_value(
            credentials,
            "APCA-API-KEY-ID",
            "api_key",
            "key",
        )
        api_secret = _extract_credential_value(
            credentials,
            "APCA-API-SECRET-KEY",
            "api_secret",
            "secret",
        )
        if not api_key or not api_secret:
            return None

        trading_base_url = _extract_credential_value(credentials, "trading_base_url", "base_url")
        if not trading_base_url:
            trading_base_url = settings.alpaca_trading_base_url

        return AlpacaTradingAdapter(
            api_key=api_key,
            api_secret=api_secret,
            trading_base_url=trading_base_url,
        )

    def _build_ccxt_adapter(self, account: BrokerAccount) -> BrokerAdapter | None:
        try:
            credentials = CredentialCipher().decrypt(account.encrypted_credentials)
        except Exception:  # noqa: BLE001
            return None

        exchange_id = (
            str(account.exchange_id).strip().lower()
            if isinstance(account.exchange_id, str)
            else ""
        )
        if not exchange_id:
            exchange_id = _extract_credential_value(credentials, "exchange_id", "exchange", "name")
        api_key = _extract_credential_value(credentials, "api_key", "key")
        api_secret = _extract_credential_value(credentials, "api_secret", "secret", "secret_key")
        if not exchange_id or not api_key or not api_secret:
            return None

        password = _extract_credential_value(credentials, "password", "passphrase")
        exchange_key = exchange_id.strip().lower()
        exchange_metadata = resolve_ccxt_exchange_metadata(exchange_key)
        required_fields = exchange_metadata.get("required_fields")
        required_fields = required_fields if isinstance(required_fields, list) else []
        if "password" in required_fields and not password:
            return None
        uid = _extract_credential_value(credentials, "uid")
        sandbox_raw = credentials.get("sandbox")
        if sandbox_raw is None:
            sandbox = exchange_key == "okx"
        else:
            sandbox = str(sandbox_raw).strip().lower() in {"1", "true", "yes", "on"}

        raw_timeout = credentials.get("timeout_seconds", settings.ccxt_market_data_timeout_seconds)
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_seconds = float(settings.ccxt_market_data_timeout_seconds)

        try:
            from packages.infra.providers.trading.adapters.ccxt_trading import (
                CcxtTradingAdapter,
            )

            return CcxtTradingAdapter(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                password=password or None,
                uid=uid or None,
                sandbox=sandbox,
                timeout_seconds=timeout_seconds,
            )
        except Exception:  # noqa: BLE001
            return None

    def _build_sandbox_adapter(self, account: BrokerAccount) -> BrokerAdapter:
        metadata = account.metadata_ if isinstance(account.metadata_, dict) else {}
        raw_starting_cash = metadata.get("starting_cash", "100000")
        raw_slippage_bps = metadata.get("slippage_bps", "0")
        raw_fee_bps = metadata.get("fee_bps", "0")
        raw_slippage_profile = metadata.get("slippage_bps_by_asset_class")
        slippage_profile = raw_slippage_profile if isinstance(raw_slippage_profile, dict) else {}
        raw_fee_profile = metadata.get("fee_bps_by_asset_class")
        fee_profile = raw_fee_profile if isinstance(raw_fee_profile, dict) else {}
        account_uid = str(account.account_uid).strip() if isinstance(account.account_uid, str) else ""
        if not account_uid:
            account_uid = f"sandbox-{account.id.hex[:16]}"
        return SandboxTradingAdapter(
            account_uid=account_uid,
            starting_cash=raw_starting_cash,
            slippage_bps=raw_slippage_bps,
            fee_bps=raw_fee_bps,
            slippage_bps_by_asset_class=slippage_profile,
            fee_bps_by_asset_class=fee_profile,
        )
