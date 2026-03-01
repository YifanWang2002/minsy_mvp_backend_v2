"""Provider-specific broker credential validation service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.domain.trading.services.ccxt_exchange_catalog import (
    resolve_ccxt_exchange_metadata,
)
from packages.infra.providers.trading.adapters.ccxt_trading import CcxtTradingAdapter
from packages.infra.providers.trading.alpaca_account_probe import AlpacaAccountProbe


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


@dataclass(frozen=True, slots=True)
class BrokerCredentialValidationResult:
    ok: bool
    status: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BrokerValidationService:
    """Validate broker credentials with provider-specific rules."""

    async def validate_credentials(
        self,
        *,
        provider: str,
        credentials: dict[str, Any],
    ) -> BrokerCredentialValidationResult:
        provider_key = str(provider).strip().lower()
        if provider_key == "alpaca":
            return await self._validate_alpaca(credentials)
        if provider_key == "ccxt":
            return await self._validate_ccxt(credentials)
        if provider_key == "sandbox":
            return self._validate_sandbox(credentials)
        return BrokerCredentialValidationResult(
            ok=False,
            status="provider_not_supported",
            message=f"Unsupported broker provider: {provider_key or 'unknown'}.",
            metadata={"provider": provider_key or "unknown"},
        )

    async def _validate_alpaca(self, credentials: dict[str, Any]) -> BrokerCredentialValidationResult:
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
            return BrokerCredentialValidationResult(
                ok=False,
                status="credentials_missing",
                message="Alpaca API key and secret are required for validation.",
                metadata={"provider": "alpaca"},
            )

        probe = AlpacaAccountProbe()
        try:
            result = await probe.probe_credentials(api_key=api_key, api_secret=api_secret)
        finally:
            await probe.aclose()

        return BrokerCredentialValidationResult(
            ok=bool(result.ok),
            status=str(result.status),
            message=str(result.message),
            metadata=result.metadata if isinstance(result.metadata, dict) else {},
        )

    async def _validate_ccxt(self, credentials: dict[str, Any]) -> BrokerCredentialValidationResult:
        exchange_id = _extract_credential_value(credentials, "exchange_id", "exchange", "name")
        api_key = _extract_credential_value(credentials, "api_key", "key")
        api_secret = _extract_credential_value(credentials, "api_secret", "secret", "secret_key")
        if not exchange_id or not api_key or not api_secret:
            return BrokerCredentialValidationResult(
                ok=False,
                status="credentials_missing",
                message="CCXT exchange_id, api_key, and api_secret are required for validation.",
                metadata={"provider": "ccxt"},
            )

        exchange_metadata = resolve_ccxt_exchange_metadata(exchange_id)
        required_fields = exchange_metadata.get("required_fields")
        required_fields = required_fields if isinstance(required_fields, list) else []
        passphrase = _extract_credential_value(credentials, "password", "passphrase")
        exchange_key = exchange_id.strip().lower()
        if "password" in required_fields and not passphrase:
            return BrokerCredentialValidationResult(
                ok=False,
                status="credentials_missing",
                message=(
                    f"{str(exchange_metadata.get('name') or exchange_key).strip() or exchange_key} requires "
                    "passphrase (password) in addition to api_key and api_secret."
                ),
                metadata={
                    "provider": "ccxt",
                    "exchange_id": exchange_key,
                    "missing_fields": ["passphrase"],
                },
            )
        uid = _extract_credential_value(credentials, "uid")
        sandbox_raw = credentials.get("sandbox")
        if sandbox_raw is None:
            sandbox = exchange_key == "okx"
        else:
            sandbox = str(sandbox_raw).strip().lower() in {"1", "true", "yes", "on"}
        timeout_raw = credentials.get("timeout_seconds", 8.0)
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = 8.0

        adapter = None
        try:
            adapter = CcxtTradingAdapter(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                password=passphrase or None,
                uid=uid or None,
                sandbox=sandbox,
                timeout_seconds=timeout_seconds,
            )
            account_state = await adapter.fetch_account_state()
            return BrokerCredentialValidationResult(
                ok=True,
                status="ccxt_probe_ok",
                message="CCXT credentials validated successfully.",
                metadata={
                    "provider": "ccxt",
                    "exchange_id": exchange_key,
                    "sandbox": sandbox,
                    "equity": float(account_state.equity),
                    "cash": float(account_state.cash),
                    "buying_power": float(account_state.buying_power),
                },
            )
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).strip()
            detail_text = detail[:240] if detail else None
            return BrokerCredentialValidationResult(
                ok=False,
                status="ccxt_probe_failed",
                message=(
                    f"CCXT validation failed: {type(exc).__name__}: {detail_text}"
                    if detail_text
                    else f"CCXT validation failed: {type(exc).__name__}"
                ),
                metadata={
                    "provider": "ccxt",
                    "exchange_id": exchange_key,
                    "sandbox": sandbox,
                    "error_type": type(exc).__name__,
                    "error_detail": detail_text,
                },
            )
        finally:
            if adapter is not None:
                try:
                    await adapter.aclose()
                except Exception:  # noqa: BLE001
                    pass

    def _validate_sandbox(self, credentials: dict[str, Any]) -> BrokerCredentialValidationResult:
        _ = credentials
        return BrokerCredentialValidationResult(
            ok=True,
            status="sandbox_ready",
            message="Sandbox broker is managed internally.",
            metadata={
                "provider": "sandbox",
                "mode": "internal",
                "is_sandbox": True,
                "market_data_source": "alpaca",
            },
        )
