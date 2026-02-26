"""Alpaca paper-account credential probe helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from packages.shared_settings.schema.settings import settings


@dataclass(frozen=True)
class AlpacaAccountProbeResult:
    """Result payload for paper-account credential validation."""

    ok: bool
    status: str
    message: str
    metadata: dict[str, Any]


class AlpacaAccountProbe:
    """Validate whether provided Alpaca credentials are paper-only keys."""

    def __init__(
        self,
        *,
        paper_base_url: str | None = None,
        live_base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.paper_base_url = (paper_base_url or settings.alpaca_paper_trading_base_url).rstrip("/")
        self.live_base_url = (live_base_url or settings.alpaca_live_trading_base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.alpaca_account_probe_timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=self.timeout_seconds)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def probe_credentials(
        self,
        *,
        api_key: str,
        api_secret: str,
        verify_live_endpoint: bool = True,
    ) -> AlpacaAccountProbeResult:
        """Probe paper and live account endpoints using user credentials."""
        headers = {
            "APCA-API-KEY-ID": api_key.strip(),
            "APCA-API-SECRET-KEY": api_secret.strip(),
        }

        paper = await self._probe_endpoint(self.paper_base_url, headers=headers)
        live = await self._probe_endpoint(self.live_base_url, headers=headers) if verify_live_endpoint else None

        paper_status = paper["status_code"]
        live_status = live["status_code"] if isinstance(live, dict) else None
        paper_account_status = paper["account_status"]
        account_status_readable = bool(paper_account_status)

        metadata: dict[str, Any] = {
            "paper_endpoint": paper["endpoint"],
            "paper_http_status": paper_status,
            "paper_account_status": paper_account_status,
            "paper_account_id": paper["account_id"],
            "paper_currency": paper["currency"],
            "paper_equity": paper["equity"],
            "paper_cash": paper["cash"],
            "paper_buying_power": paper["buying_power"],
            "paper_error": paper["error"],
            "account_status_readable": account_status_readable,
            "live_endpoint_checked": verify_live_endpoint,
            "live_endpoint": live["endpoint"] if isinstance(live, dict) else None,
            "live_http_status": live_status,
            "live_error": live["error"] if isinstance(live, dict) else None,
        }

        if paper_status == 200 and not account_status_readable:
            return AlpacaAccountProbeResult(
                ok=False,
                status="paper_probe_failed",
                message="Paper account status is not readable.",
                metadata=metadata,
            )

        if paper_status == 200 and account_status_readable and verify_live_endpoint:
            if live_status == 401:
                return AlpacaAccountProbeResult(
                    ok=True,
                    status="paper_probe_ok",
                    message="Paper credentials validated successfully.",
                    metadata=metadata,
                )
            if live_status == 200:
                return AlpacaAccountProbeResult(
                    ok=False,
                    status="paper_probe_failed",
                    message="Credentials are not paper-only keys.",
                    metadata=metadata,
                )
            return AlpacaAccountProbeResult(
                ok=False,
                status="paper_probe_failed",
                message="Live endpoint returned unexpected result during paper-only validation.",
                metadata=metadata,
            )

        if paper_status == 200 and account_status_readable and not verify_live_endpoint:
            return AlpacaAccountProbeResult(
                ok=True,
                status="paper_probe_ok",
                message="Paper credentials validated successfully.",
                metadata=metadata,
            )

        if paper_status == 401:
            return AlpacaAccountProbeResult(
                ok=False,
                status="paper_probe_failed",
                message="Invalid Alpaca credentials or wrong endpoint.",
                metadata=metadata,
            )

        return AlpacaAccountProbeResult(
            ok=False,
            status="paper_probe_failed",
            message="Paper endpoint probe failed.",
            metadata=metadata,
        )

    async def _probe_endpoint(
        self,
        base_url: str,
        *,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        endpoint = f"{base_url}/v2/account"
        status_code: int | None = None
        error: str | None = None
        account_status: str | None = None
        account_id: str | None = None
        equity: str | None = None
        cash: str | None = None
        buying_power: str | None = None
        currency: str | None = None

        try:
            response = await self._client.get(endpoint, headers=headers)
            status_code = response.status_code
            payload: Any = None
            if response.content:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            if isinstance(payload, dict):
                status_value = payload.get("status")
                if isinstance(status_value, str) and status_value.strip():
                    account_status = status_value.strip()
                account_id_value = payload.get("id")
                if account_id_value is not None:
                    normalized_id = str(account_id_value).strip()
                    if normalized_id:
                        account_id = normalized_id
                currency_value = payload.get("currency")
                if currency_value is not None:
                    normalized_currency = str(currency_value).strip()
                    if normalized_currency:
                        currency = normalized_currency
                for key, target in (
                    ("equity", "equity"),
                    ("cash", "cash"),
                    ("buying_power", "buying_power"),
                ):
                    raw = payload.get(key)
                    if raw is None:
                        continue
                    normalized = str(raw).strip()
                    if not normalized:
                        continue
                    if target == "equity":
                        equity = normalized
                    elif target == "cash":
                        cash = normalized
                    else:
                        buying_power = normalized
        except httpx.HTTPError as exc:
            error = str(exc)

        return {
            "endpoint": endpoint,
            "status_code": status_code,
            "account_status": account_status,
            "account_id": account_id,
            "currency": currency,
            "equity": equity,
            "cash": cash,
            "buying_power": buying_power,
            "error": error,
        }
