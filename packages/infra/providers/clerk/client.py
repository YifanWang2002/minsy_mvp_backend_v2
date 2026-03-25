"""Clerk Backend API wrapper used by migration and sync workflows."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from packages.shared_settings.schema.settings import settings


class ClerkClientConfigError(RuntimeError):
    """Raised when Clerk is used without required configuration."""


class ClerkApiError(RuntimeError):
    """Raised when Clerk Backend API requests fail."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ClerkClient:
    """Thin async wrapper around Clerk's Backend API."""

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        api_url: str | None = None,
        api_version: str | None = None,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self._secret_key = (secret_key or settings.clerk_secret_key).strip()
        self._api_url = (api_url or settings.clerk_backend_api_url).rstrip("/")
        self._api_version = (api_version or settings.clerk_api_version).strip()
        self._timeout = max(float(timeout_seconds), 1.0)
        self._transport = transport
        self._sleep = sleep

    @property
    def is_configured(self) -> bool:
        return bool(self._secret_key and self._api_url)

    def _ensure_configured(self) -> None:
        if not self.is_configured:
            raise ClerkClientConfigError(
                "Clerk backend API is not configured. "
                "Expected CLERK_SECRET_KEY and CLERK_BACKEND_API_URL.",
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._secret_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_version:
            headers["Clerk-Version"] = self._api_version
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str]] | None = None,
        json_body: dict[str, Any] | None = None,
        not_found_returns_none: bool = False,
    ) -> dict[str, Any] | None:
        self._ensure_configured()
        retries = 3
        last_error: ClerkApiError | None = None

        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=self._api_url,
                    timeout=self._timeout,
                    headers=self._headers(),
                    transport=self._transport,
                    trust_env=False,
                ) as client:
                    response = await client.request(
                        method,
                        path,
                        params=params,
                        json=json_body,
                    )
            except httpx.HTTPError as exc:
                if attempt >= retries:
                    raise ClerkApiError(f"Clerk request failed: {exc}") from exc
                await self._sleep(0.25 * attempt)
                continue

            payload = self._json_or_none(response)
            if response.status_code == 404 and not_found_returns_none:
                return None
            if response.status_code < 400:
                return payload if isinstance(payload, dict) else {}

            error = ClerkApiError(
                f"Clerk API request failed with status {response.status_code}.",
                status_code=response.status_code,
                payload=payload,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                last_error = error
                await self._sleep(0.25 * (2 ** (attempt - 1)))
                continue
            raise error

        if last_error is not None:
            raise last_error
        raise ClerkApiError("Clerk API request failed.")

    @staticmethod
    def _json_or_none(response: httpx.Response) -> Any | None:
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _extract_primary_email(user: dict[str, Any]) -> str:
        email_addresses = user.get("email_addresses")
        if isinstance(email_addresses, list):
            for item in email_addresses:
                if not isinstance(item, dict):
                    continue
                email = str(item.get("email_address") or "").strip().lower()
                if email:
                    return email
        return ""

    async def get_user(self, clerk_user_id: str) -> dict[str, Any] | None:
        normalized = clerk_user_id.strip()
        if not normalized:
            return None
        return await self._request(
            "GET",
            f"/v1/users/{normalized}",
            not_found_returns_none=True,
        )

    async def list_users(self, *, limit: int = 20) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/v1/users",
            params=[("limit", str(max(1, min(int(limit), 500))))],
        )
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return [item for item in rows if isinstance(item, dict)]

    async def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        normalized = email.strip().lower()
        if not normalized:
            return None

        filters = [
            [("email_address", normalized), ("limit", "10")],
            [("query", normalized), ("limit", "10")],
        ]
        for params in filters:
            payload = await self._request("GET", "/v1/users", params=params)
            rows = payload.get("data")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if self._extract_primary_email(row) == normalized:
                    return row
        return None

    async def find_user_by_external_id(self, external_id: str) -> dict[str, Any] | None:
        normalized = external_id.strip()
        if not normalized:
            return None
        payload = await self._request(
            "GET",
            "/v1/users",
            params=[("external_id", normalized), ("limit", "10")],
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("external_id") or "").strip() == normalized:
                return row
        return None

    async def create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                "/v1/users",
                json_body=payload,
            )
            or {}
        )

    async def delete_user(self, clerk_user_id: str) -> dict[str, Any]:
        return (
            await self._request(
                "DELETE",
                f"/v1/users/{clerk_user_id.strip()}",
            )
            or {}
        )

    async def create_session(self, user_id: str) -> dict[str, Any]:
        return (
            await self._request(
                "POST",
                "/v1/sessions",
                json_body={"user_id": user_id.strip()},
            )
            or {}
        )

    async def create_session_token(
        self,
        session_id: str,
        *,
        authorized_party: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        normalized_party = (authorized_party or "").strip()
        if normalized_party:
            payload["azp"] = normalized_party
        return (
            await self._request(
                "POST",
                f"/v1/sessions/{session_id.strip()}/tokens",
                json_body=payload,
            )
            or {}
        )

    async def update_user(self, clerk_user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return (
            await self._request(
                "PATCH",
                f"/v1/users/{clerk_user_id.strip()}",
                json_body=payload,
            )
            or {}
        )

    async def update_user_metadata(
        self,
        clerk_user_id: str,
        *,
        public_metadata: dict[str, Any] | None = None,
        private_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if public_metadata is not None:
            payload["public_metadata"] = public_metadata
        if private_metadata is not None:
            payload["private_metadata"] = private_metadata
        return (
            await self._request(
                "PATCH",
                f"/v1/users/{clerk_user_id.strip()}/metadata",
                json_body=payload,
            )
            or {}
        )


clerk_client = ClerkClient()
