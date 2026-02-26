"""Credential encryption helpers for broker secrets."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from packages.shared_settings.schema.settings import settings


class CredentialEncryptionError(RuntimeError):
    """Raised when credential payload cannot be encrypted/decrypted."""


CURRENT_ENCRYPTION_VERSION = "fernet_v1"


def _derive_fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def credential_key_fingerprint(payload: dict[str, Any]) -> str:
    """Build a stable credential fingerprint without exposing raw keys."""
    key_candidates = (
        "APCA-API-KEY-ID",
        "api_key",
        "key",
    )
    key_id = ""
    for candidate in key_candidates:
        value = payload.get(candidate)
        if isinstance(value, str) and value.strip():
            key_id = value.strip()
            break
    if not key_id:
        # Fall back to payload-wide deterministic hash when key field is absent.
        try:
            serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise CredentialEncryptionError("Credential payload is not JSON serializable.") from exc
        key_id = serialized

    salted = f"{settings.effective_trading_credentials_secret}:{key_id}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


class CredentialCipher:
    """Symmetric encryption wrapper for broker credential payloads."""

    def __init__(self, secret: str | None = None) -> None:
        resolved_secret = (secret or settings.effective_trading_credentials_secret).strip()
        if not resolved_secret:
            raise CredentialEncryptionError("Trading credential secret is empty.")
        self._fernet = Fernet(_derive_fernet_key(resolved_secret))
        self.encryption_version = CURRENT_ENCRYPTION_VERSION

    def encrypt(self, payload: dict[str, Any]) -> str:
        try:
            serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise CredentialEncryptionError("Credential payload is not JSON serializable.") from exc
        token = self._fernet.encrypt(serialized.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, encrypted_payload: str) -> dict[str, Any]:
        try:
            decrypted = self._fernet.decrypt(encrypted_payload.encode("utf-8"))
            payload = json.loads(decrypted.decode("utf-8"))
        except (InvalidToken, ValueError, json.JSONDecodeError) as exc:
            raise CredentialEncryptionError("Failed to decrypt broker credentials.") from exc
        if not isinstance(payload, dict):
            raise CredentialEncryptionError("Decrypted broker credentials are invalid.")
        return payload
