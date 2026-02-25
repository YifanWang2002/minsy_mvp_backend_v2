"""Telegram callback-data codec for trade approval actions."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID

from src.config import settings

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True, slots=True)
class ParsedApprovalCallback:
    request_id: UUID
    action: str
    expires_at: datetime
    expired: bool


class TelegramApprovalCodec:
    """Encode/verify compact callback_data payload with HMAC signature."""

    PREFIX = "apv1"
    _ACTION_TO_CODE = {"approve": "a", "reject": "r"}
    _CODE_TO_ACTION = {"a": "approve", "r": "reject"}
    _SIG_LEN = 10
    _MAX_CALLBACK_DATA_LEN = 64

    def __init__(self, *, secret: str | None = None) -> None:
        self._secret = (
            secret
            or settings.telegram_approval_callback_secret.strip()
            or settings.secret_key.strip()
        )

    @classmethod
    def looks_like(cls, raw: str) -> bool:
        return str(raw).strip().startswith(f"{cls.PREFIX}:")

    def encode(
        self,
        *,
        request_id: UUID,
        action: str,
        expires_at: datetime,
    ) -> str:
        normalized_action = str(action).strip().lower()
        action_code = self._ACTION_TO_CODE.get(normalized_action)
        if action_code is None:
            raise ValueError("action must be approve/reject.")
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        exp_token = _to_base36(int(expires_at.timestamp()))
        body = f"{action_code}:{request_id.hex}:{exp_token}"
        signature = self._sign(body)[: self._SIG_LEN]
        callback_data = f"{self.PREFIX}:{body}:{signature}"
        if len(callback_data) > self._MAX_CALLBACK_DATA_LEN:
            raise ValueError("callback_data exceeds Telegram 64-byte limit.")
        return callback_data

    def decode(
        self,
        raw: str,
        *,
        now: datetime | None = None,
    ) -> tuple[ParsedApprovalCallback | None, str | None]:
        text = str(raw).strip()
        parts = text.split(":")
        if len(parts) != 5:
            return None, "format_invalid"
        prefix, action_code, request_id_hex, exp_token, signature = parts
        if prefix != self.PREFIX:
            return None, "prefix_invalid"
        action = self._CODE_TO_ACTION.get(action_code)
        if action is None:
            return None, "action_invalid"
        body = f"{action_code}:{request_id_hex}:{exp_token}"
        expected = self._sign(body)[: self._SIG_LEN]
        if not hmac.compare_digest(signature, expected):
            return None, "signature_invalid"
        try:
            request_id = UUID(hex=request_id_hex)
        except ValueError:
            return None, "request_id_invalid"
        try:
            exp_epoch = _from_base36(exp_token)
        except ValueError:
            return None, "expires_invalid"
        expires_at = datetime.fromtimestamp(exp_epoch, tz=UTC)
        now_ts = now or datetime.now(UTC)
        if now_ts.tzinfo is None:
            now_ts = now_ts.replace(tzinfo=UTC)
        parsed = ParsedApprovalCallback(
            request_id=request_id,
            action=action,
            expires_at=expires_at,
            expired=now_ts > expires_at,
        )
        return parsed, None

    def _sign(self, raw: str) -> str:
        return hmac.new(
            self._secret.encode("utf-8"),
            raw.encode("utf-8"),
            sha256,
        ).hexdigest()


def _to_base36(value: int) -> str:
    if value < 0:
        raise ValueError("base36 encoder only supports non-negative integers.")
    if value == 0:
        return "0"
    output: list[str] = []
    current = value
    while current > 0:
        current, remainder = divmod(current, 36)
        output.append(_ALPHABET[remainder])
    return "".join(reversed(output))


def _from_base36(value: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError("base36 input cannot be empty.")
    result = 0
    for char in text:
        idx = _ALPHABET.find(char)
        if idx < 0:
            raise ValueError("invalid base36 input.")
        result = result * 36 + idx
    return result
