"""HMAC helpers for local-collector <-> remote-import API calls."""

from __future__ import annotations

import hashlib
import hmac


def build_signature_payload(
    *,
    timestamp_epoch_seconds: int,
    method: str,
    path: str,
    body: bytes,
) -> str:
    method_key = str(method).strip().upper() or "GET"
    path_key = str(path).strip() or "/"
    digest = hashlib.sha256(body).hexdigest()
    return f"{int(timestamp_epoch_seconds)}\n{method_key}\n{path_key}\n{digest}"


def sign_payload(
    *,
    secret: str,
    payload: str,
) -> str:
    return hmac.new(
        str(secret).encode("utf-8"),
        str(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
