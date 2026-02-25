from __future__ import annotations

from src.engine.execution.credentials import (
    CURRENT_ENCRYPTION_VERSION,
    CredentialCipher,
    credential_key_fingerprint,
)


def test_credential_cipher_round_trip() -> None:
    cipher = CredentialCipher("unit-test-secret")
    payload = {"api_key": "demo_key", "api_secret": "demo_secret"}

    encrypted = cipher.encrypt(payload)
    decrypted = cipher.decrypt(encrypted)

    assert encrypted != '{"api_key":"demo_key","api_secret":"demo_secret"}'
    assert decrypted == payload
    assert cipher.encryption_version == CURRENT_ENCRYPTION_VERSION


def test_credential_key_fingerprint_is_stable_by_key_id() -> None:
    first = credential_key_fingerprint({"api_key": "KEY-A", "api_secret": "secret-1"})
    second = credential_key_fingerprint({"api_key": "KEY-A", "api_secret": "secret-2"})
    third = credential_key_fingerprint({"api_key": "KEY-B", "api_secret": "secret-2"})

    assert first == second
    assert first != third
    assert len(first) == 64


def test_credential_key_fingerprint_fallback_without_key_id() -> None:
    first = credential_key_fingerprint({"token": "abc"})
    second = credential_key_fingerprint({"token": "abc"})
    third = credential_key_fingerprint({"token": "xyz"})

    assert first == second
    assert first != third
    assert len(first) == 64
