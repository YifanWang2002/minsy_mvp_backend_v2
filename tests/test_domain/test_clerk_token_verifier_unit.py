from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException

from packages.domain.user.services.clerk_token_verifier import ClerkTokenVerifier

_TEST_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIICeQIBADANBgkqhkiG9w0BAQEFAASCAmMwggJfAgEAAoGBANZduwEN0ylXkggr
8mjL/3DLT/fpXxLq/sTpTtv0z7mP8nCGYq8zMJLCZQW6vOlgAuNVZYlXvXHzY0uQ
6Whmmt4z2CO2dF/L9qzK7ZFYgKF+NboG/TYSO2EsjuPuTSNm38KLsAeQKqtyLQtN
XN3LnkL6aE+QQ+3WwWRVSMWgx+3ZAgMBAAECgYEAj+M+YMi80mU7WkzVW86CWV2/
AbMd4/7kn5vTGQVMYUvj+e/aUatUkU32rU/Y+fU+OwXZL8U7Hj+2iMRuR2uHyx9f
7lqQYjXq0qqPELvF3jzAh7aVHZuzAjZnxqCAwU0gxSBjyVydLFWJikFQqVvCO7Jo
gi7PdWB119DusZWCx2ECQQD6sWKuUaUXAOMN9P0FTNW8BAAvXVKwVnqkjiFb07L6
aY68ha1HgVRaKLwNywmFzjASBQTutOXwQNp0JNTYZgSlAkEA2ud5bPu+6x8+J7bD
0aUf+Y55eNrcqXV1H8mP1grbqos1lVBy1v3nWu4jchzOgkqczZIEyhOBRb+nw5AN
Y+aaJQJBAN8SMMkEhW5ur5ufv/WTZSykMrXyyL14djEu96gKPFxuyUAfgwz5m+GO
FagAXzzdOBEQvk7aUTDzxG9Mxsi4HrECQQDJT8KddU74j7zrbOrcq8yiBmKzwCLa
PMi/uO/sWgP17RwT+u4BxXK0bvhuAwvvSoq1iqmY5SMnb7/q21lVHEd5AkEAiSnk
IxaMzu6RfCNZWokIQMjgFiQNyrajHZKW8hZp+STdJnZxadi0KgNdLUI0KOiWzLbv
gQmU5UikcqXDyQR0Pw==
-----END PRIVATE KEY-----"""

_TEST_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDWXbsBDdMpV5IIK/Joy/9wy0/3
6V8S6v7E6U7b9M+5j/JwhmKvMzCSwmUFurzpYALjVWWJV71x82NLkOloZpreM9gj
tnRfy/asyu2RWIChfjW6Bv02EjthLI7j7k0jZt/Ci7AHkCqrci0LTVzdy55C+mhP
kEPt1sFkVUjFoMft2QIDAQAB
-----END PUBLIC KEY-----"""


def _issue_token(
    *,
    sub: str = "user_clerk_123",
    azp: str | None = "https://dev.minsyai.com",
    expires_at: datetime | None = None,
) -> str:
    now = datetime.now(UTC)
    expiry = expires_at or (now + timedelta(minutes=5))
    payload = {
        "sub": sub,
        "sid": "sess_123",
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
    }
    if azp is not None:
        payload["azp"] = azp
    return jwt.encode(
        payload,
        _TEST_PRIVATE_KEY,
        algorithm="RS256",
    )


def test_clerk_token_verifier_accepts_valid_rs256_token() -> None:
    verifier = ClerkTokenVerifier(
        jwt_key=_TEST_PUBLIC_KEY,
        authorized_parties=["https://dev.minsyai.com"],
    )

    verified = verifier.verify(_issue_token())

    assert verified.clerk_user_id == "user_clerk_123"
    assert verified.session_id == "sess_123"
    assert verified.authorized_party == "https://dev.minsyai.com"


def test_clerk_token_verifier_rejects_mismatched_authorized_party() -> None:
    verifier = ClerkTokenVerifier(
        jwt_key=_TEST_PUBLIC_KEY,
        authorized_parties=["https://app.minsyai.com"],
    )

    with pytest.raises(HTTPException, match="Invalid authorized party."):
        verifier.verify(_issue_token())


def test_clerk_token_verifier_allows_missing_authorized_party_claim() -> None:
    verifier = ClerkTokenVerifier(
        jwt_key=_TEST_PUBLIC_KEY,
        authorized_parties=["https://app.minsyai.com"],
    )

    verified = verifier.verify(_issue_token(azp=None))

    assert verified.clerk_user_id == "user_clerk_123"
    assert verified.authorized_party is None


def test_clerk_token_verifier_rejects_expired_token() -> None:
    verifier = ClerkTokenVerifier(
        jwt_key=_TEST_PUBLIC_KEY,
        authorized_parties=["https://dev.minsyai.com"],
    )

    with pytest.raises(HTTPException, match="Token has expired."):
        verifier.verify(
            _issue_token(expires_at=datetime.now(UTC) - timedelta(seconds=10))
        )


def test_clerk_token_verifier_accepts_wildcard_localhost_authorized_party() -> None:
    verifier = ClerkTokenVerifier(
        jwt_key=_TEST_PUBLIC_KEY,
        authorized_parties=["http://localhost:*", "http://127.0.0.1:*"],
    )

    verified = verifier.verify(_issue_token(azp="http://localhost:63756"))

    assert verified.authorized_party == "http://localhost:63756"
