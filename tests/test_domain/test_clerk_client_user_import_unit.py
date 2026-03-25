from __future__ import annotations

import httpx
import pytest

from packages.infra.providers.clerk.client import ClerkApiError, ClerkClient


@pytest.mark.asyncio
async def test_clerk_client_create_user_sends_expected_payload_and_headers() -> None:
    seen_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"id": "user_123", "external_id": "local-user-1"},
        )

    client = ClerkClient(
        secret_key="sk_test_123",
        api_url="https://api.clerk.test",
        api_version="2025-11-10",
        transport=httpx.MockTransport(handler),
    )

    payload = {
        "email_address": ["alice@example.com"],
        "password_digest": "$2b$12$hashed",
        "password_hasher": "bcrypt",
        "external_id": "local-user-1",
    }
    response = await client.create_user(payload)

    assert response["id"] == "user_123"
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/users"
    assert request.headers["Authorization"] == "Bearer sk_test_123"
    assert request.headers["Clerk-Version"] == "2025-11-10"
    assert request.read().decode("utf-8").count('"password_hasher":"bcrypt"') == 1


@pytest.mark.asyncio
async def test_clerk_client_retries_on_429_and_succeeds() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"errors": [{"message": "slow down"}]})
        return httpx.Response(200, json={"id": "user_retry"})

    async def no_sleep(_: float) -> None:
        return None

    client = ClerkClient(
        secret_key="sk_test_123",
        api_url="https://api.clerk.test",
        transport=httpx.MockTransport(handler),
        sleep=no_sleep,
    )

    response = await client.create_user({"email_address": ["retry@example.com"]})

    assert response["id"] == "user_retry"
    assert attempts == 2


@pytest.mark.asyncio
async def test_clerk_client_raises_after_retry_budget_exhausted() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"errors": [{"message": "unavailable"}]})

    async def no_sleep(_: float) -> None:
        return None

    client = ClerkClient(
        secret_key="sk_test_123",
        api_url="https://api.clerk.test",
        transport=httpx.MockTransport(handler),
        sleep=no_sleep,
    )

    with pytest.raises(ClerkApiError, match="status 503"):
        await client.create_user({"email_address": ["retry@example.com"]})


@pytest.mark.asyncio
async def test_clerk_client_creates_session_token_with_authorized_party_payload() -> None:
    seen_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={"object": "token", "jwt": "jwt_test_123"},
        )

    client = ClerkClient(
        secret_key="sk_test_123",
        api_url="https://api.clerk.test",
        api_version="2025-11-10",
        transport=httpx.MockTransport(handler),
    )

    response = await client.create_session_token(
        "sess_test_123",
        authorized_party="http://localhost:3000",
    )

    assert response["jwt"] == "jwt_test_123"
    assert len(seen_requests) == 1
    request = seen_requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/sessions/sess_test_123/tokens"
    assert request.read().decode("utf-8") == '{"azp":"http://localhost:3000"}'
