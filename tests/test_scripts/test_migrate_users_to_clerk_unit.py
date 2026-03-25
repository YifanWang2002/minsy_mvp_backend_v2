from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.migrate_users_to_clerk import (
    LocalUserSnapshot,
    _plan_migration,
    build_clerk_create_payload,
)


class _FakeClerkClient:
    def __init__(self, *, existing_by_email: dict[str, dict] | None = None) -> None:
        self.existing_by_email = existing_by_email or {}
        self.created_payloads: list[dict] = []
        self.updated_payloads: list[tuple[str, dict]] = []
        self.updated_metadata: list[tuple[str, dict, dict]] = []

    async def get_user(self, clerk_user_id: str) -> dict | None:
        return None

    async def find_user_by_external_id(self, external_id: str) -> dict | None:
        return None

    async def find_user_by_email(self, email: str) -> dict | None:
        return self.existing_by_email.get(email)

    async def create_user(self, payload: dict) -> dict:
        self.created_payloads.append(payload)
        return {"id": "user_created", **payload}

    async def update_user(self, clerk_user_id: str, payload: dict) -> dict:
        self.updated_payloads.append((clerk_user_id, payload))
        return {"id": clerk_user_id, **payload}

    async def update_user_metadata(
        self,
        clerk_user_id: str,
        *,
        public_metadata: dict | None = None,
        private_metadata: dict | None = None,
    ) -> dict:
        self.updated_metadata.append(
            (
                clerk_user_id,
                public_metadata or {},
                private_metadata or {},
            )
        )
        return {"id": clerk_user_id}


def _snapshot(
    *,
    password_hash: str | None = "$2b$12$example",
    clerk_user_id: str | None = None,
) -> LocalUserSnapshot:
    return LocalUserSnapshot(
        user_id="00000000-0000-0000-0000-000000000001",
        email="alice@example.com",
        name="Alice Smith",
        password_hash=password_hash,
        clerk_user_id=clerk_user_id,
        current_tier="go",
        auth_provider="legacy_password",
    )


def test_build_clerk_create_payload_uses_bcrypt_digest_and_name_split() -> None:
    payload = build_clerk_create_payload(_snapshot())

    assert payload["email_address"] == ["alice@example.com"]
    assert payload["password_hasher"] == "bcrypt"
    assert payload["password_digest"] == "$2b$12$example"
    assert payload["first_name"] == "Alice"
    assert payload["last_name"] == "Smith"
    assert payload["public_metadata"]["tier"] == "go"


@pytest.mark.asyncio
async def test_plan_migration_dry_run_marks_user_ready_for_import() -> None:
    client = _FakeClerkClient()

    result, payload = await _plan_migration(_snapshot(), client, apply=False)

    assert result.action == "would_create"
    assert result.reason == "ready_for_import"
    assert payload is not None
    assert client.created_payloads == []


@pytest.mark.asyncio
async def test_plan_migration_apply_matches_existing_remote_user_idempotently() -> None:
    client = _FakeClerkClient(
        existing_by_email={
            "alice@example.com": {
                "id": "user_existing",
                "email_addresses": [{"email_address": "alice@example.com"}],
            }
        }
    )

    result, payload = await _plan_migration(_snapshot(), client, apply=True)

    assert result.action == "matched_existing"
    assert result.remote_clerk_user_id == "user_existing"
    assert payload is not None
    assert client.created_payloads == []
    assert client.updated_payloads[0][0] == "user_existing"
    assert client.updated_metadata[0][0] == "user_existing"


@pytest.mark.asyncio
async def test_plan_migration_skips_user_without_password_hash() -> None:
    client = _FakeClerkClient()

    result, payload = await _plan_migration(
        _snapshot(password_hash=None),
        client,
        apply=True,
    )

    assert result.action == "skipped"
    assert result.reason == "missing_password_hash"
    assert payload is None
