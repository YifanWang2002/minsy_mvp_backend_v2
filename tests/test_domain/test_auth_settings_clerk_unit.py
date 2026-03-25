from __future__ import annotations

from pathlib import Path

import pytest

from packages.shared_settings.schema.settings import Settings


def _write_env_file(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_settings_parses_clerk_authorized_parties_json(tmp_path: Path) -> None:
    env_file = tmp_path / "auth.env"
    _write_env_file(
        env_file,
        """
        OPENAI_API_KEY=test-openai-key
        SECRET_KEY=test-secret
        AUTH_MODE=hybrid
        CLERK_FRONTEND_API_URL=https://example.clerk.accounts.dev
        CLERK_BACKEND_API_URL=https://api.clerk.com
        CLERK_JWKS_URL=https://example.clerk.accounts.dev/.well-known/jwks.json
        CLERK_JWT_KEY=test-public-key
        CLERK_PUBLISHABLE_KEY=pk_test_123
        CLERK_SECRET_KEY=sk_test_123
        CLERK_AUTHORIZED_PARTIES=["http://localhost:3000","https://dev.minsyai.com"]
        """,
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.auth_mode == "hybrid"
    assert settings.clerk_authorized_parties == [
        "http://localhost:3000",
        "https://dev.minsyai.com",
    ]


def test_settings_rejects_missing_clerk_values_when_auth_mode_is_clerk(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "auth.env"
    _write_env_file(
        env_file,
        """
        OPENAI_API_KEY=test-openai-key
        SECRET_KEY=test-secret
        AUTH_MODE=clerk
        CLERK_FRONTEND_API_URL=https://example.clerk.accounts.dev
        CLERK_PUBLISHABLE_KEY=pk_test_123
        """,
    )

    with pytest.raises(ValueError, match="Missing Clerk configuration"):
        Settings(_env_file=str(env_file))


def test_settings_accepts_legacy_mode_without_clerk_configuration(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "auth.env"
    _write_env_file(
        env_file,
        """
        OPENAI_API_KEY=test-openai-key
        SECRET_KEY=test-secret
        AUTH_MODE=legacy
        """,
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.auth_mode == "legacy"
    assert settings.clerk_publishable_key == ""
