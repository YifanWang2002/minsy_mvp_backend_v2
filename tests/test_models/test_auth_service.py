from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.auth_service import AuthService


@pytest.mark.asyncio
async def test_password_not_stored_in_plaintext_and_bcrypt_verify_passes(
    db_session: AsyncSession,
) -> None:
    raw_password = "pass1234"
    service = AuthService(db_session)

    user, _ = await service.register(
        email="hash_test@example.com",
        password=raw_password,
        name="Hash User",
    )

    assert user.password_hash != raw_password
    assert user.password_hash.startswith("$2")
    assert service.verify_password(raw_password, user.password_hash)
