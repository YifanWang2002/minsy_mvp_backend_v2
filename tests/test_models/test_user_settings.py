from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.models.user_settings import UserSetting


@pytest.mark.asyncio
async def test_user_settings_unique_user_id_constraint(db_session: AsyncSession) -> None:
    user = User(email="prefs_unique@example.com", password_hash="hashed", name="Prefs Unique")
    db_session.add(user)
    await db_session.flush()

    first = UserSetting(user_id=user.id, theme_mode="light", locale="en", font_scale="small")
    second = UserSetting(user_id=user.id, theme_mode="dark", locale="zh", font_scale="large")
    db_session.add(first)
    await db_session.commit()

    db_session.add(second)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_user_settings_check_constraint_rejects_invalid_theme_mode(
    db_session: AsyncSession,
) -> None:
    user = User(email="prefs_invalid_theme@example.com", password_hash="hashed", name="Prefs Invalid")
    db_session.add(user)
    await db_session.flush()

    db_session.add(
        UserSetting(
            user_id=user.id,
            theme_mode="blue",
            locale="en",
            font_scale="default",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_user_settings_cascade_delete_on_user(db_session: AsyncSession) -> None:
    user = User(email="prefs_cascade@example.com", password_hash="hashed", name="Prefs Cascade")
    db_session.add(user)
    await db_session.flush()

    setting = UserSetting(user_id=user.id, theme_mode="dark", locale="zh", font_scale="large")
    db_session.add(setting)
    await db_session.commit()

    loaded = await db_session.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    assert loaded is not None

    await db_session.delete(user)
    await db_session.commit()

    deleted = await db_session.scalar(select(UserSetting).where(UserSetting.user_id == user.id))
    assert deleted is None
