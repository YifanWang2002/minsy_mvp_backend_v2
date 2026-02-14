from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.session import Message
from src.models.user import User, UserProfile


@pytest.mark.asyncio
async def test_unique_email_constraint(db_session: AsyncSession) -> None:
    user_1 = User(email="unique@example.com", password_hash="p1", name="u1")
    user_2 = User(email="unique@example.com", password_hash="p2", name="u2")

    db_session.add(user_1)
    await db_session.commit()

    db_session.add(user_2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_foreign_key_constraint_on_message_session_id(db_session: AsyncSession) -> None:
    invalid_message = Message(
        session_id=uuid4(),
        role="user",
        content="invalid fk",
        phase="kyc",
    )
    db_session.add(invalid_message)

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_risk_tolerance_allowed_values_constraint(db_session: AsyncSession) -> None:
    user = User(email="risk@example.com", password_hash="pw", name="Risk User")
    db_session.add(user)
    await db_session.flush()

    invalid_profile = UserProfile(
        user_id=user.id,
        trading_years_bucket="years_1_3",
        risk_tolerance="extreme",
        return_expectation="growth",
        kyc_status="incomplete",
    )
    db_session.add(invalid_profile)

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_return_expectation_allowed_values_constraint(db_session: AsyncSession) -> None:
    user = User(email="return@example.com", password_hash="pw", name="Return User")
    db_session.add(user)
    await db_session.flush()

    invalid_profile = UserProfile(
        user_id=user.id,
        trading_years_bucket="years_3_5",
        risk_tolerance="moderate",
        return_expectation="moonshot",
        kyc_status="incomplete",
    )
    db_session.add(invalid_profile)

    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()
