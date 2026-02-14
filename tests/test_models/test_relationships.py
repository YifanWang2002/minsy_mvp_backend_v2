from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.session import Message
from src.models.session import Session as AgentSession
from src.models.user import User


@pytest.mark.asyncio
async def test_user_sessions_messages_relationship_query(db_session: AsyncSession) -> None:
    user = User(
        email="relations@example.com",
        password_hash="hashed_pw",
        name="Relations Tester",
    )
    session = AgentSession(
        user=user,
        current_phase="strategy",
        status="active",
    )
    message_1 = Message(
        session=session,
        role="user",
        content="first",
        phase="strategy",
    )
    message_2 = Message(
        session=session,
        role="assistant",
        content="second",
        phase="strategy",
    )

    db_session.add_all([user, session, message_1, message_2])
    await db_session.commit()

    stmt = (
        select(User)
        .options(selectinload(User.sessions).selectinload(AgentSession.messages))
        .where(User.email == "relations@example.com")
    )
    result = await db_session.scalar(stmt)

    assert result is not None
    assert len(result.sessions) == 1
    assert len(result.sessions[0].messages) == 2
    assert {msg.role for msg in result.sessions[0].messages} == {"user", "assistant"}
