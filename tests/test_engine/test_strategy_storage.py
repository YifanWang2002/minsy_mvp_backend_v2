from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.strategy import (
    EXAMPLE_PATH,
    StrategyDslValidationException,
    load_strategy_payload,
)
from src.engine.strategy.storage import (
    StrategyStorageNotFoundError,
    upsert_strategy_dsl,
    validate_stored_strategy,
)
from src.models.session import Session as AgentSession
from src.models.user import User


async def _create_user_and_session(
    db_session: AsyncSession,
    *,
    email: str,
) -> tuple[User, AgentSession]:
    user = User(email=email, password_hash="hashed", name=email)
    db_session.add(user)
    await db_session.flush()

    session = AgentSession(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()
    return user, session


@pytest.mark.asyncio
async def test_upsert_strategy_create_and_update_with_version_increment(
    db_session: AsyncSession,
) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_a@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    assert created.strategy.user_id == session.user_id
    assert created.receipt.user_id == session.user_id
    assert created.receipt.version == 1
    assert created.receipt.strategy_name == payload["strategy"]["name"]

    updated_payload = deepcopy(payload)
    updated_payload["strategy"]["name"] = "EMA + RSI Updated"

    updated = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        dsl_payload=updated_payload,
    )

    assert updated.strategy.id == created.strategy.id
    assert updated.receipt.version == 2
    assert updated.receipt.strategy_name == "EMA + RSI Updated"
    assert updated.receipt.last_updated_at >= created.receipt.last_updated_at
    assert updated.receipt.payload_hash != created.receipt.payload_hash
    assert updated.strategy.parameters["dsl_hash"] == updated.receipt.payload_hash


@pytest.mark.asyncio
async def test_upsert_strategy_rejects_cross_user_update(db_session: AsyncSession) -> None:
    _, session_a = await _create_user_and_session(db_session, email="dsl_storage_b1@example.com")
    _, session_b = await _create_user_and_session(db_session, email="dsl_storage_b2@example.com")

    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session_a.id, dsl_payload=payload)

    with pytest.raises(StrategyStorageNotFoundError):
        await upsert_strategy_dsl(
            db_session,
            session_id=session_b.id,
            strategy_id=created.strategy.id,
            dsl_payload=payload,
        )


@pytest.mark.asyncio
async def test_upsert_strategy_rejects_invalid_payload(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_c@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload.pop("timeframe", None)

    with pytest.raises(StrategyDslValidationException) as exc_info:
        await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    assert any(item.code == "MISSING_REQUIRED_FIELD" for item in exc_info.value.errors)


@pytest.mark.asyncio
async def test_upsert_strategy_requires_valid_session(db_session: AsyncSession) -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)

    with pytest.raises(StrategyStorageNotFoundError):
        await upsert_strategy_dsl(
            db_session,
            session_id=uuid4(),
            dsl_payload=payload,
        )


@pytest.mark.asyncio
async def test_upsert_strategy_requires_existing_strategy_id(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_d@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    with pytest.raises(StrategyStorageNotFoundError):
        await upsert_strategy_dsl(
            db_session,
            session_id=session.id,
            strategy_id=uuid4(),
            dsl_payload=payload,
        )


@pytest.mark.asyncio
async def test_validate_stored_strategy_round_trip(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_e@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    created = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=payload,
    )

    validation = await validate_stored_strategy(db_session, strategy_id=created.strategy.id)
    assert validation.is_valid is True
    assert validation.errors == ()
