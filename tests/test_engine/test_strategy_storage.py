from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.strategy import (
    EXAMPLE_PATH,
    StrategyDslValidationException,
    load_strategy_payload,
)
from src.engine.strategy.storage import (
    StrategyPatchApplyError,
    StrategyRevisionNotFoundError,
    StrategyStorageNotFoundError,
    StrategyVersionConflictError,
    diff_strategy_versions,
    get_strategy_version_payload,
    list_strategy_versions,
    patch_strategy_dsl,
    rollback_strategy_dsl,
    upsert_strategy_dsl,
    validate_stored_strategy,
)
from src.models.session import Session as AgentSession
from src.models.strategy_revision import StrategyRevision
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


@pytest.mark.asyncio
async def test_patch_strategy_updates_payload_and_increments_version(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_patch_a@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    patched = await patch_strategy_dsl(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        expected_version=1,
        patch_ops=[
            {"op": "test", "path": "/strategy/name", "value": payload["strategy"]["name"]},
            {"op": "replace", "path": "/strategy/name", "value": "Patched Strategy Name"},
        ],
    )

    assert patched.strategy.id == created.strategy.id
    assert patched.receipt.version == 2
    assert patched.strategy.dsl_payload["strategy"]["name"] == "Patched Strategy Name"
    assert patched.receipt.payload_hash != created.receipt.payload_hash


@pytest.mark.asyncio
async def test_patch_strategy_rejects_version_conflict(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_patch_b@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    with pytest.raises(StrategyVersionConflictError):
        await patch_strategy_dsl(
            db_session,
            session_id=session.id,
            strategy_id=created.strategy.id,
            expected_version=999,
            patch_ops=[{"op": "replace", "path": "/strategy/name", "value": "Nope"}],
        )


@pytest.mark.asyncio
async def test_patch_strategy_rejects_invalid_patch_path(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_patch_c@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    with pytest.raises(StrategyPatchApplyError):
        await patch_strategy_dsl(
            db_session,
            session_id=session.id,
            strategy_id=created.strategy.id,
            patch_ops=[{"op": "replace", "path": "/trade/long/exits/99/name", "value": "bad-index"}],
        )


@pytest.mark.asyncio
async def test_patch_strategy_rejects_semantic_or_schema_invalid_result(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_patch_d@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)

    with pytest.raises(StrategyDslValidationException):
        await patch_strategy_dsl(
            db_session,
            session_id=session.id,
            strategy_id=created.strategy.id,
            patch_ops=[{"op": "remove", "path": "/timeframe"}],
        )


@pytest.mark.asyncio
async def test_patch_strategy_rejects_cross_user_access(db_session: AsyncSession) -> None:
    _, session_a = await _create_user_and_session(db_session, email="dsl_storage_patch_e1@example.com")
    _, session_b = await _create_user_and_session(db_session, email="dsl_storage_patch_e2@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)
    created = await upsert_strategy_dsl(db_session, session_id=session_a.id, dsl_payload=payload)

    with pytest.raises(StrategyStorageNotFoundError):
        await patch_strategy_dsl(
            db_session,
            session_id=session_b.id,
            strategy_id=created.strategy.id,
            patch_ops=[{"op": "replace", "path": "/strategy/name", "value": "forbidden"}],
        )


@pytest.mark.asyncio
async def test_strategy_version_history_list_get_and_diff(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_history_a@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)
    await patch_strategy_dsl(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        expected_version=1,
        patch_ops=[
            {"op": "replace", "path": "/strategy/name", "value": "History Patched Name"},
        ],
    )

    versions = await list_strategy_versions(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        limit=10,
    )
    assert [item.version for item in versions] == [2, 1]
    assert versions[0].change_type == "patch"
    assert versions[1].change_type == "create"

    v1 = await get_strategy_version_payload(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        version=1,
    )
    v2 = await get_strategy_version_payload(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        version=2,
    )
    assert v1.dsl_payload["strategy"]["name"] == payload["strategy"]["name"]
    assert v2.dsl_payload["strategy"]["name"] == "History Patched Name"

    diff = await diff_strategy_versions(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        from_version=1,
        to_version=2,
    )
    assert diff.op_count >= 1
    assert any(item.get("path") == "/strategy/name" for item in diff.patch_ops)


@pytest.mark.asyncio
async def test_strategy_rollback_creates_new_head_version(db_session: AsyncSession) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_history_b@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)
    await patch_strategy_dsl(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        expected_version=1,
        patch_ops=[
            {"op": "replace", "path": "/strategy/name", "value": "Rollback Candidate"},
        ],
    )

    rolled_back = await rollback_strategy_dsl(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        target_version=1,
        expected_version=2,
    )
    assert rolled_back.receipt.version == 3
    assert rolled_back.strategy.dsl_payload["strategy"]["name"] == payload["strategy"]["name"]

    versions = await list_strategy_versions(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        limit=10,
    )
    assert [item.version for item in versions[:3]] == [3, 2, 1]
    assert versions[0].change_type == "rollback"
    assert versions[0].source_version == 1


@pytest.mark.asyncio
async def test_strategy_version_get_falls_back_to_current_for_legacy_rows(
    db_session: AsyncSession,
) -> None:
    _, session = await _create_user_and_session(db_session, email="dsl_storage_history_c@example.com")
    payload = load_strategy_payload(EXAMPLE_PATH)

    created = await upsert_strategy_dsl(db_session, session_id=session.id, dsl_payload=payload)
    await db_session.execute(
        delete(StrategyRevision).where(StrategyRevision.strategy_id == created.strategy.id),
    )
    await db_session.commit()

    current = await get_strategy_version_payload(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        version=1,
    )
    assert current.receipt.change_type == "legacy_current"
    assert current.dsl_payload["strategy"]["name"] == payload["strategy"]["name"]

    listed = await list_strategy_versions(
        db_session,
        session_id=session.id,
        strategy_id=created.strategy.id,
        limit=10,
    )
    assert len(listed) == 1
    assert listed[0].change_type == "legacy_current"

    with pytest.raises(StrategyRevisionNotFoundError):
        await get_strategy_version_payload(
            db_session,
            session_id=session.id,
            strategy_id=created.strategy.id,
            version=2,
        )
