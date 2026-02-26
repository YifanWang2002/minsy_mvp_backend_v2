"""Strategy DSL persistence helpers backed by PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import jsonpatch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.strategy.errors import (
    StrategyDslValidationException,
    StrategyDslValidationResult,
)
from packages.domain.strategy.parser import build_parsed_strategy
from packages.domain.strategy.pipeline import validate_strategy_payload
from packages.infra.db.models.session import Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.strategy_revision import StrategyRevision


@dataclass(frozen=True, slots=True)
class StrategyMetadataReceipt:
    """Traceable strategy metadata returned by persistence operations."""

    strategy_id: UUID
    user_id: UUID
    session_id: UUID
    strategy_name: str
    dsl_version: str
    version: int
    status: str
    timeframe: str
    symbol_count: int
    payload_hash: str
    last_updated_at: datetime


@dataclass(frozen=True, slots=True)
class StrategyPersistenceResult:
    """Storage output tuple: ORM entity + serializable receipt."""

    strategy: Strategy
    receipt: StrategyMetadataReceipt


@dataclass(frozen=True, slots=True)
class StrategyRevisionReceipt:
    """Serializable metadata for one historical strategy revision."""

    strategy_id: UUID
    session_id: UUID | None
    version: int
    dsl_version: str
    payload_hash: str
    change_type: str
    source_version: int | None
    patch_op_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StrategyVersionPayload:
    """Resolved DSL payload for one strategy version."""

    strategy_id: UUID
    version: int
    dsl_payload: dict[str, Any]
    receipt: StrategyRevisionReceipt


@dataclass(frozen=True, slots=True)
class StrategyVersionDiff:
    """RFC 6902 diff between two strategy versions."""

    strategy_id: UUID
    from_version: int
    to_version: int
    patch_ops: list[dict[str, Any]]
    op_count: int
    from_payload_hash: str
    to_payload_hash: str


class StrategyStorageNotFoundError(LookupError):
    """Raised when session/strategy records cannot be resolved."""


class StrategyPatchApplyError(ValueError):
    """Raised when a JSON patch payload cannot be applied safely."""


class StrategyVersionConflictError(ValueError):
    """Raised when the caller's expected strategy version does not match current."""


class StrategyRevisionNotFoundError(LookupError):
    """Raised when the requested strategy revision does not exist."""


def _payload_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_receipt(strategy: Strategy) -> StrategyMetadataReceipt:
    updated_at = strategy.updated_at if strategy.updated_at else strategy.created_at
    payload_hash = _payload_hash(strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {})
    return StrategyMetadataReceipt(
        strategy_id=strategy.id,
        user_id=strategy.user_id,
        session_id=strategy.session_id,
        strategy_name=strategy.name,
        dsl_version=strategy.dsl_version or "",
        version=strategy.version,
        status=strategy.status,
        timeframe=strategy.timeframe,
        symbol_count=len(strategy.symbols or []),
        payload_hash=payload_hash,
        last_updated_at=_as_utc(updated_at),
    )


def _build_revision_receipt(revision: StrategyRevision) -> StrategyRevisionReceipt:
    patch_ops = revision.patch_ops if isinstance(revision.patch_ops, list) else []
    return StrategyRevisionReceipt(
        strategy_id=revision.strategy_id,
        session_id=revision.session_id,
        version=int(revision.version),
        dsl_version=revision.dsl_version or "",
        payload_hash=revision.payload_hash,
        change_type=revision.change_type,
        source_version=revision.source_version,
        patch_op_count=len(patch_ops),
        created_at=_as_utc(revision.created_at),
    )


def _build_synthetic_revision_receipt(strategy: Strategy) -> StrategyRevisionReceipt:
    payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    updated_at = strategy.updated_at if strategy.updated_at else strategy.created_at
    return StrategyRevisionReceipt(
        strategy_id=strategy.id,
        session_id=strategy.session_id,
        version=int(strategy.version),
        dsl_version=strategy.dsl_version or "",
        payload_hash=_payload_hash(payload),
        change_type="legacy_current",
        source_version=None,
        patch_op_count=0,
        created_at=_as_utc(updated_at),
    )


def _extract_dsl_version(*, payload: dict[str, Any], fallback: str | None = None) -> str | None:
    raw_dsl_version = payload.get("dsl_version")
    if isinstance(raw_dsl_version, str):
        value = raw_dsl_version.strip()
        return value or None
    if isinstance(fallback, str):
        value = fallback.strip()
        return value or None
    return None


def _normalize_patch_ops(
    patch_ops: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if patch_ops is None:
        return None

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(patch_ops):
        if not isinstance(item, dict):
            raise StrategyPatchApplyError(
                f"Patch operation at index {index} must be a JSON object.",
            )
        normalized.append(deepcopy(item))
    return normalized


def _derive_strategy_columns(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = build_parsed_strategy(payload)
    payload_hash = _payload_hash(payload)

    long_side = parsed.trade.get("long", {}) if isinstance(parsed.trade, dict) else {}
    short_side = parsed.trade.get("short", {}) if isinstance(parsed.trade, dict) else {}

    return {
        "name": parsed.strategy.name,
        "description": parsed.strategy.description or None,
        "strategy_type": "dsl",
        "symbols": list(parsed.universe.tickers),
        "timeframe": parsed.universe.timeframe,
        "parameters": {
            "dsl_hash": payload_hash,
            "factor_count": len(parsed.factors),
            "factor_ids": sorted(parsed.factors.keys()),
        },
        "entry_rules": {
            "long": long_side.get("entry"),
            "short": short_side.get("entry"),
        },
        "exit_rules": {
            "long": long_side.get("exits"),
            "short": short_side.get("exits"),
        },
        "risk_management": {
            "long_position_sizing": long_side.get("position_sizing"),
            "short_position_sizing": short_side.get("position_sizing"),
        },
        "dsl_version": parsed.dsl_version,
        "dsl_payload": payload,
        "status": "validated",
        "last_validated_at": datetime.now(UTC),
    }


async def validate_strategy_payload_or_raise(
    payload: dict[str, Any],
) -> StrategyDslValidationResult:
    result = validate_strategy_payload(payload)
    if not result.is_valid:
        raise StrategyDslValidationException(list(result.errors))
    return result


async def get_session_user_id(
    db: AsyncSession,
    *,
    session_id: UUID,
) -> UUID:
    user_id = await db.scalar(select(Session.user_id).where(Session.id == session_id))
    if user_id is None:
        raise StrategyStorageNotFoundError(f"Session not found: {session_id}")
    return user_id


async def get_strategy_or_raise(
    db: AsyncSession,
    *,
    strategy_id: UUID,
) -> Strategy:
    strategy = await db.scalar(select(Strategy).where(Strategy.id == strategy_id))
    if strategy is None:
        raise StrategyStorageNotFoundError(f"Strategy not found: {strategy_id}")
    return strategy


async def _get_owned_strategy_or_raise(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
) -> Strategy:
    session_user_id = await get_session_user_id(db, session_id=session_id)
    strategy = await get_strategy_or_raise(db, strategy_id=strategy_id)
    if strategy.user_id != session_user_id:
        raise StrategyStorageNotFoundError(
            "Strategy ownership mismatch for the provided session/user context.",
        )
    return strategy


async def _insert_revision_if_absent(
    db: AsyncSession,
    *,
    strategy: Strategy,
    version: int,
    payload: dict[str, Any],
    change_type: str,
    session_id: UUID | None,
    source_version: int | None = None,
    patch_ops: list[dict[str, Any]] | None = None,
) -> None:
    if version <= 0:
        raise StrategyRevisionNotFoundError("version must be >= 1")

    existing = await db.scalar(
        select(StrategyRevision.id).where(
            StrategyRevision.strategy_id == strategy.id,
            StrategyRevision.version == version,
        )
    )
    if existing is not None:
        return

    normalized_payload = dict(payload)
    normalized_patch_ops = _normalize_patch_ops(patch_ops)
    revision = StrategyRevision(
        strategy_id=strategy.id,
        session_id=session_id,
        version=version,
        dsl_version=_extract_dsl_version(payload=normalized_payload, fallback=strategy.dsl_version),
        dsl_payload=deepcopy(normalized_payload),
        payload_hash=_payload_hash(normalized_payload),
        change_type=change_type,
        source_version=source_version,
        patch_ops=normalized_patch_ops,
    )
    db.add(revision)
    await db.flush()


async def _ensure_current_revision_snapshot(
    db: AsyncSession,
    *,
    strategy: Strategy,
) -> None:
    payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    await _insert_revision_if_absent(
        db,
        strategy=strategy,
        version=int(strategy.version),
        payload=payload,
        change_type="bootstrap",
        session_id=strategy.session_id,
    )


async def _persist_next_strategy_version(
    db: AsyncSession,
    *,
    strategy: Strategy,
    session_id: UUID,
    current_version: int,
    next_payload: dict[str, Any],
    change_type: str,
    source_version: int | None = None,
    patch_ops: list[dict[str, Any]] | None = None,
    auto_commit: bool = True,
) -> StrategyPersistenceResult:
    columns = _derive_strategy_columns(next_payload)
    strategy.session_id = session_id
    strategy.version = current_version + 1
    strategy.updated_at = datetime.now(UTC)
    for field, value in columns.items():
        setattr(strategy, field, value)
    await db.flush()

    await _insert_revision_if_absent(
        db,
        strategy=strategy,
        version=int(strategy.version),
        payload=next_payload,
        change_type=change_type,
        session_id=session_id,
        source_version=source_version,
        patch_ops=patch_ops,
    )
    if auto_commit:
        await db.commit()
        await db.refresh(strategy)
    return StrategyPersistenceResult(strategy=strategy, receipt=_build_receipt(strategy))


async def upsert_strategy_dsl(
    db: AsyncSession,
    *,
    session_id: UUID,
    dsl_payload: dict[str, Any],
    strategy_id: UUID | None = None,
    auto_commit: bool = True,
) -> StrategyPersistenceResult:
    """Create or update a strategy by DSL payload.

    User ownership is inferred from ``session_id`` and never taken from payload.
    """
    await validate_strategy_payload_or_raise(dsl_payload)

    session_user_id = await get_session_user_id(db, session_id=session_id)
    strategy = None

    if strategy_id is not None:
        strategy = await db.scalar(select(Strategy).where(Strategy.id == strategy_id))
        if strategy is None:
            raise StrategyStorageNotFoundError(f"Strategy not found: {strategy_id}")
        if strategy.user_id != session_user_id:
            raise StrategyStorageNotFoundError(
                "Strategy ownership mismatch for the provided session/user context.",
            )

    columns = _derive_strategy_columns(dsl_payload)

    if strategy is None:
        strategy = Strategy(
            user_id=session_user_id,
            session_id=session_id,
            version=1,
            **columns,
        )
        db.add(strategy)
        await db.flush()
        await _insert_revision_if_absent(
            db,
            strategy=strategy,
            version=1,
            payload=dsl_payload,
            change_type="create",
            session_id=session_id,
        )
        if auto_commit:
            await db.commit()
            await db.refresh(strategy)
        return StrategyPersistenceResult(strategy=strategy, receipt=_build_receipt(strategy))

    await _ensure_current_revision_snapshot(db, strategy=strategy)
    current_version = int(strategy.version)
    return await _persist_next_strategy_version(
        db,
        strategy=strategy,
        session_id=session_id,
        current_version=current_version,
        next_payload=dsl_payload,
        change_type="upsert",
        auto_commit=auto_commit,
    )


def apply_strategy_json_patch(
    *,
    current_payload: dict[str, Any],
    patch_ops: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply RFC 6902 JSON Patch operations and return a new payload."""
    if not isinstance(current_payload, dict):
        raise StrategyPatchApplyError("Current strategy payload must be a JSON object.")

    if not isinstance(patch_ops, list) or not patch_ops:
        raise StrategyPatchApplyError("Patch payload must be a non-empty JSON array.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(patch_ops):
        if not isinstance(item, dict):
            raise StrategyPatchApplyError(
                f"Patch operation at index {index} must be a JSON object.",
            )
        op = item.get("op")
        path = item.get("path")
        if not isinstance(op, str) or not op.strip():
            raise StrategyPatchApplyError(
                f"Patch operation at index {index} has invalid 'op'.",
            )
        if not isinstance(path, str):
            raise StrategyPatchApplyError(
                f"Patch operation at index {index} has invalid 'path'.",
            )
        normalized.append(dict(item))

    try:
        patch = jsonpatch.JsonPatch(normalized)
        patched = patch.apply(deepcopy(current_payload), in_place=False)
    except Exception as exc:  # noqa: BLE001
        raise StrategyPatchApplyError(str(exc)) from exc

    if not isinstance(patched, dict):
        raise StrategyPatchApplyError("Patched strategy payload must remain a JSON object.")

    return patched


async def patch_strategy_dsl(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
    patch_ops: list[dict[str, Any]],
    expected_version: int | None = None,
    auto_commit: bool = True,
) -> StrategyPersistenceResult:
    """Update a stored strategy with RFC 6902 patch operations."""
    if expected_version is not None and expected_version <= 0:
        raise StrategyVersionConflictError("expected_version must be >= 1.")

    strategy = await _get_owned_strategy_or_raise(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
    )

    current_version = int(strategy.version)
    if expected_version is not None and current_version != expected_version:
        raise StrategyVersionConflictError(
            f"Strategy version mismatch: expected {expected_version}, got {current_version}.",
        )

    raw_payload = strategy.dsl_payload
    if not isinstance(raw_payload, dict):
        raise StrategyPatchApplyError("Stored strategy payload must be a JSON object.")

    next_payload = apply_strategy_json_patch(
        current_payload=raw_payload,
        patch_ops=patch_ops,
    )
    await validate_strategy_payload_or_raise(next_payload)

    await _ensure_current_revision_snapshot(db, strategy=strategy)

    return await _persist_next_strategy_version(
        db,
        strategy=strategy,
        session_id=session_id,
        current_version=current_version,
        next_payload=next_payload,
        change_type="patch",
        patch_ops=patch_ops,
        auto_commit=auto_commit,
    )


async def list_strategy_versions(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
    limit: int = 20,
) -> list[StrategyRevisionReceipt]:
    """Return newest-first revision metadata for one strategy."""
    if limit <= 0 or limit > 200:
        raise ValueError("limit must be between 1 and 200.")

    strategy = await _get_owned_strategy_or_raise(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
    )

    rows = await db.scalars(
        select(StrategyRevision)
        .where(StrategyRevision.strategy_id == strategy.id)
        .order_by(StrategyRevision.version.desc())
        .limit(limit),
    )
    versions = [_build_revision_receipt(item) for item in rows]

    current_version = int(strategy.version)
    if not any(item.version == current_version for item in versions):
        versions = [_build_synthetic_revision_receipt(strategy), *versions]

    return versions[:limit]


async def get_strategy_version_payload(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
    version: int,
) -> StrategyVersionPayload:
    """Return DSL payload for one concrete strategy version."""
    if version <= 0:
        raise ValueError("version must be >= 1.")

    strategy = await _get_owned_strategy_or_raise(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
    )

    revision = await db.scalar(
        select(StrategyRevision).where(
            StrategyRevision.strategy_id == strategy.id,
            StrategyRevision.version == version,
        )
    )
    if revision is not None:
        payload = revision.dsl_payload if isinstance(revision.dsl_payload, dict) else {}
        return StrategyVersionPayload(
            strategy_id=strategy.id,
            version=version,
            dsl_payload=payload,
            receipt=_build_revision_receipt(revision),
        )

    current_version = int(strategy.version)
    if version == current_version:
        payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
        return StrategyVersionPayload(
            strategy_id=strategy.id,
            version=version,
            dsl_payload=payload,
            receipt=_build_synthetic_revision_receipt(strategy),
        )

    raise StrategyRevisionNotFoundError(
        f"Strategy revision not found: strategy_id={strategy_id} version={version}",
    )


async def diff_strategy_versions(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
    from_version: int,
    to_version: int,
) -> StrategyVersionDiff:
    """Compute RFC 6902 operations from one version to another."""
    left = await get_strategy_version_payload(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
        version=from_version,
    )
    right = await get_strategy_version_payload(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
        version=to_version,
    )

    patch = jsonpatch.make_patch(left.dsl_payload, right.dsl_payload)
    patch_ops = patch.patch if isinstance(patch.patch, list) else []

    return StrategyVersionDiff(
        strategy_id=strategy_id,
        from_version=from_version,
        to_version=to_version,
        patch_ops=patch_ops,
        op_count=len(patch_ops),
        from_payload_hash=left.receipt.payload_hash,
        to_payload_hash=right.receipt.payload_hash,
    )


async def rollback_strategy_dsl(
    db: AsyncSession,
    *,
    session_id: UUID,
    strategy_id: UUID,
    target_version: int,
    expected_version: int | None = None,
    auto_commit: bool = True,
) -> StrategyPersistenceResult:
    """Rollback to a previous version by creating a new version snapshot."""
    if target_version <= 0:
        raise ValueError("target_version must be >= 1.")
    if expected_version is not None and expected_version <= 0:
        raise StrategyVersionConflictError("expected_version must be >= 1.")

    strategy = await _get_owned_strategy_or_raise(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
    )

    current_version = int(strategy.version)
    if expected_version is not None and current_version != expected_version:
        raise StrategyVersionConflictError(
            f"Strategy version mismatch: expected {expected_version}, got {current_version}.",
        )
    if target_version == current_version:
        raise StrategyPatchApplyError("target_version equals current version; rollback is a no-op.")

    current_payload = strategy.dsl_payload
    if not isinstance(current_payload, dict):
        raise StrategyPatchApplyError("Stored strategy payload must be a JSON object.")

    target = await get_strategy_version_payload(
        db,
        session_id=session_id,
        strategy_id=strategy_id,
        version=target_version,
    )
    next_payload = deepcopy(target.dsl_payload)

    await validate_strategy_payload_or_raise(next_payload)
    await _ensure_current_revision_snapshot(db, strategy=strategy)

    rollback_patch = jsonpatch.make_patch(current_payload, next_payload).patch
    normalized_patch_ops = rollback_patch if isinstance(rollback_patch, list) else None
    return await _persist_next_strategy_version(
        db,
        strategy=strategy,
        session_id=session_id,
        current_version=current_version,
        next_payload=next_payload,
        change_type="rollback",
        source_version=target_version,
        patch_ops=normalized_patch_ops,
        auto_commit=auto_commit,
    )


async def validate_stored_strategy(
    db: AsyncSession,
    *,
    strategy_id: UUID,
) -> StrategyDslValidationResult:
    """Validate a stored strategy DSL payload by ``strategy_id``."""
    strategy = await get_strategy_or_raise(db, strategy_id=strategy_id)
    payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    return validate_strategy_payload(payload)
