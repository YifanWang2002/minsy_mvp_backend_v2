"""Strategy DSL persistence helpers backed by PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.strategy.errors import (
    StrategyDslValidationException,
    StrategyDslValidationResult,
)
from src.engine.strategy.parser import build_parsed_strategy
from src.engine.strategy.pipeline import validate_strategy_payload
from src.models.session import Session
from src.models.strategy import Strategy


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


class StrategyStorageNotFoundError(LookupError):
    """Raised when session/strategy records cannot be resolved."""


def _payload_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _build_receipt(strategy: Strategy) -> StrategyMetadataReceipt:
    updated_at = strategy.updated_at if strategy.updated_at else strategy.created_at
    payload_hash = _payload_hash(strategy.dsl_payload or {})
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
    else:
        strategy.session_id = session_id
        strategy.version = int(strategy.version) + 1
        for field, value in columns.items():
            setattr(strategy, field, value)

    await db.flush()

    if auto_commit:
        await db.commit()
        await db.refresh(strategy)

    return StrategyPersistenceResult(strategy=strategy, receipt=_build_receipt(strategy))


async def validate_stored_strategy(
    db: AsyncSession,
    *,
    strategy_id: UUID,
) -> StrategyDslValidationResult:
    """Validate a stored strategy DSL payload by ``strategy_id``."""
    strategy = await get_strategy_or_raise(db, strategy_id=strategy_id)
    payload = strategy.dsl_payload or {}
    return validate_strategy_payload(payload)
