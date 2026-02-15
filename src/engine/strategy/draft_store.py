"""Temporary strategy draft storage for pre-confirmation rendering."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

from src.models.redis import get_redis_client, init_redis
from src.util.logger import logger

_DRAFT_KEY_PREFIX = "strategy:draft:"
_DEFAULT_DRAFT_TTL_SECONDS = 60 * 60 * 6  # 6 hours
_IN_MEMORY_DRAFTS: dict[str, tuple[dict[str, Any], datetime]] = {}
_ALLOW_IN_MEMORY_DRAFT_FALLBACK = (
    os.getenv("ALLOW_IN_MEMORY_DRAFT_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
)


@dataclass(frozen=True, slots=True)
class StrategyDraftRecord:
    """One temporary strategy draft persisted in cache."""

    strategy_draft_id: UUID
    user_id: UUID
    session_id: UUID
    dsl_json: dict[str, Any]
    payload_hash: str
    created_at: datetime
    expires_at: datetime
    ttl_seconds: int


def _key(draft_id: UUID) -> str:
    return f"{_DRAFT_KEY_PREFIX}{draft_id}"


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _payload_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def _use_in_memory_fallback() -> bool:
    # Keep fallback for pytest/local isolated tests unless explicitly disabled.
    if _ALLOW_IN_MEMORY_DRAFT_FALLBACK:
        return True
    return "PYTEST_CURRENT_TEST" in os.environ


async def _get_ready_redis_client():
    try:
        return get_redis_client()
    except RuntimeError:
        # MCP worker process does not run FastAPI lifespan; initialize lazily in-place.
        await init_redis()
        return get_redis_client()


def _to_raw(record: StrategyDraftRecord) -> dict[str, Any]:
    return {
        "strategy_draft_id": str(record.strategy_draft_id),
        "user_id": str(record.user_id),
        "session_id": str(record.session_id),
        "dsl_json": deepcopy(record.dsl_json),
        "payload_hash": record.payload_hash,
        "created_at": _normalize_utc(record.created_at).isoformat(),
        "expires_at": _normalize_utc(record.expires_at).isoformat(),
        "ttl_seconds": int(record.ttl_seconds),
    }


def _from_raw(raw: dict[str, Any]) -> StrategyDraftRecord | None:
    try:
        strategy_draft_id = UUID(str(raw["strategy_draft_id"]))
        user_id = UUID(str(raw["user_id"]))
        session_id = UUID(str(raw["session_id"]))
        dsl_json = raw.get("dsl_json")
        if not isinstance(dsl_json, dict):
            return None
        payload_hash = str(raw.get("payload_hash", "")).strip()
        created_at_raw = raw.get("created_at")
        expires_at_raw = raw.get("expires_at")
        created_at = datetime.fromisoformat(str(created_at_raw)) if created_at_raw else datetime.now(UTC)
        expires_at = datetime.fromisoformat(str(expires_at_raw)) if expires_at_raw else datetime.now(UTC)
        ttl_seconds_raw = raw.get("ttl_seconds", _DEFAULT_DRAFT_TTL_SECONDS)
        ttl_seconds = int(ttl_seconds_raw)
    except Exception:  # noqa: BLE001
        return None

    return StrategyDraftRecord(
        strategy_draft_id=strategy_draft_id,
        user_id=user_id,
        session_id=session_id,
        dsl_json=deepcopy(dsl_json),
        payload_hash=payload_hash or _payload_hash(dsl_json),
        created_at=_normalize_utc(created_at),
        expires_at=_normalize_utc(expires_at),
        ttl_seconds=max(ttl_seconds, 1),
    )


async def create_strategy_draft(
    *,
    user_id: UUID,
    session_id: UUID,
    dsl_json: dict[str, Any],
    ttl_seconds: int = _DEFAULT_DRAFT_TTL_SECONDS,
) -> StrategyDraftRecord:
    """Persist a temporary strategy draft and return its draft id record."""

    resolved_ttl = max(int(ttl_seconds), 1)
    now = datetime.now(UTC)
    record = StrategyDraftRecord(
        strategy_draft_id=uuid4(),
        user_id=user_id,
        session_id=session_id,
        dsl_json=deepcopy(dsl_json),
        payload_hash=_payload_hash(dsl_json),
        created_at=now,
        expires_at=now + timedelta(seconds=resolved_ttl),
        ttl_seconds=resolved_ttl,
    )
    raw = _to_raw(record)
    encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))

    try:
        redis = await _get_ready_redis_client()
        await redis.set(_key(record.strategy_draft_id), encoded, ex=resolved_ttl)
    except Exception as exc:  # noqa: BLE001
        if not _use_in_memory_fallback():
            logger.exception("strategy draft persist failed (redis unavailable)")
            raise RuntimeError("Strategy draft storage unavailable.") from exc
        # Fallback for isolated tests where Redis may be unavailable.
        _IN_MEMORY_DRAFTS[str(record.strategy_draft_id)] = (raw, _normalize_utc(record.expires_at))
        logger.warning("strategy draft stored in memory fallback: %s", type(exc).__name__)

    return record


async def get_strategy_draft(strategy_draft_id: UUID) -> StrategyDraftRecord | None:
    """Resolve a temporary strategy draft by id."""

    now = datetime.now(UTC)

    try:
        redis = await _get_ready_redis_client()
        raw_value = await redis.get(_key(strategy_draft_id))
        if raw_value:
            if not isinstance(raw_value, str):
                raw_value = str(raw_value)
            parsed = json.loads(raw_value)
            if isinstance(parsed, dict):
                record = _from_raw(parsed)
                if record is not None and _normalize_utc(record.expires_at) > now:
                    return record
    except Exception as exc:  # noqa: BLE001
        if not _use_in_memory_fallback():
            logger.exception("strategy draft read failed (redis unavailable)")
            raise RuntimeError("Strategy draft storage unavailable.") from exc
        # Silent fallback for isolated tests.
        pass

    fallback = _IN_MEMORY_DRAFTS.get(str(strategy_draft_id))
    if fallback is None:
        return None
    raw, expires_at = fallback
    if _normalize_utc(expires_at) <= now:
        _IN_MEMORY_DRAFTS.pop(str(strategy_draft_id), None)
        return None

    return _from_raw(raw)
