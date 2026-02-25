"""Session title derivation and metadata synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.session import Session
from src.models.strategy import Strategy

SESSION_TITLE_META_KEY = "session_title"
SESSION_TITLE_RECORD_META_KEY = "session_title_record"
SESSION_TITLE_RECORD_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionTitleResult:
    """Derived session title payload."""

    title: str | None
    record: dict[str, Any] | None


def _as_map(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_profile(artifacts: dict[str, Any], phase_key: str) -> dict[str, Any]:
    phase_block = _as_map(artifacts.get(phase_key))
    return _as_map(phase_block.get("profile"))


def _coerce_non_empty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_uuid_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return str(UUID(cleaned))
    except ValueError:
        return None


def _pick_text(*values: Any) -> str | None:
    for value in values:
        picked = _coerce_non_empty_text(value)
        if picked is not None:
            return picked
    return None


def _market_label_en(value: str | None) -> str | None:
    if value is None:
        return None
    table = {
        "us_stocks": "US Stocks",
        "crypto": "Crypto",
        "forex": "Forex",
        "futures": "Futures",
    }
    return table.get(value.strip().lower(), value)


def _holding_label_en(value: str | None) -> str | None:
    if value is None:
        return None
    table = {
        "intraday_scalp": "Scalp",
        "intraday": "Intraday",
        "swing_days": "Swing",
        "position_weeks_plus": "Position",
    }
    return table.get(value.strip().lower(), value)


def _phase_default_title_en(phase: str) -> str:
    table = {
        "kyc": "KYC-In Progress",
        "pre_strategy": "Pre-Strategy In Progress",
        "strategy": "Strategy In Progress",
        "stress_test": "Strategy In Progress",
        "deployment": "Deployment In Progress",
        "completed": "Completed",
        "error": "Error",
    }
    return table.get(phase, "Session")


def _join_title_parts(parts: list[str]) -> str:
    return " ".join(part for part in parts if part).strip()


def _build_pre_strategy_title(
    *,
    market: str | None,
    instrument: str | None,
    holding_period_bucket: str | None,
) -> str:
    market_label = _market_label_en(market)
    holding_label = _holding_label_en(holding_period_bucket)

    if market_label and instrument and holding_label:
        return _join_title_parts(
            [market_label, instrument, holding_label, "Strategy", "Development"]
        )
    if market_label and instrument:
        return _join_title_parts([market_label, instrument, "Strategy", "Development"])
    if market_label and holding_label:
        return _join_title_parts(
            [market_label, holding_label, "Strategy", "Development"]
        )
    if market_label:
        return _join_title_parts([market_label, "Strategy", "Development"])
    if instrument and holding_label:
        return _join_title_parts([instrument, holding_label, "Strategy", "Development"])
    if instrument:
        return _join_title_parts([instrument, "Strategy", "Development"])
    return "Pre-Strategy In Progress"


def _build_strategy_title(
    *,
    market: str | None,
    strategy_name: str | None,
    instrument: str | None,
) -> str:
    market_label = _market_label_en(market)

    if market_label and strategy_name:
        return f"{market_label} · {strategy_name}"
    if strategy_name:
        return strategy_name
    if market_label:
        return _join_title_parts([market_label, "Strategy"])
    if instrument:
        return _join_title_parts([instrument, "Strategy"])
    return "Strategy In Progress"


async def _resolve_strategy_name(
    *,
    db: AsyncSession,
    session: Session,
    strategy_id: str | None,
    existing_name: str | None,
) -> str | None:
    if existing_name is not None:
        return existing_name
    if strategy_id is None:
        return None

    strategy_uuid = UUID(strategy_id)
    strategy_name = await db.scalar(
        select(Strategy.name).where(
            Strategy.id == strategy_uuid,
            Strategy.user_id == session.user_id,
        )
    )
    if not isinstance(strategy_name, str):
        return None
    cleaned = strategy_name.strip()
    return cleaned or None


def read_session_title_from_metadata(
    metadata: dict[str, Any] | None,
) -> SessionTitleResult:
    payload = _as_map(metadata)
    title = _coerce_non_empty_text(payload.get(SESSION_TITLE_META_KEY))
    record = _as_map(payload.get(SESSION_TITLE_RECORD_META_KEY))
    return SessionTitleResult(
        title=title,
        record=record if record else None,
    )


async def refresh_session_title(
    *,
    db: AsyncSession,
    session: Session,
) -> SessionTitleResult:
    """Derive session title from current phase/artifacts and sync metadata."""

    artifacts = _as_map(session.artifacts)
    metadata = _as_map(session.metadata_)
    phase = (_coerce_non_empty_text(session.current_phase) or "kyc").lower()
    pre_profile = _as_profile(artifacts, "pre_strategy")
    strategy_profile = _as_profile(artifacts, "strategy")
    stress_profile = _as_profile(artifacts, "stress_test")

    target_market = _pick_text(pre_profile.get("target_market"))
    target_instrument = _pick_text(pre_profile.get("target_instrument"))
    holding_period_bucket = _pick_text(pre_profile.get("holding_period_bucket"))
    opportunity_frequency_bucket = _pick_text(
        pre_profile.get("opportunity_frequency_bucket")
    )

    strategy_market = _pick_text(
        strategy_profile.get("strategy_market"),
        metadata.get("strategy_market"),
        target_market,
    )
    strategy_instrument = _pick_text(
        strategy_profile.get("strategy_primary_symbol"),
        metadata.get("strategy_primary_symbol"),
        target_instrument,
    )
    strategy_id = _coerce_uuid_text(
        _pick_text(
            strategy_profile.get("strategy_id"),
            metadata.get("strategy_id"),
            stress_profile.get("strategy_id"),
        )
    )
    strategy_name_seed = _pick_text(
        strategy_profile.get("strategy_name"),
        metadata.get("strategy_name"),
    )
    strategy_name = await _resolve_strategy_name(
        db=db,
        session=session,
        strategy_id=strategy_id,
        existing_name=strategy_name_seed,
    )

    if strategy_name is not None:
        metadata["strategy_name"] = strategy_name

    if phase == "kyc":
        kind = "kyc_in_progress"
        title = "KYC-In Progress"
    elif phase == "pre_strategy":
        kind = "pre_strategy_scope"
        title = _build_pre_strategy_title(
            market=target_market,
            instrument=target_instrument,
            holding_period_bucket=holding_period_bucket,
        )
    elif phase in {"strategy", "stress_test"}:
        kind = "strategy_named" if strategy_name else "strategy_scope"
        title = _build_strategy_title(
            market=strategy_market,
            strategy_name=strategy_name,
            instrument=strategy_instrument,
        )
    elif phase == "deployment":
        kind = "deployment_in_progress"
        title = _phase_default_title_en(phase)
    else:
        kind = "phase_default"
        title = _phase_default_title_en(phase)

    now_iso = datetime.now(UTC).isoformat()
    record: dict[str, Any] = {
        "version": SESSION_TITLE_RECORD_VERSION,
        "kind": kind,
        "phase": phase,
        "market": (
            strategy_market
            if phase in {"strategy", "stress_test", "deployment"}
            else target_market
        ),
        "instrument": (
            strategy_instrument
            if phase in {"strategy", "stress_test", "deployment"}
            else target_instrument
        ),
        "holding_period_bucket": holding_period_bucket,
        "opportunity_frequency_bucket": opportunity_frequency_bucket,
        "strategy_name": strategy_name,
        "strategy_id": strategy_id,
        "title": title,
        "updated_at": now_iso,
    }

    metadata[SESSION_TITLE_META_KEY] = title
    metadata[SESSION_TITLE_RECORD_META_KEY] = record
    session.metadata_ = metadata
    return SessionTitleResult(title=title, record=record)
