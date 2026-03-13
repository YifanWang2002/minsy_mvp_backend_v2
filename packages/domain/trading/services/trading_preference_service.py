"""Service for user trading execution preferences."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.trading_preference import TradingPreference

_EXECUTION_MODES: frozenset[str] = frozenset({"auto_execute", "approval_required"})
_APPROVAL_CHANNELS: frozenset[str] = frozenset({"telegram", "discord", "slack", "whatsapp"})
_APPROVAL_SCOPES: frozenset[str] = frozenset({"open_only", "open_and_close"})
_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "execution_mode",
        "approval_channel",
        "approval_timeout_seconds",
        "approval_scope",
        "deploy_defaults",
    }
)


@dataclass(frozen=True, slots=True)
class TradingPreferenceView:
    user_id: UUID
    execution_mode: str
    approval_channel: str
    approval_timeout_seconds: int
    approval_scope: str
    deploy_defaults: dict[str, Any]

    @property
    def open_approval_required(self) -> bool:
        return self.execution_mode == "approval_required" and self.approval_scope in {
            "open_only",
            "open_and_close",
        }


class TradingPreferenceService:
    """Read/write helpers for one-user trading preference row."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_create(self, *, user_id: UUID) -> TradingPreference:
        row = await self.db.scalar(
            select(TradingPreference).where(TradingPreference.user_id == user_id)
        )
        if row is not None:
            return row

        row = TradingPreference(user_id=user_id)
        self.db.add(row)
        await self.db.flush()
        return row

    async def get_view(self, *, user_id: UUID) -> TradingPreferenceView:
        row = await self.get_or_create(user_id=user_id)
        try:
            deploy_defaults = _normalize_deploy_defaults(row.deploy_defaults)
        except ValueError:
            deploy_defaults = {}
        return TradingPreferenceView(
            user_id=row.user_id,
            execution_mode=str(row.execution_mode),
            approval_channel=str(row.approval_channel),
            approval_timeout_seconds=int(row.approval_timeout_seconds),
            approval_scope=str(row.approval_scope),
            deploy_defaults=deploy_defaults,
        )

    async def update(self, *, user_id: UUID, updates: dict[str, object]) -> TradingPreferenceView:
        row = await self.get_or_create(user_id=user_id)
        for key, raw_value in updates.items():
            if key not in _UPDATABLE_FIELDS:
                continue
            if key == "execution_mode":
                value = str(raw_value).strip().lower()
                if value not in _EXECUTION_MODES:
                    raise ValueError("execution_mode must be one of auto_execute/approval_required.")
                row.execution_mode = value
                continue
            if key == "approval_channel":
                value = str(raw_value).strip().lower()
                if value not in _APPROVAL_CHANNELS:
                    raise ValueError("approval_channel is not supported.")
                row.approval_channel = value
                continue
            if key == "approval_scope":
                value = str(raw_value).strip().lower()
                if value not in _APPROVAL_SCOPES:
                    raise ValueError("approval_scope must be one of open_only/open_and_close.")
                row.approval_scope = value
                continue
            if key == "approval_timeout_seconds":
                try:
                    timeout = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("approval_timeout_seconds must be a positive integer.") from exc
                if timeout <= 0:
                    raise ValueError("approval_timeout_seconds must be > 0.")
                row.approval_timeout_seconds = timeout
                continue
            if key == "deploy_defaults":
                row.deploy_defaults = _normalize_deploy_defaults(raw_value)
                continue

        await self.db.flush()
        return await self.get_view(user_id=user_id)


def _normalize_deploy_defaults(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("deploy_defaults must be a JSON object.")

    normalized: dict[str, Any] = {}

    capital = _normalize_decimal_text(raw_value.get("capital_allocated"), allow_zero=False)
    if capital is not None:
        normalized["capital_allocated"] = capital

    max_position_size_pct = _normalize_percentage(raw_value.get("max_position_size_pct"))
    if max_position_size_pct is not None:
        normalized["max_position_size_pct"] = max_position_size_pct

    stop_loss_pct = _normalize_percentage(raw_value.get("stop_loss_pct"), allow_zero=True)
    if stop_loss_pct is not None:
        normalized["stop_loss_pct"] = stop_loss_pct

    max_daily_drawdown_pct = _normalize_percentage(
        raw_value.get("max_daily_drawdown_pct"),
        allow_zero=True,
    )
    if max_daily_drawdown_pct is not None:
        normalized["max_daily_drawdown_pct"] = max_daily_drawdown_pct

    auto_start = _normalize_bool(raw_value.get("auto_start"))
    if auto_start is not None:
        normalized["auto_start"] = auto_start

    risk_limits = raw_value.get("risk_limits")
    if isinstance(risk_limits, dict):
        normalized["risk_limits"] = dict(risk_limits)

    return normalized


def _normalize_decimal_text(value: Any, *, allow_zero: bool) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("deploy_defaults.capital_allocated must be a valid decimal.") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"deploy_defaults.capital_allocated must be {comparator}.")
    return format(parsed.normalize(), "f")


def _normalize_percentage(value: Any, *, allow_zero: bool = False) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("deploy_defaults percentage fields must be numeric.") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"deploy_defaults percentage fields must be {comparator}.")
    if parsed > 100:
        raise ValueError("deploy_defaults percentage fields must be <= 100.")
    return round(parsed, 6)


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None
