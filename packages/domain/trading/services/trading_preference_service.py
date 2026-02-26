"""Service for user trading execution preferences."""

from __future__ import annotations

from dataclasses import dataclass
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
    }
)


@dataclass(frozen=True, slots=True)
class TradingPreferenceView:
    user_id: UUID
    execution_mode: str
    approval_channel: str
    approval_timeout_seconds: int
    approval_scope: str

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
        return TradingPreferenceView(
            user_id=row.user_id,
            execution_mode=str(row.execution_mode),
            approval_channel=str(row.approval_channel),
            approval_timeout_seconds=int(row.approval_timeout_seconds),
            approval_scope=str(row.approval_scope),
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

        await self.db.flush()
        return await self.get_view(user_id=user_id)
