"""Service layer for open-trade approval requests and lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.trade_approval_request import TradeApprovalRequest
from src.services.notification_events import (
    EVENT_TRADE_APPROVAL_APPROVED,
    EVENT_TRADE_APPROVAL_EXPIRED,
    EVENT_TRADE_APPROVAL_REJECTED,
    EVENT_TRADE_APPROVAL_REQUESTED,
)
from src.services.notification_outbox_service import NotificationOutboxService
from src.services.telegram_approval_codec import TelegramApprovalCodec

_FINAL_STATUSES: frozenset[str] = frozenset({"rejected", "expired", "failed", "executed", "cancelled"})
_OPEN_SIGNALS: frozenset[str] = frozenset({"OPEN_LONG", "OPEN_SHORT"})


class TradeApprovalService:
    """Create, decide and expire pre-open approval requests."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(
        self,
        *,
        request_id: UUID,
        for_update: bool = False,
    ) -> TradeApprovalRequest | None:
        stmt = select(TradeApprovalRequest).where(TradeApprovalRequest.id == request_id)
        if for_update:
            stmt = stmt.with_for_update()
        return await self.db.scalar(stmt)

    async def get_owned(
        self,
        *,
        request_id: UUID,
        user_id: UUID,
        for_update: bool = False,
    ) -> TradeApprovalRequest | None:
        stmt = select(TradeApprovalRequest).where(
            TradeApprovalRequest.id == request_id,
            TradeApprovalRequest.user_id == user_id,
        )
        if for_update:
            stmt = stmt.with_for_update()
        return await self.db.scalar(stmt)

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        statuses: set[str] | None = None,
        deployment_id: UUID | None = None,
        limit: int = 100,
    ) -> list[TradeApprovalRequest]:
        stmt = (
            select(TradeApprovalRequest)
            .where(TradeApprovalRequest.user_id == user_id)
            .order_by(TradeApprovalRequest.requested_at.desc())
            .limit(max(1, min(limit, 500)))
        )
        if statuses:
            stmt = stmt.where(TradeApprovalRequest.status.in_({status.lower() for status in statuses}))
        if deployment_id is not None:
            stmt = stmt.where(TradeApprovalRequest.deployment_id == deployment_id)
        rows = (await self.db.scalars(stmt)).all()
        return list(rows)

    async def create_or_get_open_request(
        self,
        *,
        user_id: UUID,
        deployment_id: UUID,
        signal: str,
        side: str,
        symbol: str,
        qty: Decimal,
        mark_price: Decimal,
        reason: str,
        timeframe: str,
        bar_time: datetime | None,
        approval_channel: str,
        approval_timeout_seconds: int,
        intent_payload: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> tuple[TradeApprovalRequest, bool]:
        moment = now or datetime.now(UTC)
        normalized_signal = str(signal).strip().upper()
        if normalized_signal not in _OPEN_SIGNALS:
            raise ValueError("trade approval only supports OPEN_LONG/OPEN_SHORT signals.")
        normalized_symbol = str(symbol).strip().upper()
        normalized_timeframe = str(timeframe).strip().lower() or "1m"
        approval_key = self.build_approval_key(
            deployment_id=deployment_id,
            signal=normalized_signal,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            bar_time=bar_time,
        )

        existing = await self.db.scalar(
            select(TradeApprovalRequest)
            .where(TradeApprovalRequest.approval_key == approval_key)
            .with_for_update()
        )
        if existing is not None:
            await self._expire_if_due(existing, now=moment)
            if existing.status == "pending":
                await self._enqueue_event(
                    request=existing,
                    event_type=EVENT_TRADE_APPROVAL_REQUESTED,
                    event_key=f"trade_approval_requested:{existing.id}",
                    payload=self._build_event_payload(existing, include_actions=True),
                )
            return existing, False

        timeout_seconds = max(1, int(approval_timeout_seconds))
        request = TradeApprovalRequest(
            user_id=user_id,
            deployment_id=deployment_id,
            signal=normalized_signal,
            side=str(side).strip().lower(),
            symbol=normalized_symbol,
            qty=qty,
            mark_price=mark_price,
            reason=str(reason).strip()[:255] or "approval_requested",
            timeframe=normalized_timeframe,
            bar_time=bar_time,
            approval_channel=str(approval_channel).strip().lower() or "telegram",
            approval_key=approval_key,
            intent_payload=dict(intent_payload or {}),
            status="pending",
            requested_at=moment,
            expires_at=moment + timedelta(seconds=timeout_seconds),
        )
        self.db.add(request)
        await self.db.flush()
        await self._enqueue_event(
            request=request,
            event_type=EVENT_TRADE_APPROVAL_REQUESTED,
            event_key=f"trade_approval_requested:{request.id}",
            payload=self._build_event_payload(request, include_actions=True),
        )
        return request, True

    async def approve(
        self,
        *,
        request_id: UUID,
        user_id: UUID,
        via: str,
        actor: str | None,
        note: str | None = None,
    ) -> TradeApprovalRequest | None:
        moment = datetime.now(UTC)
        request = await self.get_owned(request_id=request_id, user_id=user_id, for_update=True)
        if request is None:
            return None

        await self._expire_if_due(request, now=moment)
        if request.status != "pending":
            return request

        request.status = "approved"
        request.approved_at = moment
        request.approved_via = str(via).strip().lower()[:20] or "api"
        request.decision_actor = str(actor).strip()[:128] if actor is not None else None
        if note:
            request.metadata_ = {
                **(request.metadata_ if isinstance(request.metadata_, dict) else {}),
                "decision_note": note[:500],
            }
        await self.db.flush()
        await self._enqueue_event(
            request=request,
            event_type=EVENT_TRADE_APPROVAL_APPROVED,
            event_key=f"trade_approval_approved:{request.id}",
            payload=self._build_event_payload(request, include_actions=False),
        )
        return request

    async def reject(
        self,
        *,
        request_id: UUID,
        user_id: UUID,
        via: str,
        actor: str | None,
        note: str | None = None,
    ) -> TradeApprovalRequest | None:
        moment = datetime.now(UTC)
        request = await self.get_owned(request_id=request_id, user_id=user_id, for_update=True)
        if request is None:
            return None

        await self._expire_if_due(request, now=moment)
        if request.status != "pending":
            return request

        request.status = "rejected"
        request.rejected_at = moment
        request.approved_via = str(via).strip().lower()[:20] or "api"
        request.decision_actor = str(actor).strip()[:128] if actor is not None else None
        if note:
            request.metadata_ = {
                **(request.metadata_ if isinstance(request.metadata_, dict) else {}),
                "decision_note": note[:500],
            }
        await self.db.flush()
        await self._enqueue_event(
            request=request,
            event_type=EVENT_TRADE_APPROVAL_REJECTED,
            event_key=f"trade_approval_rejected:{request.id}",
            payload=self._build_event_payload(request, include_actions=False),
        )
        return request

    async def expire_due(
        self,
        *,
        limit: int = 200,
        now: datetime | None = None,
    ) -> int:
        moment = now or datetime.now(UTC)
        rows = (
            await self.db.scalars(
                select(TradeApprovalRequest)
                .where(
                    TradeApprovalRequest.status == "pending",
                    TradeApprovalRequest.expires_at <= moment,
                )
                .order_by(TradeApprovalRequest.expires_at.asc())
                .limit(max(1, min(limit, 500)))
                .with_for_update(skip_locked=True)
            )
        ).all()
        output = list(rows)
        if not output:
            return 0
        for request in output:
            await self._expire_if_due(request, now=moment)
        return len(output)

    async def mark_executing(self, *, request_id: UUID) -> TradeApprovalRequest | None:
        request = await self.get_by_id(request_id=request_id, for_update=True)
        if request is None:
            return None
        if request.status == "approved":
            request.status = "executing"
            await self.db.flush()
        return request

    async def mark_executed(
        self,
        *,
        request_id: UUID,
        order_id: UUID | None,
    ) -> TradeApprovalRequest | None:
        request = await self.get_by_id(request_id=request_id, for_update=True)
        if request is None:
            return None
        request.status = "executed"
        request.executed_at = datetime.now(UTC)
        request.execution_order_id = order_id
        request.execution_error = None
        await self.db.flush()
        return request

    async def mark_failed(
        self,
        *,
        request_id: UUID,
        error: str,
    ) -> TradeApprovalRequest | None:
        request = await self.get_by_id(request_id=request_id, for_update=True)
        if request is None:
            return None
        if request.status not in _FINAL_STATUSES:
            request.status = "failed"
        request.executed_at = datetime.now(UTC)
        request.execution_error = str(error).strip()[:500] or "unknown_error"
        await self.db.flush()
        return request

    async def append_execution_task_id(
        self,
        *,
        request_id: UUID,
        task_id: str,
    ) -> TradeApprovalRequest | None:
        request = await self.get_by_id(request_id=request_id, for_update=True)
        if request is None:
            return None
        request.metadata_ = {
            **(request.metadata_ if isinstance(request.metadata_, dict) else {}),
            "execution_task_id": task_id,
            "execution_task_enqueued_at": datetime.now(UTC).isoformat(),
        }
        await self.db.flush()
        return request

    async def _expire_if_due(
        self,
        request: TradeApprovalRequest,
        *,
        now: datetime,
    ) -> None:
        if request.status != "pending":
            return
        if now <= request.expires_at:
            return
        request.status = "expired"
        request.expired_at = now
        await self.db.flush()
        await self._enqueue_event(
            request=request,
            event_type=EVENT_TRADE_APPROVAL_EXPIRED,
            event_key=f"trade_approval_expired:{request.id}",
            payload=self._build_event_payload(request, include_actions=False),
        )

    async def _enqueue_event(
        self,
        *,
        request: TradeApprovalRequest,
        event_type: str,
        event_key: str,
        payload: dict[str, Any],
    ) -> None:
        await NotificationOutboxService(self.db).enqueue_event(
            user_id=request.user_id,
            channel=request.approval_channel,
            event_type=event_type,
            event_key=event_key,
            payload=payload,
        )

    def _build_event_payload(
        self,
        request: TradeApprovalRequest,
        *,
        include_actions: bool,
    ) -> dict[str, Any]:
        payload = {
            "approval_request_id": str(request.id),
            "deployment_id": str(request.deployment_id),
            "signal": request.signal,
            "side": request.side,
            "symbol": request.symbol,
            "qty": float(request.qty),
            "mark_price": float(request.mark_price),
            "reason": request.reason,
            "timeframe": request.timeframe,
            "bar_time": request.bar_time.isoformat() if request.bar_time is not None else None,
            "status": request.status,
            "requested_at": request.requested_at.isoformat(),
            "expires_at": request.expires_at.isoformat(),
            "approved_at": request.approved_at.isoformat() if request.approved_at is not None else None,
            "rejected_at": request.rejected_at.isoformat() if request.rejected_at is not None else None,
            "expired_at": request.expired_at.isoformat() if request.expired_at is not None else None,
            "approved_via": request.approved_via,
            "decision_actor": request.decision_actor,
        }
        if include_actions and request.status == "pending" and request.approval_channel == "telegram":
            codec = TelegramApprovalCodec()
            payload["actions"] = [
                {
                    "key": "approve",
                    "callback_data": codec.encode(
                        request_id=request.id,
                        action="approve",
                        expires_at=request.expires_at,
                    ),
                    "label": {"en": "Approve", "zh": "批准"},
                },
                {
                    "key": "reject",
                    "callback_data": codec.encode(
                        request_id=request.id,
                        action="reject",
                        expires_at=request.expires_at,
                    ),
                    "label": {"en": "Reject", "zh": "拒绝"},
                },
            ]
        return payload

    @staticmethod
    def build_approval_key(
        *,
        deployment_id: UUID,
        signal: str,
        symbol: str,
        timeframe: str,
        bar_time: datetime | None,
    ) -> str:
        if bar_time is None:
            bar_epoch = 0
        else:
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=UTC)
            bar_epoch = int(bar_time.timestamp())
        return (
            f"open_approval:{deployment_id}:"
            f"{str(signal).strip().upper()}:{str(symbol).strip().upper()}:"
            f"{str(timeframe).strip().lower()}:{bar_epoch}"
        )
