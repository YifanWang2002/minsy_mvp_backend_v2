"""Order intent persistence + idempotent submission manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.trading.runtime.order_state_machine import apply_order_status_transition
from packages.infra.providers.trading.adapters.base import BrokerAdapter, OrderIntent
from packages.infra.db.models.order import Order
from packages.infra.db.models.order_state_transition import OrderStateTransition


@dataclass(frozen=True, slots=True)
class OrderSubmitResult:
    order: Order
    idempotent_hit: bool


def _normalize_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {
        "accepted",
        "new",
        "pending_new",
        "partially_filled",
        "filled",
        "canceled",
        "rejected",
        "expired",
    }:
        return normalized
    if normalized in {"cancelled"}:
        return "canceled"
    if normalized in {"pending"}:
        return "pending_new"
    if normalized in {"done_for_day"}:
        return "expired"
    return "new"


def _provider_status_metadata(metadata: dict[str, Any], provider_status: str) -> dict[str, Any]:
    payload = dict(metadata) if isinstance(metadata, dict) else {}
    payload["provider_status"] = provider_status
    payload["provider_status_updated_at"] = datetime.now(UTC).isoformat()
    return payload


def _extract_reject_reason(raw: dict[str, Any] | None) -> str | None:
    if not isinstance(raw, dict):
        return None
    for key in ("reject_reason", "rejected_reason", "message"):
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text[:255]
    return None


class OrderManager:
    """Handles order idempotency and optional broker submission."""

    async def submit_order_intent(
        self,
        *,
        db: AsyncSession,
        deployment_id: str,
        intent: OrderIntent,
        adapter: BrokerAdapter | None = None,
    ) -> OrderSubmitResult:
        existing = await db.scalar(select(Order).where(Order.client_order_id == intent.client_order_id))
        if existing is not None:
            return OrderSubmitResult(order=existing, idempotent_hit=True)

        if adapter is None:
            now = datetime.now(UTC)
            metadata = _provider_status_metadata(intent.metadata, "accepted")
            order = Order(
                deployment_id=deployment_id,
                provider_order_id=f"paper-{intent.client_order_id}",
                client_order_id=intent.client_order_id,
                symbol=intent.symbol,
                side=intent.side,
                type=intent.order_type,
                qty=intent.qty,
                price=intent.limit_price,
                status="new",
                submitted_at=now,
                provider_updated_at=now,
                last_sync_at=now,
                metadata_=metadata,
            )
            transition = apply_order_status_transition(
                order,
                target_status="accepted",
                reason="local_paper_accept",
            )
            db.add(order)
            await db.flush()
            db.add(
                OrderStateTransition(
                    order_id=order.id,
                    from_status=str(transition["from"]),
                    to_status=str(transition["to"]),
                    reason=str(transition["reason"]),
                    transitioned_at=datetime.fromisoformat(str(transition["ts"])),
                    metadata_={},
                )
            )
            await db.commit()
            await db.refresh(order)
            return OrderSubmitResult(order=order, idempotent_hit=False)

        state = await adapter.submit_order(intent)
        target_status = _normalize_status(state.status)
        now = datetime.now(UTC)
        provider_status = str(state.status).strip().lower() or target_status
        metadata = _provider_status_metadata(intent.metadata, provider_status)
        order = Order(
            deployment_id=deployment_id,
            provider_order_id=state.provider_order_id,
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            type=intent.order_type,
            qty=intent.qty,
            price=intent.limit_price,
            status="new",
            submitted_at=state.submitted_at or datetime.now(UTC),
            reject_reason=state.reject_reason or _extract_reject_reason(state.raw),
            provider_updated_at=state.provider_updated_at or state.submitted_at or now,
            last_sync_at=now,
            metadata_=metadata,
        )
        transition = apply_order_status_transition(
            order,
            target_status=target_status,
            reason="provider_submit_response",
            extra_metadata={"provider_status": provider_status},
        )
        db.add(order)
        await db.flush()
        db.add(
            OrderStateTransition(
                order_id=order.id,
                from_status=str(transition["from"]),
                to_status=str(transition["to"]),
                reason=str(transition["reason"]),
                transitioned_at=datetime.fromisoformat(str(transition["ts"])),
                metadata_={"provider_status": provider_status},
            )
        )
        if state.avg_fill_price is not None:
            order.price = Decimal(str(state.avg_fill_price))
        if state.submitted_at is not None:
            order.submitted_at = state.submitted_at
        await db.commit()
        await db.refresh(order)
        return OrderSubmitResult(order=order, idempotent_hit=False)
