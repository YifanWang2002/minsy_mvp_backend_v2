from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select

from src.config import settings
from src.models.deployment import Deployment
from src.models.session import Session
from src.models.social_connector import SocialConnectorBinding
from src.models.strategy import Strategy
from src.models.trade_approval_request import TradeApprovalRequest
from src.models.user import User
from src.services.telegram_approval_codec import TelegramApprovalCodec
from src.services.telegram_service import TelegramService


class _DummyBot:
    def __init__(self) -> None:
        self.answered_callbacks: list[dict[str, Any]] = []
        self.edited_text_messages: list[dict[str, Any]] = []
        self.edited_reply_markups: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> Any:  # noqa: ANN401
        return type("DummyMessage", (), {"message_id": 1})()

    async def answer_callback_query(self, **kwargs: Any) -> None:
        self.answered_callbacks.append(dict(kwargs))

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edited_text_messages.append(dict(kwargs))

    async def edit_message_reply_markup(self, **kwargs: Any) -> None:
        self.edited_reply_markups.append(dict(kwargs))


@pytest.mark.asyncio
async def test_telegram_callback_can_approve_trade_request(db_session, monkeypatch) -> None:
    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setattr(settings, "telegram_approval_callback_secret", "approval-secret")
    monkeypatch.setattr(
        "src.services.telegram_service.enqueue_execute_approved_open",
        lambda *_: "task-telegram-approval",
    )

    user = User(email=f"tg_approval_{uuid4().hex}@test.com", password_hash="pw", name="TG Approver")
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="deployment",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    strategy = Strategy(
        user_id=user.id,
        session_id=session.id,
        name="Callback Strategy",
        description="",
        strategy_type="trend",
        symbols=["AAPL"],
        timeframe="1m",
        parameters={},
        entry_rules={},
        exit_rules={},
        risk_management={},
        dsl_payload={
            "dsl_version": "1.0",
            "strategy": {"name": "Callback Strategy"},
            "universe": {"market": "stocks", "tickers": ["AAPL"]},
            "timeframe": "1m",
            "factors": {},
            "trade": {},
        },
        status="validated",
        version=1,
    )
    db_session.add(strategy)
    await db_session.flush()

    deployment = Deployment(
        strategy_id=strategy.id,
        user_id=user.id,
        mode="paper",
        status="active",
        risk_limits={"order_qty": 1},
        capital_allocated=Decimal("10000"),
    )
    db_session.add(deployment)
    await db_session.flush()

    binding = SocialConnectorBinding(
        user_id=user.id,
        provider="telegram",
        external_user_id="9001",
        external_chat_id="9001",
        external_username="tg_user",
        status="connected",
        bound_at=datetime.now(UTC),
        metadata_={"locale": "en"},
    )
    db_session.add(binding)
    await db_session.flush()

    request = TradeApprovalRequest(
        user_id=user.id,
        deployment_id=deployment.id,
        signal="OPEN_LONG",
        side="long",
        symbol="AAPL",
        qty=Decimal("1"),
        mark_price=Decimal("101.2"),
        reason="entry_signal",
        timeframe="1m",
        bar_time=datetime.now(UTC),
        approval_channel="telegram",
        approval_key=f"open_approval:{deployment.id}:OPEN_LONG:AAPL:1m:1",
        status="pending",
        requested_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=120),
        intent_payload={},
    )
    db_session.add(request)
    await db_session.commit()

    codec = TelegramApprovalCodec(secret="approval-secret")
    callback_data = codec.encode(
        request_id=request.id,
        action="approve",
        expires_at=request.expires_at,
    )

    dummy_bot = _DummyBot()
    monkeypatch.setattr(TelegramService, "_bot_client", lambda self: dummy_bot)

    service = TelegramService(db_session)
    await service.handle_webhook_update(
        {
            "update_id": 10001,
            "callback_query": {
                "id": "approval-cb-1",
                "from": {"id": 9001, "is_bot": False, "first_name": "Tester"},
                "data": callback_data,
                "chat_instance": "chat-inst",
                "message": {
                    "message_id": 88,
                    "date": 1739999999,
                    "chat": {"id": 9001, "type": "private"},
                },
            },
        }
    )
    await db_session.commit()

    reloaded = await db_session.scalar(
        select(TradeApprovalRequest).where(TradeApprovalRequest.id == request.id)
    )
    assert reloaded is not None
    assert reloaded.status == "approved"
    assert reloaded.approved_via == "telegram"
    metadata = reloaded.metadata_ if isinstance(reloaded.metadata_, dict) else {}
    assert metadata.get("execution_task_id") == "task-telegram-approval"
    assert dummy_bot.answered_callbacks
    assert dummy_bot.edited_reply_markups or dummy_bot.edited_text_messages
