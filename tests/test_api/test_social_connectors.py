from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.services.telegram_service import TelegramService
from src.services.telegram_test_batches import TelegramTestBatchService


class _DummyMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.answered_callbacks: list[dict] = []
        self.edited_messages: list[dict] = []
        self.chat_actions: list[dict] = []
        self.media_groups: list[dict] = []
        self.polls: list[dict] = []
        self.commands: list[dict] = []
        self.menu_buttons: list[dict] = []
        self.invoices: list[dict] = []
        self.inline_answers: list[dict] = []

    async def send_message(self, **kwargs: Any):
        self.sent_messages.append(dict(kwargs))
        return _DummyMessage(message_id=1000 + len(self.sent_messages))

    async def answer_callback_query(self, **kwargs: Any):
        self.answered_callbacks.append(dict(kwargs))

    async def edit_message_text(self, **kwargs: Any):
        self.edited_messages.append(dict(kwargs))

    async def send_chat_action(self, **kwargs: Any):
        self.chat_actions.append(dict(kwargs))

    async def send_media_group(self, **kwargs: Any):
        self.media_groups.append(dict(kwargs))

    async def send_poll(self, **kwargs: Any):
        self.polls.append(dict(kwargs))

    async def set_my_commands(self, **kwargs: Any):
        self.commands.append(dict(kwargs))

    async def set_chat_menu_button(self, **kwargs: Any):
        self.menu_buttons.append(dict(kwargs))

    async def send_invoice(self, **kwargs: Any):
        self.invoices.append(dict(kwargs))

    async def answer_inline_query(self, **kwargs: Any):
        self.inline_answers.append(dict(kwargs))


def _register_and_get_access_token(client: TestClient) -> str:
    email = f"social_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Connector User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _telegram_headers() -> dict[str, str]:
    secret = settings.telegram_webhook_secret_token.strip()
    if not secret:
        return {}
    return {"X-Telegram-Bot-Api-Secret-Token": secret}


def test_telegram_connector_bind_activity_and_disconnect(monkeypatch) -> None:
    dummy_bot = _DummyBot()
    chat_id = int(str(uuid4().int)[:9])
    base_update_id = int(str(uuid4().int)[:9])

    async def _fake_openai_response(
        self: TelegramTestBatchService,  # noqa: ARG001
        *,
        user_text: str,
        previous_response_id: str | None,  # noqa: ARG001
        locale: str,  # noqa: ARG001
    ) -> tuple[str, str | None]:
        return f"mock reply: {user_text}", "resp_test_1"

    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setattr(settings, "telegram_bot_username", "MinsyUnitTestBot")
    monkeypatch.setattr(settings, "telegram_webhook_secret_token", "unit-test-secret")
    monkeypatch.setattr(settings, "telegram_test_batches_enabled", True)
    monkeypatch.setattr(settings, "telegram_webapp_base_url", "https://app.minsyai.com")
    monkeypatch.setattr(settings, "telegram_test_payment_provider_token", "")
    monkeypatch.setattr(TelegramService, "_bot_client", lambda self: dummy_bot)
    monkeypatch.setattr(
        TelegramTestBatchService,
        "_request_openai_response",
        _fake_openai_response,
    )

    with TestClient(app) as client:
        access_token = _register_and_get_access_token(client)
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        link_response = client.post(
            "/api/v1/social/connectors/telegram/connect-link",
            headers=auth_headers,
            json={"locale": "zh"},
        )
        assert link_response.status_code == 200
        connect_url = link_response.json()["connect_url"]
        parsed = urlparse(connect_url)
        token = parse_qs(parsed.query)["start"][0]

        start_update = {
            "update_id": base_update_id,
            "message": {
                "message_id": 10,
                "date": 1739999999,
                "text": f"/start {token}",
                "chat": {"id": chat_id, "type": "private"},
                "from": {
                    "id": chat_id,
                    "is_bot": False,
                    "first_name": "Tester",
                    "username": "telegram_tester",
                },
            },
        }
        start_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=start_update,
        )
        assert start_response.status_code == 200
        assert start_response.json() == {"ok": True}
        assert len(dummy_bot.sent_messages) >= 5
        assert "已连接 Minsy Telegram 测试通道" in dummy_bot.sent_messages[0]["text"]
        assert any(
            "Pre-Strategy" in msg.get("text", "") and "Deployment" in msg.get("text", "")
            for msg in dummy_bot.sent_messages
        )
        assert any("Strategy 阶段交易机会" in msg.get("text", "") for msg in dummy_bot.sent_messages)
        assert any("Backtest 已完成" in msg.get("text", "") for msg in dummy_bot.sent_messages)
        assert any("实盘开仓提醒" in msg.get("text", "") for msg in dummy_bot.sent_messages)
        assert any("Market Regime 变动提醒" in msg.get("text", "") for msg in dummy_bot.sent_messages)
        assert dummy_bot.commands
        assert dummy_bot.menu_buttons
        assert dummy_bot.polls

        connectors_response = client.get(
            "/api/v1/social/connectors",
            headers=auth_headers,
        )
        assert connectors_response.status_code == 200
        connectors = connectors_response.json()
        telegram = next(item for item in connectors if item["provider"] == "telegram")
        assert telegram["status"] == "connected"
        assert telegram["connected_account"] == "@telegram_tester"

        trade_message = next(
            msg for msg in dummy_bot.sent_messages if "Strategy 阶段交易机会" in msg.get("text", "")
        )
        trade_markup = trade_message["reply_markup"]
        trade_callback_data = trade_markup.inline_keyboard[0][0].callback_data
        assert isinstance(trade_callback_data, str)
        assert trade_callback_data.startswith("trade_open:")

        callback_update = {
            "update_id": base_update_id + 1,
            "callback_query": {
                "id": "callback-1",
                "from": {"id": 910001, "is_bot": False, "first_name": "Tester"},
                "data": trade_callback_data,
                "chat_instance": "chat-instance",
                "message": {
                    "message_id": 11,
                    "date": 1739999999,
                    "chat": {"id": chat_id, "type": "private"},
                },
            },
        }
        callback_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=callback_update,
        )
        assert callback_response.status_code == 200
        assert len(dummy_bot.answered_callbacks) == 1
        assert dummy_bot.edited_messages

        backtest_message = next(
            msg for msg in dummy_bot.sent_messages if "Backtest 已完成" in msg.get("text", "")
        )
        backtest_callback = backtest_message["reply_markup"].inline_keyboard[0][0].callback_data
        backtest_callback_update = {
            "update_id": base_update_id + 2,
            "callback_query": {
                "id": "callback-2",
                "from": {"id": 910001, "is_bot": False, "first_name": "Tester"},
                "data": backtest_callback,
                "chat_instance": "chat-instance",
                "message": {
                    "message_id": 15,
                    "date": 1739999999,
                    "chat": {"id": chat_id, "type": "private"},
                },
            },
        }
        backtest_callback_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=backtest_callback_update,
        )
        assert backtest_callback_response.status_code == 200
        assert len(dummy_bot.answered_callbacks) == 2

        text_update = {
            "update_id": base_update_id + 3,
            "message": {
                "message_id": 12,
                "date": 1739999999,
                "text": "my custom message",
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
            },
        }
        text_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=text_update,
        )
        assert text_response.status_code == 200
        assert dummy_bot.chat_actions
        assert any(
            "mock reply: my custom message" in item.get("text", "")
            for item in dummy_bot.edited_messages
        )

        reset_update = {
            "update_id": base_update_id + 4,
            "message": {
                "message_id": 13,
                "date": 1739999999,
                "text": "/reset",
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
            },
        }
        reset_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=reset_update,
        )
        assert reset_response.status_code == 200
        assert any("已重置对话上下文" in msg.get("text", "") for msg in dummy_bot.sent_messages)

        webapp_update = {
            "update_id": base_update_id + 5,
            "message": {
                "message_id": 14,
                "date": 1739999999,
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
                "web_app_data": {
                    "button_text": "send",
                    "data": "{\"type\":\"trade_confirm\",\"symbol\":\"NASDAQ:AAPL\",\"interval\":\"1d\"}",
                },
            },
        }
        webapp_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=webapp_update,
        )
        assert webapp_response.status_code == 200
        assert any("WebApp 回传参数" in msg.get("text", "") for msg in dummy_bot.sent_messages)

        inline_update = {
            "update_id": base_update_id + 6,
            "inline_query": {
                "id": "inline-1",
                "from": {
                    "id": chat_id,
                    "is_bot": False,
                    "first_name": "Tester",
                    "language_code": "zh",
                },
                "query": "NASDAQ:AAPL",
                "offset": "",
            },
        }
        inline_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=inline_update,
        )
        assert inline_response.status_code == 200
        assert dummy_bot.inline_answers

        activities_response = client.get(
            "/api/v1/social/connectors/telegram/activities",
            headers=auth_headers,
        )
        assert activities_response.status_code == 200
        items = activities_response.json()["items"]
        assert len(items) >= 2
        assert any(item["event_type"] == "text" for item in items)
        assert any(item["message_text"] == "my custom message" for item in items)
        assert any(
            item["event_type"] == "choice" and item.get("choice_value") == "open"
            for item in items
        )

        disconnect_response = client.post(
            "/api/v1/social/connectors/telegram/disconnect",
            headers=auth_headers,
        )
        assert disconnect_response.status_code == 200
        assert disconnect_response.json()["detail"] == "Telegram disconnected."

        connectors_after_disconnect = client.get(
            "/api/v1/social/connectors",
            headers=auth_headers,
        )
        assert connectors_after_disconnect.status_code == 200
        telegram_after = next(
            item
            for item in connectors_after_disconnect.json()
            if item["provider"] == "telegram"
        )
        assert telegram_after["status"] == "disconnected"


def test_telegram_test_chart_webapp_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/social/connectors/telegram/test-webapp/chart",
            params={
                "symbol": "NASDAQ:AAPL",
                "interval": "1d",
                "locale": "zh",
                "theme": "light",
                "signal_id": "sig_demo",
            },
        )
        assert response.status_code == 200
        assert "tradingview-widget-container" in response.text
        assert "embed-widget-advanced-chart.js" in response.text
        assert "sig_demo" in response.text


def test_trade_buttons_follow_user_locale(monkeypatch) -> None:
    dummy_bot = _DummyBot()
    chat_id = int(str(uuid4().int)[:9])
    base_update_id = int(str(uuid4().int)[:9])

    async def _fake_openai_response(
        self: TelegramTestBatchService,  # noqa: ARG001
        *,
        user_text: str,  # noqa: ARG001
        previous_response_id: str | None,  # noqa: ARG001
        locale: str,  # noqa: ARG001
    ) -> tuple[str, str | None]:
        return "ok", "resp_test_2"

    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setattr(settings, "telegram_bot_username", "MinsyUnitTestBot")
    monkeypatch.setattr(settings, "telegram_webhook_secret_token", "unit-test-secret")
    monkeypatch.setattr(settings, "telegram_test_batches_enabled", True)
    monkeypatch.setattr(settings, "telegram_webapp_base_url", "https://app.minsyai.com")
    monkeypatch.setattr(settings, "telegram_test_payment_provider_token", "")
    monkeypatch.setattr(TelegramService, "_bot_client", lambda self: dummy_bot)
    monkeypatch.setattr(
        TelegramTestBatchService,
        "_request_openai_response",
        _fake_openai_response,
    )

    with TestClient(app) as client:
        access_token = _register_and_get_access_token(client)
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        preference_response = client.put(
            "/api/v1/auth/preferences",
            headers=auth_headers,
            json={"theme_mode": "system", "locale": "en", "font_scale": "default"},
        )
        assert preference_response.status_code == 200

        link_response = client.post(
            "/api/v1/social/connectors/telegram/connect-link",
            headers=auth_headers,
            json={"locale": "zh"},
        )
        assert link_response.status_code == 200
        connect_url = link_response.json()["connect_url"]
        parsed = urlparse(connect_url)
        token = parse_qs(parsed.query)["start"][0]

        start_update = {
            "update_id": base_update_id,
            "message": {
                "message_id": 30,
                "date": 1739999999,
                "text": f"/start {token}",
                "chat": {"id": chat_id, "type": "private"},
                "from": {
                    "id": chat_id,
                    "is_bot": False,
                    "first_name": "Tester",
                    "username": "telegram_tester_en",
                },
            },
        }
        start_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=start_update,
        )
        assert start_response.status_code == 200

        trade_message = next(
            msg for msg in dummy_bot.sent_messages if "Strategy Phase Opportunity" in msg.get("text", "")
        )
        trade_markup = trade_message["reply_markup"]
        assert trade_markup.inline_keyboard[0][0].text == "✅ Open"
        assert trade_markup.inline_keyboard[0][1].text == "🙈 Ignore"
