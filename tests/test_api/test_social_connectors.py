from __future__ import annotations

from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.services.telegram_service import TelegramService


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.answered_callbacks: list[dict] = []

    async def send_message(self, **kwargs):
        self.sent_messages.append(dict(kwargs))

    async def answer_callback_query(self, **kwargs):
        self.answered_callbacks.append(dict(kwargs))


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
    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setattr(settings, "telegram_bot_username", "MinsyUnitTestBot")
    monkeypatch.setattr(settings, "telegram_webhook_secret_token", "unit-test-secret")
    monkeypatch.setattr(TelegramService, "_bot_client", lambda self: dummy_bot)

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
        assert len(dummy_bot.sent_messages) == 1
        assert "你喜欢猫还是狗" in dummy_bot.sent_messages[0]["text"]

        connectors_response = client.get(
            "/api/v1/social/connectors",
            headers=auth_headers,
        )
        assert connectors_response.status_code == 200
        connectors = connectors_response.json()
        telegram = next(item for item in connectors if item["provider"] == "telegram")
        assert telegram["status"] == "connected"
        assert telegram["connected_account"] == "@telegram_tester"

        callback_update = {
            "update_id": base_update_id + 1,
            "callback_query": {
                "id": "callback-1",
                "from": {"id": 910001, "is_bot": False, "first_name": "Tester"},
                "data": "pref:cat",
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

        text_update = {
            "update_id": base_update_id + 2,
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

        activities_response = client.get(
            "/api/v1/social/connectors/telegram/activities",
            headers=auth_headers,
        )
        assert activities_response.status_code == 200
        items = activities_response.json()["items"]
        assert len(items) >= 2
        assert items[0]["event_type"] == "text"
        assert items[0]["message_text"] == "my custom message"
        assert items[1]["event_type"] == "choice"
        assert items[1]["choice_value"] == "cat"

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
