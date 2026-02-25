from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.services.telegram_service import TelegramService


class _DummyMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any):
        self.sent_messages.append(dict(kwargs))
        return _DummyMessage(message_id=1000 + len(self.sent_messages))


def _register(client: TestClient, *, email: str) -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "TG User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _telegram_headers() -> dict[str, str]:
    secret = settings.telegram_webhook_secret_token.strip()
    if not secret:
        return {}
    return {"X-Telegram-Bot-Api-Secret-Token": secret}


def test_telegram_test_target_status_and_send(monkeypatch) -> None:
    dummy_bot = _DummyBot()
    target_email = f"target_{uuid4().hex}@test.com"
    requester_email = f"requester_{uuid4().hex}@test.com"
    target_chat_id = int(str(uuid4().int)[:9])
    base_update_id = int(str(uuid4().int)[:9])

    monkeypatch.setattr(settings, "telegram_enabled", True)
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setattr(settings, "telegram_bot_username", "MinsyUnitTestBot")
    monkeypatch.setattr(settings, "telegram_webhook_secret_token", "unit-test-secret")
    monkeypatch.setattr(settings, "telegram_test_batches_enabled", False)
    monkeypatch.setattr(settings, "telegram_test_target_email", target_email)
    monkeypatch.setattr(settings, "telegram_test_force_target_email_enabled", True)
    monkeypatch.setattr(settings, "telegram_test_target_require_connected", True)
    monkeypatch.setattr(settings, "telegram_test_expected_chat_id", str(target_chat_id))
    monkeypatch.setattr(TelegramService, "_bot_client", lambda self: dummy_bot)

    with TestClient(app) as client:
        target_token = _register(client, email=target_email)
        target_headers = {"Authorization": f"Bearer {target_token}"}
        requester_token = _register(client, email=requester_email)
        requester_headers = {"Authorization": f"Bearer {requester_token}"}

        link_response = client.post(
            "/api/v1/social/connectors/telegram/connect-link",
            headers=target_headers,
            json={"locale": "en"},
        )
        assert link_response.status_code == 200
        connect_url = link_response.json()["connect_url"]
        token = parse_qs(urlparse(connect_url).query)["start"][0]

        start_update = {
            "update_id": base_update_id,
            "message": {
                "message_id": 10,
                "date": 1739999999,
                "text": f"/start {token}",
                "chat": {"id": target_chat_id, "type": "private"},
                "from": {
                    "id": target_chat_id,
                    "is_bot": False,
                    "first_name": "Tester",
                    "username": "telegram_target",
                },
            },
        }
        start_response = client.post(
            "/api/v1/social/webhooks/telegram",
            headers=_telegram_headers(),
            json=start_update,
        )
        assert start_response.status_code == 200

        target_status = client.get(
            "/api/v1/social/connectors/telegram/test-target",
            headers=requester_headers,
        )
        assert target_status.status_code == 200
        body = target_status.json()
        assert body["configured_email"] == target_email
        assert body["resolved_user_exists"] is True
        assert body["resolved_binding_connected"] is True
        assert body["resolved_user_id"] is not None
        assert body["resolved_chat_id_masked"] is not None

        send_response = client.post(
            "/api/v1/social/connectors/telegram/test-send",
            headers=requester_headers,
            json={"message": "integration ping"},
        )
        assert send_response.status_code == 200
        send_body = send_response.json()
        assert send_body["ok"] is True
        assert send_body["message_id"] is not None
        assert target_email in send_body["actual_target"]
        assert any(
            msg.get("chat_id") == str(target_chat_id) and msg.get("text") == "integration ping"
            for msg in dummy_bot.sent_messages
        )
