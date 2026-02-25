"""Telegram provider implementation for IM notification dispatch."""

from __future__ import annotations

import asyncio
from typing import Any

import requests
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from src.config import settings
from src.models.social_connector import SocialConnectorBinding
from src.services.im_channels import IMChannelProvider, IMDeliveryResult
from src.services.telegram_notification_templates import render_telegram_notification


class TelegramNotificationProvider(IMChannelProvider):
    """Notification provider that delivers events through Telegram bot API."""

    channel = "telegram"

    def __init__(self, *, bot: Bot | None = None) -> None:
        self._bot = bot

    async def send_event(
        self,
        *,
        binding: SocialConnectorBinding,
        event_type: str,
        payload: dict[str, Any],
        locale: str,
    ) -> IMDeliveryResult:
        text = render_telegram_notification(event_type=event_type, payload=payload, locale=locale)
        reply_markup = self._build_reply_markup(payload=payload, locale=locale)
        request_payload = {
            "chat_id": binding.external_chat_id,
            "event_type": event_type,
            "text_preview": text[:120],
            "has_reply_markup": reply_markup is not None,
        }
        try:
            response_payload = await self._send_with_http_api(
                chat_id=str(binding.external_chat_id),
                text=text,
                reply_markup=reply_markup,
            )
        except (TelegramError, requests.RequestException, ValueError) as exc:
            return IMDeliveryResult(
                success=False,
                provider=self.channel,
                request_payload=request_payload,
                response_payload={},
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
            )
        except Exception as exc:  # noqa: BLE001
            return IMDeliveryResult(
                success=False,
                provider=self.channel,
                request_payload=request_payload,
                response_payload={},
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
            )

        message_id = response_payload.get("message_id")
        return IMDeliveryResult(
            success=True,
            provider=self.channel,
            provider_message_id=str(message_id) if message_id is not None else None,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    async def _send_with_http_api(
        self,
        *,
        chat_id: str,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> dict[str, Any]:
        token = settings.telegram_bot_token.strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup.to_dict()

        def _post() -> requests.Response:
            return requests.post(
                url,
                json=payload,
                timeout=max(1.0, float(settings.notifications_delivery_timeout_seconds)),
            )

        resp = await asyncio.to_thread(_post)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict) or body.get("ok") is not True:
            raise ValueError(f"telegram_send_failed:{body}")
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        message_id = result.get("message_id")
        return {"message_id": message_id}

    def _bot_client(self) -> Bot:
        if self._bot is None:
            token = settings.telegram_bot_token.strip()
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
            self._bot = Bot(token=token)
        return self._bot

    def _build_reply_markup(
        self,
        *,
        payload: dict[str, Any],
        locale: str,
    ) -> InlineKeyboardMarkup | None:
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list):
            return None
        lang = "zh" if str(locale).strip().lower().startswith("zh") else "en"
        buttons: list[InlineKeyboardButton] = []
        for action in raw_actions[:3]:
            if not isinstance(action, dict):
                continue
            callback_data = str(action.get("callback_data") or "").strip()
            if not callback_data:
                continue
            if len(callback_data) > 64:
                continue
            label = self._resolve_action_label(action=action, locale=lang)
            if not label:
                continue
            buttons.append(InlineKeyboardButton(text=label, callback_data=callback_data))
        if not buttons:
            return None
        return InlineKeyboardMarkup([buttons])

    @staticmethod
    def _resolve_action_label(*, action: dict[str, Any], locale: str) -> str:
        raw = action.get("label")
        if isinstance(raw, str):
            text = raw.strip()
            return text
        if isinstance(raw, dict):
            localized = raw.get(locale)
            if isinstance(localized, str) and localized.strip():
                return localized.strip()
            fallback = raw.get("en")
            if isinstance(fallback, str):
                return fallback.strip()
        key = str(action.get("key") or "").strip().lower()
        if key == "approve":
            return "批准" if locale == "zh" else "Approve"
        if key == "reject":
            return "拒绝" if locale == "zh" else "Reject"
        return ""
