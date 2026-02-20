"""Telegram webhook handling and bot messaging utilities."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError

from src.config import settings
from src.services.social_connector_service import SocialConnectorService
from src.services.telegram_test_batches import TelegramTestBatchService
from src.util.logger import logger


class TelegramService:
    """Telegram bot integration for link/start, callbacks and text input."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.social_service = SocialConnectorService(db)
        self._bot: Bot | None = None

    async def handle_webhook_update(self, payload: dict[str, Any]) -> None:
        if not settings.telegram_enabled:
            return

        bot = self._bot_client()
        try:
            update = Update.de_json(payload, bot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse Telegram update payload: %s", exc)
            return

        if update is None:
            return
        if update.inline_query is not None:
            await self._handle_inline_query(update)
            return
        if update.callback_query is not None:
            await self._handle_callback_query(update)
            return
        if update.message is not None and update.message.web_app_data is not None:
            await self._handle_web_app_data(update)
            return
        if update.message is not None and isinstance(update.message.text, str):
            await self._handle_text_message(update)

    async def _handle_text_message(self, update: Update) -> None:
        message = update.message
        if message is None:
            return
        text = message.text.strip()
        if not text:
            return

        if text.startswith("/start"):
            token = self._parse_start_token(text)
            await self._handle_start(message, token)
            return

        chat_id = str(message.chat.id)
        binding = await self.social_service.get_telegram_binding_for_chat(
            telegram_chat_id=chat_id,
            require_connected=True,
        )
        if binding is None:
            await self._safe_send_message(
                chat_id=chat_id,
                text="Please connect your account from Minsy Settings first.",
            )
            return

        cleaned = text[:4000]
        await self.social_service.record_telegram_activity(
            user_id=binding.user_id,
            event_type="text",
            choice_value=None,
            message_text=cleaned,
            external_update_id=update.update_id,
            payload={"text": cleaned},
        )

        locale = self.social_service.normalize_locale((binding.metadata_ or {}).get("locale"))
        handled = await self._test_batch_service().handle_connected_text_message(
            update=update,
            binding=binding,
            text=cleaned,
            locale=locale,
        )
        if handled:
            return
        ack_text = "收到你的消息了。" if locale == "zh" else "Message received."
        await self._safe_send_message(chat_id=chat_id, text=ack_text)

    async def _handle_start(self, message: Any, token: str | None) -> None:
        chat_id = str(message.chat.id)
        from_user = message.from_user
        if not token:
            await self._safe_send_message(
                chat_id=chat_id,
                text="绑定链接无效，请返回 Minsy App 重新发起连接。",
            )
            return

        intent = await self.social_service.consume_telegram_link_intent(raw_token=token)
        if intent is None:
            await self._safe_send_message(
                chat_id=chat_id,
                text="The link is invalid or expired. Please reconnect in Minsy Settings.",
            )
            return

        locale = self.social_service.normalize_locale((intent.metadata_ or {}).get("locale"))
        try:
            binding = await self.social_service.upsert_telegram_binding(
                user_id=intent.user_id,
                telegram_chat_id=chat_id,
                telegram_user_id=str(from_user.id) if from_user is not None else chat_id,
                telegram_username=from_user.username if from_user is not None else None,
                locale=locale,
            )
        except ValueError as exc:
            await self._safe_send_message(chat_id=chat_id, text=str(exc))
            return

        await self._send_greeting(chat_id=chat_id, locale=locale)
        await self._test_batch_service().send_post_connect_batches(
            chat_id=chat_id,
            locale=locale,
            binding=binding,
        )

    async def _send_greeting(self, *, chat_id: str, locale: str) -> None:
        is_zh = locale == "zh"
        text = "你喜欢猫还是狗？" if is_zh else "Do you prefer cats or dogs?"
        cat = "猫" if is_zh else "Cat"
        dog = "狗" if is_zh else "Dog"
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(text=cat, callback_data="pref:cat"),
                    InlineKeyboardButton(text=dog, callback_data="pref:dog"),
                ]
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=text, reply_markup=markup)

    async def _handle_callback_query(self, update: Update) -> None:
        callback = update.callback_query
        if callback is None or callback.message is None:
            return

        chat_id = str(callback.message.chat.id)
        binding = await self.social_service.get_telegram_binding_for_chat(
            telegram_chat_id=chat_id,
            require_connected=True,
        )
        if binding is None:
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="Please connect your account in app settings first.",
            )
            return

        data = (callback.data or "").strip().lower()
        locale = self.social_service.normalize_locale((binding.metadata_ or {}).get("locale"))
        handled = await self._test_batch_service().handle_callback_query(
            update=update,
            binding=binding,
            locale=locale,
        )
        if handled:
            return
        if data in {"pref:cat", "pref:dog"}:
            value = data.split(":", 1)[1]
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value=value,
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data},
            )
            ack_text = {
                ("zh", "cat"): "你选择了猫。",
                ("zh", "dog"): "你选择了狗。",
                ("en", "cat"): "You picked Cat.",
                ("en", "dog"): "You picked Dog.",
            }.get((locale, value), "Saved.")
            await self._safe_answer_callback(callback_query_id=callback.id, text=ack_text)
            return

        fallback = "暂不支持该操作。" if locale == "zh" else "This action is not supported yet."
        await self._safe_answer_callback(callback_query_id=callback.id, text=fallback)

    async def _handle_web_app_data(self, update: Update) -> None:
        message = update.message
        if message is None:
            return

        chat_id = str(message.chat.id)
        binding = await self.social_service.get_telegram_binding_for_chat(
            telegram_chat_id=chat_id,
            require_connected=True,
        )
        if binding is None:
            await self._safe_send_message(
                chat_id=chat_id,
                text="Please connect your account from Minsy Settings first.",
            )
            return

        locale = self.social_service.normalize_locale((binding.metadata_ or {}).get("locale"))
        handled = await self._test_batch_service().handle_web_app_data(
            update=update,
            binding=binding,
            locale=locale,
        )
        if handled:
            return

    async def _handle_inline_query(self, update: Update) -> None:
        inline_query = update.inline_query
        if inline_query is None:
            return

        locale = self.social_service.normalize_locale(getattr(inline_query.from_user, "language_code", "en"))
        await self._test_batch_service().handle_inline_query(update=update, locale=locale)

    async def _safe_send_message(
        self,
        *,
        chat_id: str,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        try:
            await self._bot_client().send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
            )
        except TelegramError as exc:
            logger.warning("Telegram send_message failed: %s", exc)

    async def _safe_answer_callback(
        self,
        *,
        callback_query_id: str,
        text: str,
    ) -> None:
        try:
            await self._bot_client().answer_callback_query(
                callback_query_id=callback_query_id,
                text=text,
                show_alert=False,
            )
        except TelegramError as exc:
            logger.warning("Telegram answer_callback_query failed: %s", exc)

    @staticmethod
    def _parse_start_token(text: str) -> str | None:
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return None
        token = parts[1].strip()
        return token or None

    def _bot_client(self) -> Bot:
        if self._bot is None:
            token = settings.telegram_bot_token.strip()
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
            self._bot = Bot(token=token)
        return self._bot

    def _test_batch_service(self) -> TelegramTestBatchService:
        return TelegramTestBatchService(
            db=self.db,
            social_service=self.social_service,
            bot=self._bot_client(),
        )
