"""Telegram webhook handling and bot messaging utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Bot, InlineKeyboardMarkup, Update
from telegram.error import TelegramError

from packages.shared_settings.schema.settings import settings
from packages.domain.user.services.social_connector_service import SocialConnectorService
from packages.domain.trading.services.telegram_approval_codec import TelegramApprovalCodec
from packages.domain.user.services.telegram_test_batches import TelegramTestBatchService
from packages.domain.trading.services.trade_approval_service import TradeApprovalService
from packages.domain.trading.services.trading_queue_service import enqueue_execute_approved_open
from packages.infra.observability.logger import logger


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

        locale = await self._resolve_binding_locale(binding)
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

        locale = await self._resolve_binding_locale(binding, fallback_locale=locale)
        await self._send_greeting(chat_id=chat_id, locale=locale)
        await self._test_batch_service().send_post_connect_batches(
            chat_id=chat_id,
            locale=locale,
            binding=binding,
        )

    async def _send_greeting(self, *, chat_id: str, locale: str) -> None:
        text = (
            "已连接 Minsy Telegram 测试通道，正在按批次发送流程消息。"
            if locale == "zh"
            else "Connected to the Minsy Telegram test channel. Sending batched flow messages."
        )
        await self._safe_send_message(chat_id=chat_id, text=text)

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

        data = (callback.data or "").strip()
        locale = await self._resolve_binding_locale(binding)
        approval_handled = await self._handle_trade_approval_callback(
            update=update,
            binding=binding,
            locale=locale,
        )
        if approval_handled:
            return
        handled = await self._test_batch_service().handle_callback_query(
            update=update,
            binding=binding,
            locale=locale,
        )
        if handled:
            return
        lowered = data.lower()
        if lowered in {"pref:cat", "pref:dog"}:
            value = lowered.split(":", 1)[1]
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value=value,
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": lowered},
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

    async def _handle_trade_approval_callback(
        self,
        *,
        update: Update,
        binding: Any,
        locale: str,
    ) -> bool:
        callback = update.callback_query
        if callback is None or callback.message is None:
            return False

        raw_data = (callback.data or "").strip()
        if not TelegramApprovalCodec.looks_like(raw_data):
            return False

        parsed, error_code = TelegramApprovalCodec().decode(raw_data)
        if parsed is None:
            text = (
                "审批按钮校验失败，请刷新后重试。"
                if locale == "zh"
                else "Approval button verification failed. Please refresh and retry."
            )
            await self._safe_answer_callback(callback_query_id=callback.id, text=text)
            logger.warning("Telegram approval callback decode failed: %s", error_code)
            return True

        from_user_id = str(callback.from_user.id) if callback.from_user is not None else None
        bound_external_user_id = str(binding.external_user_id).strip() if binding.external_user_id else None
        if from_user_id and bound_external_user_id and from_user_id != bound_external_user_id:
            text = (
                "该审批按钮不属于当前 Telegram 账号。"
                if locale == "zh"
                else "This approval button belongs to a different Telegram account."
            )
            await self._safe_answer_callback(callback_query_id=callback.id, text=text)
            return True

        service = TradeApprovalService(self.db)
        decision_note = None
        if parsed.expired:
            decision_note = "callback_expired"

        if parsed.action == "approve":
            request = await service.approve(
                request_id=parsed.request_id,
                user_id=binding.user_id,
                via="telegram",
                actor=from_user_id,
                note=decision_note,
            )
            request_metadata = request.metadata_ if request is not None and isinstance(request.metadata_, dict) else {}
            if (
                request is not None
                and request.status == "approved"
                and not request_metadata.get("execution_task_id")
            ):
                task_id = enqueue_execute_approved_open(request.id)
                if task_id:
                    await service.append_execution_task_id(request_id=request.id, task_id=task_id)
            choice_value = "approval_approve"
        else:
            request = await service.reject(
                request_id=parsed.request_id,
                user_id=binding.user_id,
                via="telegram",
                actor=from_user_id,
                note=decision_note,
            )
            choice_value = "approval_reject"

        if request is None:
            text = "审批请求不存在或已失效。" if locale == "zh" else "Approval request not found."
            await self._safe_answer_callback(callback_query_id=callback.id, text=text)
            return True

        await self.social_service.record_telegram_activity(
            user_id=binding.user_id,
            event_type="choice",
            choice_value=choice_value,
            message_text=None,
            external_update_id=update.update_id,
            payload={
                "callback_data": raw_data,
                "kind": "trade_approval",
                "request_id": str(request.id),
                "action": parsed.action,
                "status": request.status,
            },
        )

        if locale == "zh":
            status_to_ack = {
                "approved": "审批通过，已开始执行开仓。",
                "rejected": "已拒绝本次开仓。",
                "expired": "审批已过期，未执行开仓。",
                "executing": "审批已处理，正在执行。",
                "executed": "审批已处理，开仓已执行。",
            }
        else:
            status_to_ack = {
                "approved": "Approved. Execution has started.",
                "rejected": "Rejected. No open order was placed.",
                "expired": "Approval expired. Open order skipped.",
                "executing": "Approval processed. Execution in progress.",
                "executed": "Approval processed. Open order executed.",
            }
        ack_text = status_to_ack.get(
            request.status,
            "审批结果已记录。" if locale == "zh" else "Approval decision recorded.",
        )
        await self._safe_finalize_approval_message(
            callback=callback,
            status=request.status,
            locale=locale,
            decided_at=datetime.now(UTC),
        )
        await self._safe_answer_callback(callback_query_id=callback.id, text=ack_text)
        return True

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

        locale = await self._resolve_binding_locale(binding)
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

        from_user = inline_query.from_user
        locale = self.social_service.normalize_locale(getattr(from_user, "language_code", "en"))
        if from_user is not None:
            binding = await self.social_service.get_telegram_binding_for_external_user(
                telegram_user_id=str(from_user.id),
                require_connected=True,
            )
            if binding is not None:
                locale = await self._resolve_binding_locale(binding, fallback_locale=locale)
        await self._test_batch_service().handle_inline_query(update=update, locale=locale)

    async def resolve_test_target_binding(self) -> Any:
        """Resolve configured Telegram test target binding for diagnostics."""
        target_email = settings.telegram_test_target_email.strip().lower()
        if not target_email:
            return None
        return await self.social_service.resolve_connected_telegram_binding_by_email(
            email=target_email,
            require_connected=settings.telegram_test_target_require_connected,
        )

    async def send_test_message(self, *, text: str) -> dict[str, Any]:
        """Send one debug test message honoring forced test-target routing policy."""
        binding = await self.resolve_test_target_binding()
        if binding is None:
            raise ValueError(
                "Configured TELEGRAM_TEST_TARGET_EMAIL has no connected Telegram binding. "
                "Cannot send test message."
            )
        chat_id = str(binding.external_chat_id).strip()
        expected_chat_id = settings.telegram_test_expected_chat_id.strip()
        if expected_chat_id and chat_id != expected_chat_id:
            raise ValueError(
                "Resolved chat_id does not match TELEGRAM_TEST_EXPECTED_CHAT_ID. "
                "Test message blocked."
            )

        sent = await self._bot_client().send_message(chat_id=chat_id, text=text)
        message_id = getattr(sent, "message_id", None)
        return {
            "configured_email": settings.telegram_test_target_email.strip(),
            "target_user_id": str(binding.user_id),
            "target_chat_id": chat_id,
            "target_username": binding.external_username,
            "message_id": str(message_id) if message_id is not None else None,
        }

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

    async def _safe_finalize_approval_message(
        self,
        *,
        callback: Any,
        status: str,
        locale: str,
        decided_at: datetime,
    ) -> None:
        message = getattr(callback, "message", None)
        if message is None:
            return
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        message_id = getattr(message, "message_id", None)
        if chat_id is None or message_id is None:
            return

        current_text = ""
        raw_text = getattr(message, "text", None)
        if isinstance(raw_text, str):
            current_text = raw_text
        else:
            raw_caption = getattr(message, "caption", None)
            if isinstance(raw_caption, str):
                current_text = raw_caption

        decision_line = self._approval_decision_line(
            status=status,
            locale=locale,
            decided_at=decided_at,
        )
        if current_text:
            next_text = self._append_decision_line(current_text, decision_line=decision_line)
            if next_text:
                try:
                    await self._bot_client().edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=next_text,
                        reply_markup=None,
                    )
                    return
                except TelegramError as exc:
                    lowered = str(exc).lower()
                    if "message is not modified" not in lowered:
                        logger.warning("Telegram edit_message_text failed: %s", exc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Telegram edit_message_text failed: %s", exc)

        try:
            await self._bot_client().edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=None,
            )
        except TelegramError as exc:
            logger.warning("Telegram edit_message_reply_markup failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram edit_message_reply_markup failed: %s", exc)

    @staticmethod
    def _append_decision_line(current_text: str, *, decision_line: str) -> str:
        base = current_text.rstrip()
        if not base:
            return decision_line
        if decision_line in base:
            return base
        candidate = f"{base}\n\n{decision_line}"
        if len(candidate) <= 4096:
            return candidate
        available = 4096 - len(decision_line) - 2
        if available <= 1:
            return decision_line[:4096]
        trimmed = base[: available - 1].rstrip()
        return f"{trimmed}…\n\n{decision_line}"

    @staticmethod
    def _approval_decision_line(
        *,
        status: str,
        locale: str,
        decided_at: datetime,
    ) -> str:
        stamp = decided_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        if locale == "zh":
            map_zh = {
                "approved": "✅ 状态：已批准（按钮已失效）",
                "rejected": "⛔ 状态：已拒绝（按钮已失效）",
                "expired": "⌛ 状态：已过期（按钮已失效）",
                "executing": "⏳ 状态：执行中（按钮已失效）",
                "executed": "✅ 状态：已执行（按钮已失效）",
                "failed": "⚠️ 状态：执行失败（按钮已失效）",
            }
            line = map_zh.get(status, "ℹ️ 状态：已处理（按钮已失效）")
            return f"{line}\n时间：{stamp}"
        map_en = {
            "approved": "✅ Status: Approved (buttons disabled)",
            "rejected": "⛔ Status: Rejected (buttons disabled)",
            "expired": "⌛ Status: Expired (buttons disabled)",
            "executing": "⏳ Status: Executing (buttons disabled)",
            "executed": "✅ Status: Executed (buttons disabled)",
            "failed": "⚠️ Status: Failed (buttons disabled)",
        }
        line = map_en.get(status, "ℹ️ Status: Processed (buttons disabled)")
        return f"{line}\nAt: {stamp}"

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

    async def _resolve_binding_locale(
        self,
        binding: Any,
        *,
        fallback_locale: str | None = None,
    ) -> str:
        metadata_locale = (binding.metadata_ or {}).get("locale") if hasattr(binding, "metadata_") else None
        return await self.social_service.resolve_user_locale(
            user_id=binding.user_id,
            fallback_locale=fallback_locale or metadata_locale,
        )
