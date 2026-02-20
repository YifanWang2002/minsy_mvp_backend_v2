"""Telegram connector test batches and rich interaction handlers.

This module intentionally keeps Telegram feature demos isolated from the main
connector flow so production business logic stays decoupled.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from html import escape
from typing import Any
from urllib.parse import urlencode

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputMediaPhoto,
    InputTextMessageContent,
    LabeledPrice,
    MenuButtonWebApp,
    Update,
    WebAppInfo,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError

from src.config import settings
from src.models.social_connector import SocialConnectorBinding
from src.services.social_connector_service import SocialConnectorService
from src.util.logger import logger

_TELEGRAM_TEST_META_KEY = "telegram_test"
_TRADE_SYMBOL = "NASDAQ:AAPL"
_TRADE_INTERVALS: tuple[str, ...] = ("1m", "5m", "1h", "1d")

_SYMBOL_RE = re.compile(r"^[A-Z0-9:_\-\.]{1,32}$")


class TelegramTestBatchService:
    """Owns Telegram feature-test batching and interaction handlers."""

    def __init__(
        self,
        *,
        db: AsyncSession,
        social_service: SocialConnectorService,
        bot: Bot,
    ) -> None:
        self.db = db
        self.social_service = social_service
        self.bot = bot
        self._openai_client: AsyncOpenAI | None = None

    async def send_post_connect_batches(
        self,
        *,
        chat_id: str,
        locale: str,
        binding: SocialConnectorBinding,
    ) -> None:
        """Send feature-test messages in explicit batches after successful bind."""
        if not settings.telegram_test_batches_enabled:
            return

        signal_id = self._build_signal_id()
        await self._persist_test_state(
            binding,
            {
                "last_signal_id": signal_id,
                "last_symbol": _TRADE_SYMBOL,
                "last_interval": "1d",
            },
        )

        batches = (
            self._batch_configure_menu(chat_id=chat_id, locale=locale, signal_id=signal_id),
            self._batch_trade_opportunity(chat_id=chat_id, locale=locale, signal_id=signal_id),
            self._batch_chat_intro(chat_id=chat_id, locale=locale),
            self._batch_content_features(chat_id=chat_id, locale=locale),
            self._batch_poll(chat_id=chat_id, locale=locale),
            self._batch_payments(chat_id=chat_id, locale=locale, signal_id=signal_id),
        )

        for batch in batches:
            try:
                await batch
            except TelegramError as exc:
                logger.warning("Telegram test batch failed: %s", exc)
            await asyncio.sleep(0.15)

    async def handle_connected_text_message(
        self,
        *,
        update: Update,
        binding: SocialConnectorBinding,
        text: str,
        locale: str,
    ) -> bool:
        """Handle text for test chat mode, including /reset and OpenAI replies."""
        if not settings.telegram_test_batches_enabled:
            return False

        message = update.message
        if message is None:
            return False

        chat_id = str(message.chat.id)
        normalized = text.strip()
        lowered = normalized.lower()

        if lowered.startswith("/reset"):
            await self._persist_test_state(binding, {"previous_response_id": None})
            reset_text = (
                "已重置对话上下文。你可以继续提问。"
                if locale == "zh"
                else "Context reset. You can start a new conversation now."
            )
            await self._safe_send_message(chat_id=chat_id, text=reset_text)
            return True

        if lowered.startswith("/signal"):
            signal_id = self._build_signal_id()
            await self._persist_test_state(
                binding,
                {
                    "last_signal_id": signal_id,
                    "last_symbol": _TRADE_SYMBOL,
                    "last_interval": "1d",
                },
            )
            await self._batch_trade_opportunity(chat_id=chat_id, locale=locale, signal_id=signal_id)
            return True

        if lowered.startswith("/help"):
            help_text = self._build_help_text(locale=locale)
            await self._safe_send_message(chat_id=chat_id, text=help_text)
            return True

        await self._safe_send_chat_action(chat_id=chat_id)
        thinking_text = "正在调用 OpenAI…" if locale == "zh" else "Calling OpenAI..."
        placeholder = await self._safe_send_message(chat_id=chat_id, text=thinking_text)

        previous_response_id = self._read_test_state(binding).get("previous_response_id")
        prev_id = previous_response_id if isinstance(previous_response_id, str) else None

        assistant_text: str
        response_id: str | None
        try:
            assistant_text, response_id = await self._request_openai_response(
                user_text=normalized,
                previous_response_id=prev_id,
                locale=locale,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram OpenAI chat failed: %s", exc)
            assistant_text = (
                "抱歉，OpenAI 暂时不可用，请稍后再试。"
                if locale == "zh"
                else "OpenAI is temporarily unavailable. Please try again later."
            )
            response_id = None

        assistant_text = assistant_text.strip() or (
            "我目前没有可返回的内容。" if locale == "zh" else "No output was returned."
        )

        if response_id is not None:
            await self._persist_test_state(binding, {"previous_response_id": response_id})

        if placeholder is not None and hasattr(placeholder, "message_id"):
            await self._pseudo_stream_edit(
                chat_id=chat_id,
                message_id=int(placeholder.message_id),
                full_text=assistant_text,
            )
        else:
            await self._safe_send_message(chat_id=chat_id, text=self._trim_telegram_text(assistant_text))
        return True

    async def handle_callback_query(
        self,
        *,
        update: Update,
        binding: SocialConnectorBinding,
        locale: str,
    ) -> bool:
        """Handle test callback actions (open/ignore/refresh/interval/details)."""
        if not settings.telegram_test_batches_enabled:
            return False

        callback = update.callback_query
        if callback is None or callback.message is None:
            return False

        data = (callback.data or "").strip()
        if not data:
            return False

        chat_id = str(callback.message.chat.id)
        message_id = callback.message.message_id

        if data.startswith("trade_open:") or data.startswith("trade_ignore:"):
            action = "open" if data.startswith("trade_open:") else "ignore"
            signal_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value=action,
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "trade_decision", "signal_id": signal_id},
            )
            callback_ack = (
                "已确认开单" if action == "open" else "已忽略该机会"
            ) if locale == "zh" else (
                "Order confirmation received" if action == "open" else "Opportunity ignored"
            )
            await self._safe_answer_callback(callback_query_id=callback.id, text=callback_ack)

            status_text = self._build_trade_status_text(
                locale=locale,
                signal_id=signal_id,
                action=action,
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=status_text,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("trade_interval:"):
            parts = data.split(":")
            if len(parts) < 3:
                return False
            interval = parts[1].strip().lower()
            signal_id = parts[2].strip() or "unknown"
            if interval not in _TRADE_INTERVALS:
                return False

            await self._persist_test_state(binding, {"last_interval": interval})
            chart_url = build_telegram_test_webapp_url(
                symbol=_TRADE_SYMBOL,
                signal_id=signal_id,
                interval=interval,
                locale=locale,
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    f"已切换到 {interval}"
                    if locale == "zh"
                    else f"Switched to {interval}"
                ),
            )
            text = (
                f"已切换图表周期到 {interval}，点击下方按钮重新打开图表。"
                if locale == "zh"
                else f"Chart interval updated to {interval}. Re-open the Web App chart below."
            )
            markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text="📊 打开图表(WebApp)" if locale == "zh" else "📊 Open Chart (Web App)",
                            web_app=WebAppInfo(url=chart_url),
                        )
                    ]
                ]
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=markup,
            )
            return True

        if data.startswith("trade_refresh:"):
            signal_id = data.split(":", 1)[1].strip() or self._build_signal_id()
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已刷新信号" if locale == "zh" else "Signal refreshed",
            )
            await self._batch_trade_opportunity(
                chat_id=chat_id,
                locale=locale,
                signal_id=signal_id,
            )
            return True

        if data.startswith("trade_more:"):
            details = (
                "策略细节:\n- 信号来源: EMA(12/26) + RSI\n- 方向: 多头\n- 风控: 1.2% 止损"
                if locale == "zh"
                else "Signal details:\n- Source: EMA(12/26) + RSI\n- Direction: Long\n- Risk: 1.2% SL"
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=details,
                show_alert=True,
            )
            return True

        return False

    async def handle_web_app_data(
        self,
        *,
        update: Update,
        binding: SocialConnectorBinding,
        locale: str,
    ) -> bool:
        """Handle data posted back from Telegram Web App."""
        if not settings.telegram_test_batches_enabled:
            return False

        message = update.message
        if message is None or message.web_app_data is None:
            return False

        chat_id = str(message.chat.id)
        raw = (message.web_app_data.data or "").strip()
        payload: dict[str, Any] | None = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None

        payload_for_db: dict[str, Any] = payload if payload is not None else {"raw": raw}
        await self.social_service.record_telegram_activity(
            user_id=binding.user_id,
            event_type="text",
            choice_value=None,
            message_text=raw[:1000],
            external_update_id=update.update_id,
            payload={"kind": "web_app_data", "data": payload_for_db},
        )

        symbol = payload_for_db.get("symbol") if isinstance(payload_for_db, dict) else None
        interval = payload_for_db.get("interval") if isinstance(payload_for_db, dict) else None
        symbol_text = str(symbol) if symbol else _TRADE_SYMBOL
        interval_text = str(interval) if interval else "1d"
        ack = (
            f"已收到 WebApp 回传参数: {symbol_text} / {interval_text}"
            if locale == "zh"
            else f"Received WebApp payload: {symbol_text} / {interval_text}"
        )
        await self._safe_send_message(chat_id=chat_id, text=ack)
        return True

    async def handle_inline_query(self, *, update: Update, locale: str) -> bool:
        """Serve inline mode test cards for @bot keyword calls."""
        if not settings.telegram_test_batches_enabled:
            return False

        inline_query = update.inline_query
        if inline_query is None:
            return False

        raw_query = (inline_query.query or "").strip().upper()
        symbol = raw_query if _is_valid_symbol(raw_query) else _TRADE_SYMBOL
        signal_id = self._build_signal_id()
        chart_url = build_telegram_test_webapp_url(
            symbol=symbol,
            signal_id=signal_id,
            interval="1d",
            locale=locale,
        )

        message_text = (
            f"Inline 模式机会卡片\n标的: {symbol}\nSignal ID: {signal_id}\n图表: {chart_url}"
            if locale == "zh"
            else f"Inline signal card\nSymbol: {symbol}\nSignal ID: {signal_id}\nChart: {chart_url}"
        )

        result = InlineQueryResultArticle(
            id=f"signal-{signal_id}",
            title=(f"{symbol} 交易机会" if locale == "zh" else f"{symbol} Trading Signal"),
            description=("发送机会卡片到当前对话" if locale == "zh" else "Send signal card to this chat"),
            input_message_content=InputTextMessageContent(message_text=message_text),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(text="✅ 开单", callback_data=f"trade_open:{signal_id}"),
                        InlineKeyboardButton(text="🙈 忽略", callback_data=f"trade_ignore:{signal_id}"),
                    ],
                    [
                        InlineKeyboardButton(
                            text="📊 图表(WebApp)" if locale == "zh" else "📊 Chart (Web App)",
                            web_app=WebAppInfo(url=chart_url),
                        )
                    ],
                ]
            ),
        )

        await self._safe_answer_inline_query(
            inline_query_id=inline_query.id,
            results=[result],
        )
        return True

    async def _batch_configure_menu(self, *, chat_id: str, locale: str, signal_id: str) -> None:
        chart_url = build_telegram_test_webapp_url(
            symbol=_TRADE_SYMBOL,
            signal_id=signal_id,
            interval="1d",
            locale=locale,
        )
        commands = [
            BotCommand(command="reset", description="重置对话上下文" if locale == "zh" else "Reset chat context"),
            BotCommand(command="signal", description="触发一条测试机会" if locale == "zh" else "Push a test signal"),
            BotCommand(command="help", description="查看测试命令" if locale == "zh" else "Show command help"),
        ]
        try:
            await self.bot.set_my_commands(commands=commands)
        except TelegramError as exc:
            logger.warning("Telegram set_my_commands failed: %s", exc)

        try:
            await self.bot.set_chat_menu_button(
                chat_id=chat_id,
                menu_button=MenuButtonWebApp(
                    text="打开交易面板" if locale == "zh" else "Open Trading Panel",
                    web_app=WebAppInfo(url=chart_url),
                ),
            )
        except TelegramError as exc:
            logger.warning("Telegram set_chat_menu_button failed: %s", exc)

    async def _batch_trade_opportunity(self, *, chat_id: str, locale: str, signal_id: str) -> None:
        chart_url = build_telegram_test_webapp_url(
            symbol=_TRADE_SYMBOL,
            signal_id=signal_id,
            interval="1d",
            locale=locale,
        )
        trade_text = self._build_trade_signal_text(locale=locale, signal_id=signal_id)

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(text="✅ 开单", callback_data=f"trade_open:{signal_id}"),
                    InlineKeyboardButton(text="🙈 忽略", callback_data=f"trade_ignore:{signal_id}"),
                ],
                [
                    InlineKeyboardButton(
                        text="📊 打开图表(WebApp)" if locale == "zh" else "📊 Open Chart (Web App)",
                        web_app=WebAppInfo(url=chart_url),
                    )
                ],
                [
                    InlineKeyboardButton(text="1m", callback_data=f"trade_interval:1m:{signal_id}"),
                    InlineKeyboardButton(text="5m", callback_data=f"trade_interval:5m:{signal_id}"),
                    InlineKeyboardButton(text="1h", callback_data=f"trade_interval:1h:{signal_id}"),
                    InlineKeyboardButton(text="1d", callback_data=f"trade_interval:1d:{signal_id}"),
                ],
                [
                    InlineKeyboardButton(
                        text="🔄 刷新" if locale == "zh" else "🔄 Refresh",
                        callback_data=f"trade_refresh:{signal_id}",
                    ),
                    InlineKeyboardButton(
                        text="🧩 更多细节" if locale == "zh" else "🧩 More",
                        callback_data=f"trade_more:{signal_id}",
                    ),
                ],
            ]
        )

        await self._safe_send_message(
            chat_id=chat_id,
            text=trade_text,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    async def _batch_chat_intro(self, *, chat_id: str, locale: str) -> None:
        text = (
            "✅ 连接完成。现在你可以直接发消息给我，我会调用真实 OpenAI API 回复。\n"
            "支持命令：/reset（重置上下文）、/signal（推送机会）、/help。"
            if locale == "zh"
            else "✅ Connected. You can now chat with me. I will answer via the real OpenAI API.\n"
            "Commands: /reset (context reset), /signal (push signal), /help."
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="🔎 Inline 模式测试" if locale == "zh" else "🔎 Inline Mode Test",
                        switch_inline_query_current_chat="AAPL",
                    )
                ]
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=text, reply_markup=markup)

    async def _batch_content_features(self, *, chat_id: str, locale: str) -> None:
        rich_text = (
            "<b>交易机会摘要</b>\n"
            "<code>symbol=NASDAQ:AAPL</code>\n"
            "<code>risk=1.2%</code>\n"
            "<a href=\"https://www.tradingview.com/symbols/NASDAQ-AAPL/\">TradingView 页面</a>"
            if locale == "zh"
            else "<b>Signal Snapshot</b>\n"
            "<code>symbol=NASDAQ:AAPL</code>\n"
            "<code>risk=1.2%</code>\n"
            "<a href=\"https://www.tradingview.com/symbols/NASDAQ-AAPL/\">TradingView page</a>"
        )
        await self._safe_send_message(
            chat_id=chat_id,
            text=rich_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=False,
        )

        media = [
            InputMediaPhoto(media="https://picsum.photos/seed/minsy-1d/1280/720", caption="1D"),
            InputMediaPhoto(media="https://picsum.photos/seed/minsy-1h/1280/720", caption="1H"),
            InputMediaPhoto(media="https://picsum.photos/seed/minsy-5m/1280/720", caption="5M"),
        ]
        await self._safe_send_media_group(chat_id=chat_id, media=media)

    async def _batch_poll(self, *, chat_id: str, locale: str) -> None:
        question = "你的风险偏好是？" if locale == "zh" else "What is your risk preference?"
        options = [
            "保守" if locale == "zh" else "Conservative",
            "平衡" if locale == "zh" else "Balanced",
            "激进" if locale == "zh" else "Aggressive",
        ]
        try:
            await self.bot.send_poll(
                chat_id=chat_id,
                question=question,
                options=options,
                type="quiz",
                correct_option_id=1,
                is_anonymous=False,
                explanation=(
                    "测试问卷：用于后续个性化信号。"
                    if locale == "zh"
                    else "Test quiz for future personalization."
                ),
            )
        except TelegramError as exc:
            logger.warning("Telegram send_poll failed: %s", exc)

    async def _batch_payments(self, *, chat_id: str, locale: str, signal_id: str) -> None:
        token = settings.telegram_test_payment_provider_token.strip()
        if not token:
            skip_text = (
                "Payment 测试已跳过：未配置 TELEGRAM_TEST_PAYMENT_PROVIDER_TOKEN。"
                if locale == "zh"
                else "Payments test skipped: TELEGRAM_TEST_PAYMENT_PROVIDER_TOKEN is not configured."
            )
            await self._safe_send_message(chat_id=chat_id, text=skip_text)
            return

        try:
            await self.bot.send_invoice(
                chat_id=chat_id,
                title="Minsy Pro Signals (Test)",
                description=(
                    "Telegram Payments 功能测试，不会触发真实交易。"
                    if locale == "zh"
                    else "Telegram Payments feature test. No real trading execution."
                ),
                payload=f"telegram-test-invoice:{signal_id}",
                provider_token=token,
                currency="USD",
                prices=[LabeledPrice(label="Pro Signal", amount=199)],
                start_parameter="minsy-telegram-test",
            )
        except TelegramError as exc:
            logger.warning("Telegram send_invoice failed: %s", exc)

    async def _request_openai_response(
        self,
        *,
        user_text: str,
        previous_response_id: str | None,
        locale: str,
    ) -> tuple[str, str | None]:
        instructions = (
            "你是一个交易助手，请简洁回答用户，并在必要时提醒风险。"
            if locale == "zh"
            else "You are a trading assistant. Keep responses concise and mention risk when relevant."
        )
        request_kwargs: dict[str, Any] = {
            "model": settings.openai_response_model,
            "input": user_text,
            "instructions": instructions,
        }
        if previous_response_id:
            request_kwargs["previous_response_id"] = previous_response_id

        response = await self._get_openai_client().responses.create(**request_kwargs)
        response_id = self._extract_response_id(response)
        text = self._extract_response_text(response)
        return text, response_id

    async def _pseudo_stream_edit(self, *, chat_id: str, message_id: int, full_text: str) -> None:
        output = self._trim_telegram_text(full_text)
        steps = self._chunk_text_for_streaming(output)
        rendered = ""
        for idx, piece in enumerate(steps):
            rendered = f"{rendered}{piece}"
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=self._trim_telegram_text(rendered),
            )
            if idx < len(steps) - 1:
                await asyncio.sleep(0.55)

    async def _persist_test_state(
        self,
        binding: SocialConnectorBinding,
        updates: dict[str, Any],
    ) -> None:
        metadata = dict(binding.metadata_ or {})
        test_state_raw = metadata.get(_TELEGRAM_TEST_META_KEY)
        test_state = dict(test_state_raw) if isinstance(test_state_raw, dict) else {}
        for key, value in updates.items():
            if value is None:
                test_state.pop(key, None)
            else:
                test_state[key] = value
        test_state["updated_at"] = datetime.now(UTC).isoformat()
        metadata[_TELEGRAM_TEST_META_KEY] = test_state
        binding.metadata_ = metadata
        await self.db.flush()

    @staticmethod
    def _read_test_state(binding: SocialConnectorBinding) -> dict[str, Any]:
        metadata = dict(binding.metadata_ or {})
        raw = metadata.get(_TELEGRAM_TEST_META_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    @staticmethod
    def _extract_response_id(response: Any) -> str | None:
        value = getattr(response, "id", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @classmethod
    def _extract_response_text(cls, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        if isinstance(output_text, list):
            parts = [str(item).strip() for item in output_text if str(item).strip()]
            if parts:
                return "\n".join(parts)

        dumped: dict[str, Any] | None = None
        if hasattr(response, "model_dump"):
            try:
                dumped_candidate = response.model_dump(mode="json", exclude_none=True, warnings=False)
                if isinstance(dumped_candidate, dict):
                    dumped = dumped_candidate
            except Exception:  # noqa: BLE001
                dumped = None

        if dumped is not None:
            parsed = cls._extract_response_text_from_dict(dumped)
            if parsed:
                return parsed

        return str(response).strip()

    @classmethod
    def _extract_response_text_from_dict(cls, payload: dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        parts: list[str] = []
        output_items = payload.get("output")
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                        continue
                    if isinstance(text, dict):
                        text_value = text.get("value")
                        if isinstance(text_value, str) and text_value.strip():
                            parts.append(text_value.strip())

        if parts:
            return "\n".join(parts)
        return ""

    @staticmethod
    def _chunk_text_for_streaming(text: str, *, chunk_size: int = 220) -> list[str]:
        compact = text.strip()
        if not compact:
            return [""]

        chunks: list[str] = []
        current = ""
        for char in compact:
            current += char
            if len(current) >= chunk_size and char in {"。", "！", "？", ".", "!", "?", "\n"}:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)
        return chunks or [compact]

    @staticmethod
    def _trim_telegram_text(text: str, *, max_len: int = 3900) -> str:
        compact = text.strip()
        if len(compact) <= max_len:
            return compact
        return f"{compact[:max_len]}..."

    @staticmethod
    def _build_signal_id() -> str:
        return datetime.now(UTC).strftime("sig%Y%m%d%H%M%S")

    @staticmethod
    def _build_trade_signal_text(*, locale: str, signal_id: str) -> str:
        if locale == "zh":
            return (
                "<b>交易机会出现</b>\n"
                f"Signal ID: <code>{escape(signal_id)}</code>\n"
                "标的: <code>NASDAQ:AAPL</code>\n"
                "方向: <b>Long</b>\n"
                "建议: 点击下方按钮选择 <b>开单</b> 或 <b>忽略</b>，并通过 WebApp 查看内嵌图表。"
            )
        return (
            "<b>New Trading Opportunity</b>\n"
            f"Signal ID: <code>{escape(signal_id)}</code>\n"
            "Symbol: <code>NASDAQ:AAPL</code>\n"
            "Direction: <b>Long</b>\n"
            "Action: choose <b>Open</b> or <b>Ignore</b>, then inspect the embedded WebApp chart."
        )

    @staticmethod
    def _build_trade_status_text(*, locale: str, signal_id: str, action: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        if locale == "zh":
            status = "已开单" if action == "open" else "已忽略"
            return (
                "<b>机会状态已更新</b>\n"
                f"Signal ID: <code>{escape(signal_id)}</code>\n"
                f"状态: <b>{status}</b>\n"
                f"时间: <code>{timestamp}</code>"
            )
        status_en = "Opened" if action == "open" else "Ignored"
        return (
            "<b>Signal Status Updated</b>\n"
            f"Signal ID: <code>{escape(signal_id)}</code>\n"
            f"Status: <b>{status_en}</b>\n"
            f"Time: <code>{timestamp}</code>"
        )

    @staticmethod
    def _build_help_text(*, locale: str) -> str:
        if locale == "zh":
            return (
                "可用测试命令:\n"
                "/reset - 重置 OpenAI 对话上下文\n"
                "/signal - 立即推送一条交易机会\n"
                "/help - 查看命令列表\n"
                "你也可以直接发送任意文本进行对话。"
            )
        return (
            "Available test commands:\n"
            "/reset - reset OpenAI conversation context\n"
            "/signal - push a trade opportunity now\n"
            "/help - show this command list\n"
            "You can also send any text to chat."
        )

    async def _safe_send_message(
        self,
        *,
        chat_id: str,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = None,
        disable_web_page_preview: bool | None = None,
    ) -> Any | None:
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview,
            )
        except TelegramError as exc:
            logger.warning("Telegram send_message failed: %s", exc)
            return None

    async def _safe_edit_message_text(
        self,
        *,
        chat_id: str,
        message_id: int,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = None,
    ) -> None:
        try:
            await self.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
        except TelegramError as exc:
            logger.warning("Telegram edit_message_text failed: %s", exc)

    async def _safe_answer_callback(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None:
        try:
            await self.bot.answer_callback_query(
                callback_query_id=callback_query_id,
                text=text,
                show_alert=show_alert,
            )
        except TelegramError as exc:
            logger.warning("Telegram answer_callback_query failed: %s", exc)

    async def _safe_send_chat_action(self, *, chat_id: str) -> None:
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except TelegramError as exc:
            logger.warning("Telegram send_chat_action failed: %s", exc)

    async def _safe_send_media_group(self, *, chat_id: str, media: list[InputMediaPhoto]) -> None:
        try:
            await self.bot.send_media_group(chat_id=chat_id, media=media)
        except TelegramError as exc:
            logger.warning("Telegram send_media_group failed: %s", exc)

    async def _safe_answer_inline_query(
        self,
        *,
        inline_query_id: str,
        results: list[InlineQueryResultArticle],
    ) -> None:
        try:
            await self.bot.answer_inline_query(
                inline_query_id=inline_query_id,
                results=results,
                cache_time=0,
                is_personal=True,
            )
        except TelegramError as exc:
            logger.warning("Telegram answer_inline_query failed: %s", exc)


def build_telegram_test_webapp_url(
    *,
    symbol: str,
    signal_id: str,
    interval: str,
    locale: str,
) -> str:
    """Build absolute Web App URL for the Telegram chart page."""
    base = settings.telegram_webapp_base_url.strip().rstrip("/")
    if not base:
        base = "https://app.minsyai.com"

    endpoint = f"{settings.api_v1_prefix}/social/connectors/telegram/test-webapp/chart"
    query = urlencode(
        {
            "symbol": _sanitize_symbol(symbol),
            "interval": _sanitize_interval(interval),
            "locale": "zh" if locale == "zh" else "en",
            "signal_id": signal_id,
        }
    )
    return f"{base}{endpoint}?{query}"


def build_telegram_test_chart_html(
    *,
    symbol: str,
    interval: str,
    locale: str,
    theme: str,
    signal_id: str,
) -> str:
    """Render chart Web App HTML with embedded TradingView widget."""
    safe_symbol = _sanitize_symbol(symbol)
    safe_interval = _sanitize_interval(interval)
    safe_locale = "zh" if locale == "zh" else "en"
    safe_theme = "dark" if theme == "dark" else "light"
    safe_signal = escape(signal_id.strip() or "unknown")

    tradingview_interval = {
        "1m": "1",
        "5m": "5",
        "1h": "60",
        "1d": "D",
        "d": "D",
    }.get(safe_interval.lower(), "D")

    tv_locale = "zh_CN" if safe_locale == "zh" else "en"
    widget_config = {
        "allow_symbol_change": True,
        "calendar": False,
        "details": False,
        "hide_side_toolbar": True,
        "hide_top_toolbar": False,
        "hide_legend": False,
        "hide_volume": False,
        "hotlist": False,
        "interval": tradingview_interval,
        "locale": tv_locale,
        "save_image": True,
        "style": "1",
        "symbol": safe_symbol,
        "theme": safe_theme,
        "timezone": "Etc/UTC",
        "backgroundColor": "#ffffff" if safe_theme == "light" else "#0f172a",
        "gridColor": "rgba(46, 46, 46, 0.06)",
        "watchlist": [],
        "withdateranges": False,
        "compareSymbols": [],
        "studies": [],
        "autosize": True,
    }

    title = "Minsy Telegram TradingView 测试" if safe_locale == "zh" else "Minsy Telegram TradingView Test"
    action_label = "✅ 回传开单参数" if safe_locale == "zh" else "✅ Send Trade Params"

    return f"""<!doctype html>
<html lang="{safe_locale}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{escape(title)}</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
      html, body {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
      .page {{ min-height: 100%; display: flex; flex-direction: column; background: #f8fafc; color: #0f172a; }}
      .header {{ padding: 12px 14px; font-size: 14px; border-bottom: 1px solid #e2e8f0; }}
      .chart-wrap {{ flex: 1; min-height: 360px; }}
      .footer {{ padding: 12px 14px 18px; border-top: 1px solid #e2e8f0; display: grid; gap: 10px; }}
      button {{ border: 0; border-radius: 10px; background: #0ea5e9; color: white; font-size: 15px; padding: 11px 14px; }}
      .meta {{ font-size: 12px; color: #64748b; }}
    </style>
  </head>
  <body>
    <div class="page">
      <div class="header">Signal ID: <b>{safe_signal}</b> | Symbol: <b>{escape(safe_symbol)}</b> | Interval: <b>{escape(safe_interval)}</b></div>
      <div class="chart-wrap">
        <div class="tradingview-widget-container" style="height:100%;width:100%">
          <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
          <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/NASDAQ-AAPL/" rel="noopener nofollow" target="_blank"><span class="blue-text">AAPL stock chart</span></a><span class="trademark"> by TradingView</span></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>{json.dumps(widget_config, ensure_ascii=False)}</script>
        </div>
      </div>
      <div class="footer">
        <button id="sendDataBtn">{escape(action_label)}</button>
        <div class="meta">Web App Data: symbol={escape(safe_symbol)}, interval={escape(safe_interval)}, signal_id={safe_signal}</div>
      </div>
    </div>
    <script>
      const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
      if (tg) {{
        tg.ready();
        tg.expand();
      }}

      document.getElementById('sendDataBtn').addEventListener('click', function () {{
        const payload = {{
          type: 'trade_confirm',
          signal_id: '{safe_signal}',
          symbol: '{escape(safe_symbol)}',
          interval: '{escape(safe_interval)}',
          ts_utc: new Date().toISOString(),
        }};
        if (tg) {{
          tg.sendData(JSON.stringify(payload));
          if (tg.HapticFeedback) {{
            tg.HapticFeedback.notificationOccurred('success');
          }}
          tg.close();
        }}
      }});
    </script>
  </body>
</html>
"""


def _sanitize_symbol(raw: str) -> str:
    symbol = (raw or "").strip().upper()
    if _is_valid_symbol(symbol):
        return symbol
    return _TRADE_SYMBOL


def _is_valid_symbol(raw: str) -> bool:
    return bool(raw and _SYMBOL_RE.fullmatch(raw))


def _sanitize_interval(raw: str) -> str:
    interval = (raw or "").strip().lower()
    if interval in {"1m", "5m", "1h", "1d", "d"}:
        return interval
    return "1d"
