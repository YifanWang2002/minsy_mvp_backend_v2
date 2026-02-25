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
from urllib.parse import urlencode, urlparse

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
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
from src.observability.openai_cost import build_turn_usage_snapshot
from src.services.social_connector_service import SocialConnectorService
from src.util.logger import logger

_TELEGRAM_TEST_META_KEY = "telegram_test"
_TRADE_SYMBOL = "NASDAQ:AAPL"
_TRADE_INTERVALS: tuple[str, ...] = ("1m", "5m", "1h", "1d")
_PRE_STRATEGY_MARKETS: tuple[str, ...] = ("us_stocks", "crypto", "forex", "futures")
_FREQ_BUCKETS: tuple[str, ...] = ("few_per_month", "few_per_week", "daily", "multiple_per_day")
_HOLDING_BUCKETS: tuple[str, ...] = (
    "intraday_scalp",
    "intraday",
    "swing_days",
    "position_weeks_plus",
)
_DEPLOYMENT_STATUS: tuple[str, ...] = ("ready", "deployed", "blocked")

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
        try:
            target_chat_id = await self._resolve_test_target_chat_id(chat_id=chat_id)
        except ValueError as exc:
            logger.warning("Telegram test target routing failed: %s", exc)
            await self._safe_send_message(chat_id=chat_id, text=str(exc))
            return

        signal_id = self._build_signal_id()
        backtest_id = self._build_event_id(prefix="bt")
        position_id = self._build_event_id(prefix="pos")
        regime_id = self._build_event_id(prefix="reg")
        await self._persist_test_state(
            binding,
            {
                "last_signal_id": signal_id,
                "last_symbol": _TRADE_SYMBOL,
                "last_interval": "1d",
                "last_backtest_id": backtest_id,
                "last_position_id": position_id,
                "last_regime_id": regime_id,
            },
        )

        batches = (
            self._batch_configure_menu(chat_id=target_chat_id, locale=locale, signal_id=signal_id),
            self._batch_flow_intro(chat_id=target_chat_id, locale=locale),
            self._batch_pre_strategy_scope(chat_id=target_chat_id, locale=locale),
            self._batch_trade_opportunity(chat_id=target_chat_id, locale=locale, signal_id=signal_id),
            self._batch_chat_intro(chat_id=target_chat_id, locale=locale),
            self._batch_deployment_status(chat_id=target_chat_id, locale=locale),
            self._batch_backtest_completed(chat_id=target_chat_id, locale=locale, backtest_id=backtest_id),
            self._batch_live_trade_open_reminder(chat_id=target_chat_id, locale=locale, position_id=position_id),
            self._batch_live_trade_close_reminder(chat_id=target_chat_id, locale=locale, position_id=position_id),
            self._batch_market_regime_change(chat_id=target_chat_id, locale=locale, regime_id=regime_id),
            self._batch_poll(chat_id=target_chat_id, locale=locale),
            self._batch_payments(chat_id=target_chat_id, locale=locale, signal_id=signal_id),
        )

        for batch in batches:
            try:
                await batch
            except TelegramError as exc:
                logger.warning("Telegram test batch failed: %s", exc)
            await asyncio.sleep(0.15)

    async def send_post_strategy_batches(
        self,
        *,
        chat_id: str,
        locale: str,
        binding: SocialConnectorBinding,
    ) -> None:
        """Replay post-strategy alert batches only."""
        try:
            target_chat_id = await self._resolve_test_target_chat_id(chat_id=chat_id)
        except ValueError as exc:
            logger.warning("Telegram test target routing failed: %s", exc)
            await self._safe_send_message(chat_id=chat_id, text=str(exc))
            return
        backtest_id = self._build_event_id(prefix="bt")
        position_id = self._build_event_id(prefix="pos")
        regime_id = self._build_event_id(prefix="reg")
        await self._persist_test_state(
            binding,
            {
                "last_backtest_id": backtest_id,
                "last_position_id": position_id,
                "last_regime_id": regime_id,
            },
        )
        batches = (
            self._batch_backtest_completed(chat_id=target_chat_id, locale=locale, backtest_id=backtest_id),
            self._batch_live_trade_open_reminder(chat_id=target_chat_id, locale=locale, position_id=position_id),
            self._batch_live_trade_close_reminder(chat_id=target_chat_id, locale=locale, position_id=position_id),
            self._batch_market_regime_change(chat_id=target_chat_id, locale=locale, regime_id=regime_id),
        )
        for batch in batches:
            try:
                await batch
            except TelegramError as exc:
                logger.warning("Telegram post-strategy batch failed: %s", exc)
            await asyncio.sleep(0.12)

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
            target_chat_id = await self._resolve_target_chat_or_report(
                source_chat_id=chat_id,
                command="signal",
            )
            if target_chat_id is None:
                return True
            signal_id = self._build_signal_id()
            await self._persist_test_state(
                binding,
                {
                    "last_signal_id": signal_id,
                    "last_symbol": _TRADE_SYMBOL,
                    "last_interval": "1d",
                },
            )
            await self._batch_trade_opportunity(chat_id=target_chat_id, locale=locale, signal_id=signal_id)
            return True

        if lowered.startswith("/testall"):
            target_chat_id = await self._resolve_target_chat_or_report(
                source_chat_id=chat_id,
                command="testall",
            )
            if target_chat_id is None:
                return True
            await self.send_post_connect_batches(chat_id=target_chat_id, locale=locale, binding=binding)
            return True

        if lowered.startswith("/postflow"):
            target_chat_id = await self._resolve_target_chat_or_report(
                source_chat_id=chat_id,
                command="postflow",
            )
            if target_chat_id is None:
                return True
            await self.send_post_strategy_batches(chat_id=target_chat_id, locale=locale, binding=binding)
            return True

        if lowered.startswith("/help"):
            target_chat_id = await self._resolve_target_chat_or_report(
                source_chat_id=chat_id,
                command="help",
            )
            if target_chat_id is None:
                return True
            help_text = self._build_help_text(locale=locale)
            await self._safe_send_message(chat_id=target_chat_id, text=help_text)
            return True

        await self._safe_send_chat_action(chat_id=chat_id)
        thinking_text = "正在调用 OpenAI…" if locale == "zh" else "Calling OpenAI..."
        placeholder = await self._safe_send_message(chat_id=chat_id, text=thinking_text)

        previous_response_id = self._read_test_state(binding).get("previous_response_id")
        prev_id = previous_response_id if isinstance(previous_response_id, str) else None

        assistant_text: str
        response_id: str | None
        usage_snapshot: dict[str, Any] | None
        try:
            assistant_text, response_id, usage_snapshot = await self._request_openai_response(
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
            usage_snapshot = None

        assistant_text = assistant_text.strip() or (
            "我目前没有可返回的内容。" if locale == "zh" else "No output was returned."
        )

        next_state_updates: dict[str, Any] = {}
        if response_id is not None:
            next_state_updates["previous_response_id"] = response_id
        if usage_snapshot is not None:
            next_state_updates["last_openai_usage"] = usage_snapshot
        if next_state_updates:
            await self._persist_test_state(binding, next_state_updates)

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

        if data.startswith("backtest_continue:"):
            backtest_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="backtest_continue",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "backtest_continue", "backtest_id": backtest_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    "收到，继续在聊天里优化策略。"
                    if locale == "zh"
                    else "Got it. Continue strategy iteration in chat."
                ),
            )
            summary = (
                "<b>Backtest 已完成</b>\n"
                f"backtest_id: <code>{escape(backtest_id)}</code>\n"
                "你可以直接回复：调整因子、参数、或风控阈值。"
                if locale == "zh"
                else "<b>Backtest Completed</b>\n"
                f"backtest_id: <code>{escape(backtest_id)}</code>\n"
                "Reply in chat to adjust factors, params, or risk controls."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("backtest_rerun:"):
            backtest_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="backtest_rerun",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "backtest_rerun", "backtest_id": backtest_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已登记重跑请求" if locale == "zh" else "Rerun request queued",
            )
            summary = (
                "<b>回测重跑请求已登记</b>\n"
                f"backtest_id: <code>{escape(backtest_id)}</code>\n"
                "测试环境中仅做提醒，不会触发真实任务。"
                if locale == "zh"
                else "<b>Backtest rerun request recorded</b>\n"
                f"backtest_id: <code>{escape(backtest_id)}</code>\n"
                "In this test flow, this is a notification only."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("live_open_ack:"):
            position_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="live_open_ack",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "live_open_ack", "position_id": position_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已确认开仓提醒" if locale == "zh" else "Open-position alert acknowledged",
            )
            summary = (
                "<b>实盘开仓提醒</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "状态: <b>已确认</b>"
                if locale == "zh"
                else "<b>Live Open-Position Alert</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "Status: <b>Acknowledged</b>"
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("live_force_close:"):
            position_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="live_force_close",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "live_force_close", "position_id": position_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已记录平仓请求" if locale == "zh" else "Close request recorded",
            )
            summary = (
                "<b>实盘开仓提醒</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "动作: <b>已请求紧急平仓</b>（测试）"
                if locale == "zh"
                else "<b>Live Open-Position Alert</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "Action: <b>Emergency close requested</b> (test)"
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("live_close_ack:"):
            position_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="live_close_ack",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "live_close_ack", "position_id": position_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已确认平仓回执" if locale == "zh" else "Close-position receipt acknowledged",
            )
            summary = (
                "<b>实盘平仓回执</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "状态: <b>已确认</b>"
                if locale == "zh"
                else "<b>Live Close-Position Receipt</b>\n"
                f"position_id: <code>{escape(position_id)}</code>\n"
                "Status: <b>Acknowledged</b>"
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("regime_ack:"):
            regime_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="regime_ack",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "regime_ack", "regime_id": regime_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已确认 regime 变动" if locale == "zh" else "Regime-change alert acknowledged",
            )
            summary = (
                "<b>Market Regime 变动</b>\n"
                f"regime_event_id: <code>{escape(regime_id)}</code>\n"
                "状态: <b>已确认</b>"
                if locale == "zh"
                else "<b>Market Regime Change</b>\n"
                f"regime_event_id: <code>{escape(regime_id)}</code>\n"
                "Status: <b>Acknowledged</b>"
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("regime_adjust:"):
            regime_id = data.split(":", 1)[1].strip() or "unknown"
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="regime_adjust",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "regime_adjust", "regime_id": regime_id},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text="已记录调整请求" if locale == "zh" else "Adjustment request recorded",
            )
            summary = (
                "<b>Market Regime 变动</b>\n"
                f"regime_event_id: <code>{escape(regime_id)}</code>\n"
                "动作: <b>请求调整策略参数</b>"
                if locale == "zh"
                else "<b>Market Regime Change</b>\n"
                f"regime_event_id: <code>{escape(regime_id)}</code>\n"
                "Action: <b>Request strategy parameter adjustment</b>"
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            await self._safe_send_message(
                chat_id=chat_id,
                text=(
                    "请直接回复你要调整的内容，例如：降低杠杆、提高止损阈值、切换到更长持仓周期。"
                    if locale == "zh"
                    else "Reply with what to adjust, e.g. reduce leverage, widen stop-loss threshold, or switch to longer holding period."
                ),
            )
            return True

        if data.startswith("flow_market:"):
            market = data.split(":", 1)[1].strip().lower()
            if market not in _PRE_STRATEGY_MARKETS:
                return False
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="flow_market",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "flow_market", "market": market},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    "已记录目标市场，继续设置机会频率。"
                    if locale == "zh"
                    else "Target market saved. Next: opportunity frequency."
                ),
            )
            summary = (
                f"<b>Pre-Strategy 已更新</b>\n目标市场: <code>{escape(market)}</code>\n下一步: 选择机会频率。"
                if locale == "zh"
                else f"<b>Pre-Strategy Updated</b>\nTarget market: <code>{escape(market)}</code>\nNext: choose opportunity frequency."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            await self._send_frequency_prompt(chat_id=chat_id, locale=locale)
            return True

        if data.startswith("flow_frequency:"):
            value = data.split(":", 1)[1].strip().lower()
            if value not in _FREQ_BUCKETS:
                return False
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="flow_freq",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "flow_frequency", "value": value},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    "已记录机会频率，继续设置持仓周期。"
                    if locale == "zh"
                    else "Opportunity frequency saved. Next: holding period."
                ),
            )
            summary = (
                f"<b>Pre-Strategy 已更新</b>\n机会频率: <code>{escape(value)}</code>\n下一步: 选择持仓周期。"
                if locale == "zh"
                else f"<b>Pre-Strategy Updated</b>\nOpportunity frequency: <code>{escape(value)}</code>\nNext: choose holding period."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            await self._send_holding_period_prompt(chat_id=chat_id, locale=locale)
            return True

        if data.startswith("flow_holding:"):
            value = data.split(":", 1)[1].strip().lower()
            if value not in _HOLDING_BUCKETS:
                return False
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="flow_hold",
                message_text=None,
                external_update_id=update.update_id,
                payload={"callback_data": data, "kind": "flow_holding", "value": value},
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    "已记录持仓周期，进入 Strategy 阶段。"
                    if locale == "zh"
                    else "Holding period saved. Entering strategy phase."
                ),
            )
            summary = (
                "<b>Pre-Strategy 完成</b>\n已收集 market / frequency / holding。\n"
                "下一步：进入 Strategy，验证 DSL，生成 strategy_draft_id 后由前端确认。"
                if locale == "zh"
                else "<b>Pre-Strategy Completed</b>\nCollected market / frequency / holding.\n"
                "Next: enter Strategy, validate DSL, generate strategy_draft_id, then confirm in frontend."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=summary,
                parse_mode=ParseMode.HTML,
            )
            return True

        if data.startswith("flow_deploy:"):
            status_value = data.split(":", 1)[1].strip().lower()
            if status_value not in _DEPLOYMENT_STATUS:
                return False
            await self.social_service.record_telegram_activity(
                user_id=binding.user_id,
                event_type="choice",
                choice_value="flow_deploy",
                message_text=None,
                external_update_id=update.update_id,
                payload={
                    "callback_data": data,
                    "kind": "flow_deployment_status",
                    "deployment_status": status_value,
                },
            )
            await self._safe_answer_callback(
                callback_query_id=callback.id,
                text=(
                    f"部署状态已更新为 {status_value}"
                    if locale == "zh"
                    else f"Deployment status updated to {status_value}"
                ),
            )
            deploy_text = (
                "<b>Deployment 阶段状态</b>\n"
                f"deployment_status: <code>{escape(status_value)}</code>\n"
                "说明：ready/deployed/blocked 将驱动后续交付分支。"
                if locale == "zh"
                else "<b>Deployment Phase Status</b>\n"
                f"deployment_status: <code>{escape(status_value)}</code>\n"
                "Note: ready/deployed/blocked drives handoff branches."
            )
            await self._safe_edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=deploy_text,
                parse_mode=ParseMode.HTML,
            )
            return True

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
                "策略细节:\n- 阶段: Strategy/artifact_ops\n- 来源: DSL + 回测筛选\n- 风控: 1.2% 止损（示例）"
                if locale == "zh"
                else "Signal details:\n- Stage: Strategy/artifact_ops\n- Source: DSL + backtest filter\n- Risk: 1.2% stop (sample)"
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
            f"⚡ Inline 机会卡片\n📌 标的: {symbol}\n🆔 Signal: {signal_id}\n📊 图表: {chart_url}"
            if locale == "zh"
            else f"⚡ Inline Signal Card\n📌 Symbol: {symbol}\n🆔 Signal: {signal_id}\n📊 Chart: {chart_url}"
        )

        result = InlineQueryResultArticle(
            id=f"signal-{signal_id}",
            title=(f"🚨 {symbol} 交易机会" if locale == "zh" else f"🚨 {symbol} Trading Opportunity"),
            description=(
                "发送机会卡片到当前聊天"
                if locale == "zh"
                else "Send this opportunity card to current chat"
            ),
            input_message_content=InputTextMessageContent(message_text=message_text),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            text=("✅ 开单" if locale == "zh" else "✅ Open"),
                            callback_data=f"trade_open:{signal_id}",
                        ),
                        InlineKeyboardButton(
                            text=("🙈 忽略" if locale == "zh" else "🙈 Ignore"),
                            callback_data=f"trade_ignore:{signal_id}",
                        ),
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
            BotCommand(command="signal", description="触发交易机会批次" if locale == "zh" else "Push trade opportunity batch"),
            BotCommand(command="testall", description="重放全部测试批次" if locale == "zh" else "Replay all test batches"),
            BotCommand(command="postflow", description="重放策略后提醒批次" if locale == "zh" else "Replay post-strategy alerts"),
            BotCommand(command="help", description="查看测试命令" if locale == "zh" else "Show test commands"),
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

    async def _batch_flow_intro(self, *, chat_id: str, locale: str) -> None:
        text = (
            "🎉 Telegram 连接成功！\n\n"
            "接下来会按真实流程推送测试批次：\n"
            "🧾 KYC → 🧭 Pre-Strategy → 🧠 Strategy → 🚀 Deployment\n\n"
            "🛡️ 说明：这些消息仅用于验证 Telegram 体验，不会改动你的正式策略数据。"
            if locale == "zh"
            else "🎉 Telegram connected successfully!\n\n"
            "You will now receive test batches that mirror the real journey:\n"
            "🧾 KYC → 🧭 Pre-Strategy → 🧠 Strategy → 🚀 Deployment\n\n"
            "🛡️ Note: these are test interactions only and won't modify production strategy data."
        )
        await self._safe_send_message(chat_id=chat_id, text=text)

    async def _batch_pre_strategy_scope(self, *, chat_id: str, locale: str) -> None:
        question = (
            "🧭 进入 Pre-Strategy 阶段\n"
            "请选择目标市场 `target_market`："
            if locale == "zh"
            else "🧭 Entering Pre-Strategy\n"
            "Please choose your target market (`target_market`):"
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text=("🇺🇸 美股 us_stocks" if locale == "zh" else "🇺🇸 US Stocks"),
                        callback_data="flow_market:us_stocks",
                    ),
                    InlineKeyboardButton(
                        text=("🪙 加密 crypto" if locale == "zh" else "🪙 Crypto"),
                        callback_data="flow_market:crypto",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=("💱 外汇 forex" if locale == "zh" else "💱 Forex"),
                        callback_data="flow_market:forex",
                    ),
                    InlineKeyboardButton(
                        text=("📉 期货 futures" if locale == "zh" else "📉 Futures"),
                        callback_data="flow_market:futures",
                    ),
                ],
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=question, reply_markup=markup)

    async def _send_frequency_prompt(self, *, chat_id: str, locale: str) -> None:
        question = (
            "⏱️ 请选择机会频率 `opportunity_frequency_bucket`："
            if locale == "zh"
            else "⏱️ Choose opportunity frequency (`opportunity_frequency_bucket`):"
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text=("📆 每月几次" if locale == "zh" else "📆 Few per month"),
                        callback_data="flow_frequency:few_per_month",
                    ),
                    InlineKeyboardButton(
                        text=("🗓️ 每周几次" if locale == "zh" else "🗓️ Few per week"),
                        callback_data="flow_frequency:few_per_week",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=("☀️ 每日" if locale == "zh" else "☀️ Daily"),
                        callback_data="flow_frequency:daily",
                    ),
                    InlineKeyboardButton(
                        text=("⚡ 日内多次" if locale == "zh" else "⚡ Multiple per day"),
                        callback_data="flow_frequency:multiple_per_day",
                    ),
                ],
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=question, reply_markup=markup)

    async def _send_holding_period_prompt(self, *, chat_id: str, locale: str) -> None:
        question = (
            "🕒 请选择持仓周期 `holding_period_bucket`："
            if locale == "zh"
            else "🕒 Choose holding period (`holding_period_bucket`):"
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text=("⚡ 超短 intraday_scalp" if locale == "zh" else "⚡ Intraday scalp"),
                        callback_data="flow_holding:intraday_scalp",
                    ),
                    InlineKeyboardButton(
                        text=("📌 日内 intraday" if locale == "zh" else "📌 Intraday"),
                        callback_data="flow_holding:intraday",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=("🌙 数日 swing_days" if locale == "zh" else "🌙 Swing days"),
                        callback_data="flow_holding:swing_days",
                    ),
                    InlineKeyboardButton(
                        text=("🧱 数周+ position_weeks_plus" if locale == "zh" else "🧱 Weeks+"),
                        callback_data="flow_holding:position_weeks_plus",
                    ),
                ],
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=question, reply_markup=markup)

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
                    InlineKeyboardButton(
                        text=("✅ 开单" if locale == "zh" else "✅ Open"),
                        callback_data=f"trade_open:{signal_id}",
                    ),
                    InlineKeyboardButton(
                        text=("🙈 忽略" if locale == "zh" else "🙈 Ignore"),
                        callback_data=f"trade_ignore:{signal_id}",
                    ),
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
            "🧠 已进入 Strategy 阶段\n"
            "先验证 DSL 生成 `strategy_draft_id`，前端确认后继续回测与迭代。\n\n"
            "💬 现在你可以直接和我聊天，我会调用真实 OpenAI API 回复。\n"
            "⌨️ 命令：/reset、/signal、/testall、/postflow、/help。"
            if locale == "zh"
            else "🧠 Strategy phase is live.\n"
            "First validate DSL and generate `strategy_draft_id`, then keep iterating with backtests.\n\n"
            "💬 You can chat with me now. Replies use the real OpenAI API.\n"
            "⌨️ Commands: /reset, /signal, /testall, /postflow, /help."
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

    async def _batch_deployment_status(self, *, chat_id: str, locale: str) -> None:
        text = (
            "🚀 Deployment 阶段测试\n"
            "请选择 `deployment_status`：ready / deployed / blocked"
            if locale == "zh"
            else "🚀 Deployment phase test\n"
            "Choose `deployment_status`: ready / deployed / blocked"
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(text="ready", callback_data="flow_deploy:ready"),
                    InlineKeyboardButton(text="deployed", callback_data="flow_deploy:deployed"),
                    InlineKeyboardButton(text="blocked", callback_data="flow_deploy:blocked"),
                ]
            ]
        )
        await self._safe_send_message(chat_id=chat_id, text=text, reply_markup=markup)

    async def _batch_backtest_completed(
        self,
        *,
        chat_id: str,
        locale: str,
        backtest_id: str,
    ) -> None:
        text = (
            "✅ <b>Backtest 已完成</b>\n"
            f"backtest_id: <code>{escape(backtest_id)}</code>\n"
            "📈 结果摘要: 年化 21.4% | 最大回撤 9.8% | Sharpe 1.57\n"
            "💡 你可以继续聊天，让我帮你做下一轮策略迭代。"
            if locale == "zh"
            else "✅ <b>Backtest Completed</b>\n"
            f"backtest_id: <code>{escape(backtest_id)}</code>\n"
            "📈 Summary: Annualized 21.4% | Max DD 9.8% | Sharpe 1.57\n"
            "💡 You can continue chatting now for the next strategy iteration."
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="继续聊天优化" if locale == "zh" else "Continue Iteration",
                        callback_data=f"backtest_continue:{backtest_id}",
                    ),
                    InlineKeyboardButton(
                        text="重跑回测" if locale == "zh" else "Rerun Backtest",
                        callback_data=f"backtest_rerun:{backtest_id}",
                    ),
                ]
            ]
        )
        await self._safe_send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _batch_live_trade_open_reminder(
        self,
        *,
        chat_id: str,
        locale: str,
        position_id: str,
    ) -> None:
        text = (
            "🟢 <b>实盘开仓提醒</b>\n"
            f"position_id: <code>{escape(position_id)}</code>\n"
            "symbol: <code>NASDAQ:AAPL</code>\n"
            "方向: <b>LONG</b> | 数量: <code>100</code> | 开仓价: <code>186.25</code>\n"
            "⚠️ 测试提醒，不会触发真实下单。"
            if locale == "zh"
            else "🟢 <b>Live Open-Position Alert</b>\n"
            f"position_id: <code>{escape(position_id)}</code>\n"
            "symbol: <code>NASDAQ:AAPL</code>\n"
            "Side: <b>LONG</b> | Qty: <code>100</code> | Entry: <code>186.25</code>\n"
            "⚠️ Test notification only; no real execution."
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="已知晓" if locale == "zh" else "Acknowledge",
                        callback_data=f"live_open_ack:{position_id}",
                    ),
                    InlineKeyboardButton(
                        text="请求平仓" if locale == "zh" else "Request Close",
                        callback_data=f"live_force_close:{position_id}",
                    ),
                ]
            ]
        )
        await self._safe_send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _batch_live_trade_close_reminder(
        self,
        *,
        chat_id: str,
        locale: str,
        position_id: str,
    ) -> None:
        text = (
            "🔴 <b>实盘平仓回执</b>\n"
            f"position_id: <code>{escape(position_id)}</code>\n"
            "平仓价: <code>188.74</code> | PnL: <code>+1.34%</code> | 原因: 信号反转"
            if locale == "zh"
            else "🔴 <b>Live Close-Position Receipt</b>\n"
            f"position_id: <code>{escape(position_id)}</code>\n"
            "Exit: <code>188.74</code> | PnL: <code>+1.34%</code> | Reason: signal reversal"
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="确认回执" if locale == "zh" else "Confirm Receipt",
                        callback_data=f"live_close_ack:{position_id}",
                    )
                ]
            ]
        )
        await self._safe_send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _batch_market_regime_change(
        self,
        *,
        chat_id: str,
        locale: str,
        regime_id: str,
    ) -> None:
        text = (
            "🌪️ <b>Market Regime 变动提醒</b>\n"
            f"regime_event_id: <code>{escape(regime_id)}</code>\n"
            "旧状态: <code>risk_on</code> → 新状态: <code>risk_off</code>\n"
            "建议: 降低仓位上限并提高止损保护。"
            if locale == "zh"
            else "🌪️ <b>Market Regime Change Alert</b>\n"
            f"regime_event_id: <code>{escape(regime_id)}</code>\n"
            "Transition: <code>risk_on</code> → <code>risk_off</code>\n"
            "Suggestion: reduce position cap and tighten protection."
        )
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        text="已知晓" if locale == "zh" else "Acknowledge",
                        callback_data=f"regime_ack:{regime_id}",
                    ),
                    InlineKeyboardButton(
                        text="请求调整策略" if locale == "zh" else "Request Adjustments",
                        callback_data=f"regime_adjust:{regime_id}",
                    ),
                ]
            ]
        )
        await self._safe_send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _batch_poll(self, *, chat_id: str, locale: str) -> None:
        question = (
            "KYC 回顾：你的风险偏好是？"
            if locale == "zh"
            else "KYC recap: what is your risk tolerance?"
        )
        options = [
            "保守 conservative" if locale == "zh" else "Conservative",
            "中等 moderate" if locale == "zh" else "Moderate",
            "激进 aggressive" if locale == "zh" else "Aggressive",
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
                    "对应 KYC 的 risk_tolerance 字段，用于测试映射。"
                    if locale == "zh"
                    else "Maps to KYC risk_tolerance for test validation."
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
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        instructions = (
            "你是 Minsy 的交易助手。回答需贴合 KYC→Pre-Strategy→Strategy→Deployment 流程，简洁并提示风险。"
            if locale == "zh"
            else "You are Minsy's trading assistant. Keep answers aligned with KYC→Pre-Strategy→Strategy→Deployment flow, concise, and risk-aware."
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
        usage_snapshot = self._extract_response_usage_snapshot(
            response=response,
            model=settings.openai_response_model,
            response_id=response_id,
        )
        return text, response_id, usage_snapshot

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
    def _extract_response_usage_snapshot(
        *,
        response: Any,
        model: str,
        response_id: str | None,
    ) -> dict[str, Any] | None:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None and not isinstance(raw_usage, dict) and hasattr(
            raw_usage,
            "model_dump",
        ):
            try:
                raw_usage = raw_usage.model_dump(mode="json", exclude_none=True)
            except Exception:  # noqa: BLE001
                raw_usage = None

        if not isinstance(raw_usage, dict):
            return None

        return build_turn_usage_snapshot(
            raw_usage=raw_usage,
            model=model,
            response_id=response_id,
            at=datetime.now(UTC),
            pricing=settings.openai_pricing_json,
            cost_tracking_enabled=settings.openai_cost_tracking_enabled,
        )

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
    def _build_event_id(*, prefix: str) -> str:
        normalized = re.sub(r"[^a-z0-9]", "", prefix.lower()) or "evt"
        return datetime.now(UTC).strftime(f"{normalized}%Y%m%d%H%M%S")

    @staticmethod
    def _build_trade_signal_text(*, locale: str, signal_id: str) -> str:
        if locale == "zh":
            return (
                "🚨 <b>Strategy 阶段交易机会</b>\n"
                f"Signal ID: <code>{escape(signal_id)}</code>\n"
                "标的: <code>NASDAQ:AAPL</code>（示例）\n"
                "来源: <b>DSL + 回测筛选后机会</b>\n"
                "动作: 选择 <b>开单</b> / <b>忽略</b>，并通过 WebApp 查看 TradingView 图表。\n"
                "🧪 注：这是功能测试消息，不触发真实交易。"
            )
        return (
            "🚨 <b>Strategy Phase Opportunity</b>\n"
            f"Signal ID: <code>{escape(signal_id)}</code>\n"
            "Symbol: <code>NASDAQ:AAPL</code> (sample)\n"
            "Source: <b>post-DSL and backtest candidate</b>\n"
            "Action: choose <b>Open</b> / <b>Ignore</b>, then inspect the TradingView WebApp chart.\n"
            "🧪 Note: this is a feature-test signal; no real trade execution."
        )

    @staticmethod
    def _build_trade_status_text(*, locale: str, signal_id: str, action: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        if locale == "zh":
            status = "已开单" if action == "open" else "已忽略"
            return (
                "✅ <b>机会状态已更新</b>\n"
                f"Signal ID: <code>{escape(signal_id)}</code>\n"
                f"状态: <b>{status}</b>\n"
                f"时间: <code>{timestamp}</code>"
            )
        status_en = "Opened" if action == "open" else "Ignored"
        return (
            "✅ <b>Signal Status Updated</b>\n"
            f"Signal ID: <code>{escape(signal_id)}</code>\n"
            f"Status: <b>{status_en}</b>\n"
            f"Time: <code>{timestamp}</code>"
        )

    @staticmethod
    def _build_help_text(*, locale: str) -> str:
        if locale == "zh":
            return (
                "🧰 可用测试命令：\n"
                "/reset - 重置 OpenAI 对话上下文\n"
                "/signal - 立即推送一条 Strategy 交易机会\n"
                "/testall - 重放连接后的全部测试批次\n"
                "/postflow - 仅重放 Strategy 后续提醒（backtest/实盘/regime）\n"
                "/help - 查看命令列表\n"
                "\n💬 也可以直接发送任意文本与我对话。"
            )
        return (
            "🧰 Available test commands:\n"
            "/reset - reset OpenAI conversation context\n"
            "/signal - push one Strategy opportunity now\n"
            "/testall - replay all post-connect test batches\n"
            "/postflow - replay post-strategy alerts only (backtest/live/regime)\n"
            "/help - show this command list\n"
            "\n💬 You can also send any text to chat."
        )

    async def _resolve_target_chat_or_report(
        self,
        *,
        source_chat_id: str,
        command: str,
    ) -> str | None:
        try:
            return await self._resolve_test_target_chat_id(chat_id=source_chat_id)
        except ValueError as exc:
            logger.warning("Telegram test command routing failed command=%s error=%s", command, exc)
            await self._safe_send_message(chat_id=source_chat_id, text=str(exc))
            return None

    async def _resolve_test_target_chat_id(self, *, chat_id: str) -> str:
        if not settings.telegram_test_force_target_email_enabled:
            return chat_id

        target_email = settings.telegram_test_target_email.strip().lower()
        if not target_email:
            raise ValueError("TELEGRAM_TEST_TARGET_EMAIL is empty; cannot route test messages.")

        binding = await self.social_service.resolve_connected_telegram_binding_by_email(
            email=target_email,
            require_connected=settings.telegram_test_target_require_connected,
        )
        if binding is None:
            raise ValueError(
                f"Test target {target_email} has no connected Telegram binding. "
                "Testing flow halted by TELEGRAM_TEST_FORCE_TARGET_EMAIL_ENABLED."
            )

        resolved_chat_id = str(binding.external_chat_id).strip()
        if not resolved_chat_id:
            raise ValueError(
                f"Test target {target_email} has empty external_chat_id. "
                "Testing flow halted."
            )

        expected_chat_id = settings.telegram_test_expected_chat_id.strip()
        if expected_chat_id and resolved_chat_id != expected_chat_id:
            raise ValueError(
                "Resolved test target chat_id does not match TELEGRAM_TEST_EXPECTED_CHAT_ID. "
                "Testing flow halted."
            )

        logger.info(
            "Telegram test target resolved test_target_email=%s source_chat_id=%s target_chat_id=%s",
            target_email,
            chat_id,
            resolved_chat_id,
        )
        return resolved_chat_id

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
    base = _resolve_webapp_base_url().rstrip("/")

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


def _resolve_webapp_base_url() -> str:
    """Resolve WebApp base origin and avoid frontend-router hash conflicts."""
    raw = settings.telegram_webapp_base_url.strip()
    if not raw:
        return "https://api.minsyai.com"

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    scheme = parsed.scheme or "https"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "https://api.minsyai.com"

    # When frontend origin is provided, force API origin to avoid SPA hash-router
    # collisions with Telegram tgWebAppData query/hash parameters.
    if host == "app.minsyai.com":
        host = "api.minsyai.com"

    netloc = host
    if isinstance(parsed.port, int):
        netloc = f"{host}:{parsed.port}"
    return f"{scheme}://{netloc}"


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

    title = "Minsy TradingView" if safe_locale == "en" else "Minsy 图表"

    return f"""<!doctype html>
<html lang="{safe_locale}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no" />
    <title>{escape(title)}</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: {"#0f172a" if safe_theme == "dark" else "#ffffff"};
      }}
      .chart-root {{
        width: 100vw;
        height: 100vh;
      }}
      .tradingview-widget-container,
      .tradingview-widget-container__widget {{
        width: 100%;
        height: 100%;
      }}
      .tradingview-widget-copyright {{
        display: none !important;
      }}
    </style>
  </head>
  <body>
    <div
      class="chart-root"
      data-signal-id="{safe_signal}"
      data-symbol="{escape(safe_symbol)}"
      data-interval="{escape(safe_interval)}"
    >
      <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>{json.dumps(widget_config, ensure_ascii=False)}</script>
      </div>
    </div>
    <script>
      const tg = window.Telegram && window.Telegram.WebApp ? window.Telegram.WebApp : null;
      if (tg) {{
        tg.ready();
        tg.expand();
      }}
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
