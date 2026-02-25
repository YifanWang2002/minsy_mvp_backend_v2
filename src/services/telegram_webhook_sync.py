"""Startup-time Telegram webhook synchronization."""

from __future__ import annotations

import json

import httpx

from src.config import settings
from src.util.logger import logger


async def sync_telegram_webhook_on_startup() -> None:
    """Sync Telegram webhook to configured public API URL when enabled."""
    if not settings.telegram_enabled:
        return
    if not settings.telegram_webhook_auto_sync_enabled:
        return

    target_url = settings.effective_telegram_webhook_url.strip()
    if not target_url:
        logger.warning(
            "Telegram webhook auto-sync is enabled but no webhook URL is configured. "
            "Set TELEGRAM_WEBHOOK_URL or API_PUBLIC_BASE_URL."
        )
        return

    token = settings.telegram_bot_token.strip()
    if not token:
        logger.warning("Telegram webhook auto-sync skipped: TELEGRAM_BOT_TOKEN is empty.")
        return

    api_base = f"https://api.telegram.org/bot{token}"
    secret = settings.telegram_webhook_secret_token.strip()
    payload: dict[str, object] = {
        "url": target_url,
        "drop_pending_updates": False,
        "allowed_updates": json.dumps(["message", "callback_query", "inline_query"]),
    }
    if secret:
        payload["secret_token"] = secret

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            info_response = await client.get(f"{api_base}/getWebhookInfo")
            info_response.raise_for_status()
            info_payload = info_response.json()
            current_url = str((info_payload.get("result") or {}).get("url") or "").strip()
            if current_url == target_url:
                logger.info("Telegram webhook already up-to-date: %s", target_url)
                return

            set_response = await client.post(f"{api_base}/setWebhook", data=payload)
            set_response.raise_for_status()
            set_payload = set_response.json()
            if not bool(set_payload.get("ok")):
                logger.warning("Telegram setWebhook returned non-ok payload: %s", set_payload)
                return

            logger.info(
                "Telegram webhook synced old_url=%s new_url=%s",
                current_url or "<empty>",
                target_url,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram webhook auto-sync failed: %s", exc)

