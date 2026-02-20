"""Domain services for social connector state and persistence."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.social_connector import (
    SocialConnectorActivity,
    SocialConnectorBinding,
    SocialConnectorLinkIntent,
)
from src.models.user_settings import UserSetting

SUPPORTED_CONNECTOR_PROVIDERS: tuple[str, ...] = (
    "telegram",
    "discord",
    "slack",
    "whatsapp",
)
CONNECTABLE_CONNECTOR_PROVIDERS: frozenset[str] = frozenset({"telegram"})


@dataclass(frozen=True, slots=True)
class ConnectorView:
    provider: str
    status: str
    connected_account: str | None
    connected_at: datetime | None
    supports_connect: bool


@dataclass(frozen=True, slots=True)
class ConnectLinkResult:
    provider: str
    connect_url: str
    expires_at: datetime


class SocialConnectorService:
    """Social connector operations backed by PostgreSQL tables."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_connectors(self, *, user_id: UUID) -> list[ConnectorView]:
        bindings = (
            await self.db.scalars(
                select(SocialConnectorBinding).where(SocialConnectorBinding.user_id == user_id),
            )
        ).all()
        lookup = {binding.provider: binding for binding in bindings}
        output: list[ConnectorView] = []
        for provider in SUPPORTED_CONNECTOR_PROVIDERS:
            binding = lookup.get(provider)
            if binding is None:
                output.append(
                    ConnectorView(
                        provider=provider,
                        status="disconnected",
                        connected_account=None,
                        connected_at=None,
                        supports_connect=provider in CONNECTABLE_CONNECTOR_PROVIDERS,
                    )
                )
                continue

            is_connected = binding.status == "connected"
            output.append(
                ConnectorView(
                    provider=provider,
                    status=binding.status,
                    connected_account=self._format_account(binding) if is_connected else None,
                    connected_at=binding.bound_at if is_connected else None,
                    supports_connect=provider in CONNECTABLE_CONNECTOR_PROVIDERS,
                )
            )
        return output

    async def create_telegram_connect_link(
        self,
        *,
        user_id: UUID,
        locale: str = "en",
    ) -> ConnectLinkResult:
        self._assert_telegram_ready()
        token = self._generate_link_token()
        expires_at = datetime.now(UTC) + timedelta(seconds=settings.telegram_connect_ttl_seconds)
        intent = SocialConnectorLinkIntent(
            user_id=user_id,
            provider="telegram",
            token_hash=self._hash_token(token),
            expires_at=expires_at,
            consumed_at=None,
            metadata_={"locale": self.normalize_locale(locale)},
        )
        self.db.add(intent)
        await self.db.flush()

        username = self._telegram_bot_username()
        connect_url = f"https://t.me/{username}?start={token}"
        return ConnectLinkResult(
            provider="telegram",
            connect_url=connect_url,
            expires_at=expires_at,
        )

    async def consume_telegram_link_intent(self, *, raw_token: str) -> SocialConnectorLinkIntent | None:
        token_hash = self._hash_token(raw_token)
        now = datetime.now(UTC)
        intent = await self.db.scalar(
            select(SocialConnectorLinkIntent).where(
                SocialConnectorLinkIntent.provider == "telegram",
                SocialConnectorLinkIntent.token_hash == token_hash,
                SocialConnectorLinkIntent.consumed_at.is_(None),
                SocialConnectorLinkIntent.expires_at >= now,
            )
        )
        if intent is None:
            return None
        intent.consumed_at = now
        return intent

    async def upsert_telegram_binding(
        self,
        *,
        user_id: UUID,
        telegram_chat_id: str,
        telegram_user_id: str,
        telegram_username: str | None,
        locale: str,
    ) -> SocialConnectorBinding:
        now = datetime.now(UTC)
        existing_by_chat = await self.db.scalar(
            select(SocialConnectorBinding).where(
                SocialConnectorBinding.provider == "telegram",
                SocialConnectorBinding.external_chat_id == telegram_chat_id,
                SocialConnectorBinding.status == "connected",
            )
        )
        if existing_by_chat is not None and existing_by_chat.user_id != user_id:
            raise ValueError("This Telegram chat is already linked to another account.")

        binding = await self.db.scalar(
            select(SocialConnectorBinding).where(
                SocialConnectorBinding.provider == "telegram",
                SocialConnectorBinding.user_id == user_id,
            )
        )
        if binding is None:
            binding = SocialConnectorBinding(
                user_id=user_id,
                provider="telegram",
                external_user_id=telegram_user_id,
                external_chat_id=telegram_chat_id,
                external_username=self._normalize_username(telegram_username),
                status="connected",
                bound_at=now,
                metadata_={"locale": self.normalize_locale(locale)},
            )
            self.db.add(binding)
            await self.db.flush()
            return binding

        metadata = dict(binding.metadata_ or {})
        metadata["locale"] = self.normalize_locale(locale)
        binding.external_user_id = telegram_user_id
        binding.external_chat_id = telegram_chat_id
        binding.external_username = self._normalize_username(telegram_username)
        binding.status = "connected"
        binding.bound_at = now
        binding.metadata_ = metadata
        return binding

    async def disconnect_telegram(self, *, user_id: UUID) -> bool:
        binding = await self.db.scalar(
            select(SocialConnectorBinding).where(
                SocialConnectorBinding.provider == "telegram",
                SocialConnectorBinding.user_id == user_id,
            )
        )
        if binding is None:
            return False
        metadata = dict(binding.metadata_ or {})
        metadata["disconnected_at"] = datetime.now(UTC).isoformat()
        binding.status = "disconnected"
        binding.metadata_ = metadata
        return True

    async def get_telegram_binding_for_chat(
        self,
        *,
        telegram_chat_id: str,
        require_connected: bool = True,
    ) -> SocialConnectorBinding | None:
        filters = [
            SocialConnectorBinding.provider == "telegram",
            SocialConnectorBinding.external_chat_id == telegram_chat_id,
        ]
        if require_connected:
            filters.append(SocialConnectorBinding.status == "connected")
        return await self.db.scalar(select(SocialConnectorBinding).where(*filters))

    async def get_telegram_binding_for_external_user(
        self,
        *,
        telegram_user_id: str,
        require_connected: bool = True,
    ) -> SocialConnectorBinding | None:
        filters = [
            SocialConnectorBinding.provider == "telegram",
            SocialConnectorBinding.external_user_id == telegram_user_id,
        ]
        if require_connected:
            filters.append(SocialConnectorBinding.status == "connected")
        return await self.db.scalar(select(SocialConnectorBinding).where(*filters))

    async def list_telegram_activities(
        self,
        *,
        user_id: UUID,
        limit: int = 20,
    ) -> list[SocialConnectorActivity]:
        return (
            await self.db.scalars(
                select(SocialConnectorActivity)
                .where(
                    SocialConnectorActivity.user_id == user_id,
                    SocialConnectorActivity.provider == "telegram",
                )
                .order_by(SocialConnectorActivity.created_at.desc())
                .limit(limit),
            )
        ).all()

    async def record_telegram_activity(
        self,
        *,
        user_id: UUID,
        event_type: str,
        choice_value: str | None,
        message_text: str | None,
        external_update_id: int | None,
        payload: dict,
    ) -> SocialConnectorActivity:
        if external_update_id is not None:
            existing = await self.db.scalar(
                select(SocialConnectorActivity).where(
                    SocialConnectorActivity.provider == "telegram",
                    SocialConnectorActivity.external_update_id == external_update_id,
                )
            )
            if existing is not None:
                return existing

        activity = SocialConnectorActivity(
            user_id=user_id,
            provider="telegram",
            event_type=event_type,
            choice_value=choice_value,
            message_text=message_text,
            external_update_id=external_update_id,
            payload=dict(payload or {}),
        )
        self.db.add(activity)
        await self.db.flush()
        return activity

    @staticmethod
    def normalize_locale(raw_locale: str | None) -> str:
        value = (raw_locale or "").strip().lower().replace("_", "-")
        if value.startswith("zh"):
            return "zh"
        return "en"

    async def resolve_user_locale(
        self,
        *,
        user_id: UUID,
        fallback_locale: str | None = None,
    ) -> str:
        raw_locale = await self.db.scalar(
            select(UserSetting.locale).where(UserSetting.user_id == user_id)
        )
        if isinstance(raw_locale, str) and raw_locale.strip():
            return self.normalize_locale(raw_locale)
        return self.normalize_locale(fallback_locale)

    @staticmethod
    def _generate_link_token() -> str:
        return secrets.token_urlsafe(24)

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_username(raw_username: str | None) -> str | None:
        if raw_username is None:
            return None
        value = raw_username.strip()
        if not value:
            return None
        return value.lstrip("@")

    @staticmethod
    def _format_account(binding: SocialConnectorBinding) -> str:
        username = (binding.external_username or "").strip()
        if username:
            return f"@{username.lstrip('@')}"
        return binding.external_chat_id

    @staticmethod
    def _telegram_bot_username() -> str:
        username = settings.telegram_bot_username.strip().lstrip("@")
        if username:
            return username
        raise RuntimeError("TELEGRAM_BOT_USERNAME is not configured.")

    @staticmethod
    def _assert_telegram_ready() -> None:
        if not settings.telegram_enabled:
            raise RuntimeError("Telegram connector is disabled.")
        if not settings.telegram_bot_token.strip():
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
        if not settings.telegram_bot_username.strip():
            raise RuntimeError("TELEGRAM_BOT_USERNAME is not configured.")
