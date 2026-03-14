"""Locale helpers shared by API routes and orchestrator prompts."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.infra.db.models.user_settings import DEFAULT_LOCALE, UserSetting

_SUPPORTED_LOCALES = {"en", "zh"}


def normalize_locale(raw: str | None, *, default: str = DEFAULT_LOCALE) -> str:
    """Normalize raw locale/language strings to supported locales."""
    candidate = str(raw or "").strip().lower().replace("_", "-")
    if candidate.startswith("zh"):
        return "zh"
    if candidate.startswith("en"):
        return "en"
    if default in _SUPPORTED_LOCALES:
        return default
    return "en"


def is_zh_locale(raw: str | None) -> bool:
    """Return True when locale should use Chinese copy."""
    return normalize_locale(raw, default="en") == "zh"


async def resolve_user_locale(
    db: AsyncSession,
    *,
    user_id: UUID,
    fallback: str | None = None,
) -> str:
    """Resolve locale from user settings with request fallback."""
    locale_value = await db.scalar(
        select(UserSetting.locale).where(UserSetting.user_id == user_id)
    )
    if isinstance(locale_value, str) and locale_value.strip():
        return normalize_locale(locale_value)
    return normalize_locale(fallback)
