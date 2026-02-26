"""Account-level UI preference model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.infra.db.models.base import Base

if TYPE_CHECKING:
    from packages.infra.db.models.user import User

THEME_MODE_VALUES = ("light", "dark", "system")
LOCALE_VALUES = ("en", "zh")
FONT_SCALE_VALUES = ("small", "default", "large")

DEFAULT_THEME_MODE = "system"
DEFAULT_LOCALE = "en"
DEFAULT_FONT_SCALE = "default"


class UserSetting(Base):
    """User preference row (one row per user)."""

    __tablename__ = "user_settings"
    __table_args__ = (
        CheckConstraint(
            "theme_mode IN ('light', 'dark', 'system')",
            name="ck_user_settings_theme_mode",
        ),
        CheckConstraint(
            "locale IN ('en', 'zh')",
            name="ck_user_settings_locale",
        ),
        CheckConstraint(
            "font_scale IN ('small', 'default', 'large')",
            name="ck_user_settings_font_scale",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    theme_mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DEFAULT_THEME_MODE,
        server_default=DEFAULT_THEME_MODE,
    )
    locale: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        default=DEFAULT_LOCALE,
        server_default=DEFAULT_LOCALE,
    )
    font_scale: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DEFAULT_FONT_SCALE,
        server_default=DEFAULT_FONT_SCALE,
    )

    user: Mapped[User] = relationship(back_populates="settings")
