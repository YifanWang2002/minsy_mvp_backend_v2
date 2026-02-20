"""User domain models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base

if TYPE_CHECKING:
    from src.models.backtest import BacktestJob
    from src.models.deployment import Deployment
    from src.models.session import Session
    from src.models.social_connector import (
        SocialConnectorActivity,
        SocialConnectorBinding,
        SocialConnectorLinkIntent,
    )
    from src.models.strategy import Strategy
    from src.models.user_settings import UserSetting


class User(Base):
    """Application user."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )

    settings: Mapped[UserSetting | None] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )
    profiles: Mapped[list[UserProfile]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list[Session]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    strategies: Mapped[list[Strategy]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    backtest_jobs: Mapped[list[BacktestJob]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    deployments: Mapped[list[Deployment]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    social_connector_bindings: Mapped[list[SocialConnectorBinding]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    social_connector_link_intents: Mapped[list[SocialConnectorLinkIntent]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    social_connector_activities: Mapped[list[SocialConnectorActivity]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class UserProfile(Base):
    """KYC and preference profile attached to user."""

    __tablename__ = "user_profiles"
    __table_args__ = (
        CheckConstraint(
            "kyc_status IN ('incomplete', 'complete')",
            name="ck_user_profiles_kyc_status",
        ),
        CheckConstraint(
            "trading_years_bucket IS NULL OR "
            "trading_years_bucket IN ('years_0_1','years_1_3','years_3_5','years_5_plus')",
            name="ck_user_profiles_trading_years_bucket",
        ),
        CheckConstraint(
            "risk_tolerance IS NULL OR "
            "risk_tolerance IN ('conservative','moderate','aggressive','very_aggressive')",
            name="ck_user_profiles_risk_tolerance_values",
        ),
        CheckConstraint(
            "return_expectation IS NULL OR "
            "return_expectation IN "
            "('capital_preservation','balanced_growth','growth','high_growth')",
            name="ck_user_profiles_return_expectation_values",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trading_years_bucket: Mapped[str | None] = mapped_column(String(20), nullable=True)
    risk_tolerance: Mapped[str | None] = mapped_column(String(20), nullable=True)
    return_expectation: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    kyc_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="incomplete",
        server_default="incomplete",
    )

    user: Mapped[User] = relationship(back_populates="profiles")
