"""Ledger entries for internal sandbox broker account events."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, DateTime, Index, Numeric, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from packages.infra.db.models.base import Base


class SandboxLedgerEntry(Base):
    """Immutable ledger row for one sandbox account balance-changing event."""

    __tablename__ = "sandbox_ledger_entries"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('order_fill', 'state_init', 'manual_adjustment')",
            name="ck_sandbox_ledger_entries_event_type",
        ),
        CheckConstraint(
            "asset_class IN ('us_equity', 'crypto', 'forex', 'futures', 'unknown')",
            name="ck_sandbox_ledger_entries_asset_class",
        ),
        CheckConstraint("side IN ('buy', 'sell')", name="ck_sandbox_ledger_entries_side"),
        CheckConstraint("qty >= 0", name="ck_sandbox_ledger_entries_qty_non_negative"),
        CheckConstraint("fill_price >= 0", name="ck_sandbox_ledger_entries_fill_price_non_negative"),
        CheckConstraint("notional >= 0", name="ck_sandbox_ledger_entries_notional_non_negative"),
        CheckConstraint("fee >= 0", name="ck_sandbox_ledger_entries_fee_non_negative"),
        CheckConstraint(
            "fee_bps >= 0",
            name="ck_sandbox_ledger_entries_fee_bps_non_negative",
        ),
        CheckConstraint(
            "slippage_bps >= 0",
            name="ck_sandbox_ledger_entries_slippage_bps_non_negative",
        ),
        Index("ix_sandbox_ledger_entries_account_uid", "account_uid"),
        Index("ix_sandbox_ledger_entries_provider_order_id", "provider_order_id"),
        Index("ix_sandbox_ledger_entries_happened_at", "happened_at"),
        Index("ix_sandbox_ledger_entries_account_happened", "account_uid", "happened_at"),
    )

    account_uid: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_order_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    client_order_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    event_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="order_fill",
        server_default="order_fill",
    )
    symbol: Mapped[str] = mapped_column(String(80), nullable=False, default="", server_default="")
    asset_class: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="unknown",
        server_default="unknown",
    )
    side: Mapped[str] = mapped_column(String(10), nullable=False, default="buy", server_default="buy")
    qty: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    fill_price: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    notional: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    fee: Mapped[Decimal] = mapped_column(
        Numeric(20, 8),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    fee_bps: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    slippage_bps: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    cash_before: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    cash_after: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    position_qty_before: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    position_qty_after: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    avg_entry_before: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    avg_entry_after: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    realized_pnl_before: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    realized_pnl_after: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    happened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
