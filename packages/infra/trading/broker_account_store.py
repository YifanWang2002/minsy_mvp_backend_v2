"""Shared DB operations for broker account access across apps."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from packages.domain.trading.broker_capability_policy import build_broker_capabilities
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.broker_account_audit_log import BrokerAccountAuditLog
from packages.infra.providers.trading.credentials import (
    CredentialCipher,
    credential_key_fingerprint,
)

_SANDBOX_VALIDATION_STATUS = "sandbox_ready"
_SANDBOX_VALIDATION_METADATA = {
    "provider": "sandbox",
    "mode": "internal",
    "is_sandbox": True,
    "market_data_source": "alpaca",
}


async def list_user_broker_accounts(
    db: AsyncSession,
    *,
    user_id: UUID,
    active_only: bool = False,
) -> list[BrokerAccount]:
    stmt = (
        select(BrokerAccount)
        .where(BrokerAccount.user_id == user_id)
        .order_by(
            BrokerAccount.is_default.desc(),
            BrokerAccount.updated_at.desc(),
            BrokerAccount.created_at.desc(),
        )
    )
    if active_only:
        stmt = stmt.where(BrokerAccount.status == "active")
    return list((await db.scalars(stmt)).all())


async def find_first_active_default_paper_broker_account_id(
    db: AsyncSession,
    *,
    user_id: UUID,
) -> UUID | None:
    return await db.scalar(
        select(BrokerAccount.id)
        .where(
            BrokerAccount.user_id == user_id,
            BrokerAccount.mode == "paper",
            BrokerAccount.status == "active",
        )
        .order_by(
            BrokerAccount.is_default.desc(),
            BrokerAccount.updated_at.desc(),
            BrokerAccount.created_at.desc(),
        )
        .limit(1)
    )


async def find_builtin_sandbox_account(
    db: AsyncSession,
    *,
    user_id: UUID,
    active_only: bool = False,
) -> BrokerAccount | None:
    stmt = (
        select(BrokerAccount)
        .where(
            BrokerAccount.user_id == user_id,
            BrokerAccount.provider == "sandbox",
            BrokerAccount.mode == "paper",
        )
        .order_by(
            BrokerAccount.is_default.desc(),
            BrokerAccount.updated_at.desc(),
            BrokerAccount.created_at.desc(),
        )
    )
    if active_only:
        stmt = stmt.where(BrokerAccount.status == "active")
    return await db.scalar(stmt)


async def ensure_builtin_sandbox_account(
    db: AsyncSession,
    *,
    user_id: UUID,
    source: str = "builtin_sandbox",
    metadata: dict[str, Any] | None = None,
) -> BrokerAccount:
    existing = await find_builtin_sandbox_account(
        db,
        user_id=user_id,
        active_only=False,
    )
    cipher = CredentialCipher()
    credentials: dict[str, Any] = {}
    encrypted_credentials = cipher.encrypt(credentials)
    key_fingerprint = credential_key_fingerprint(credentials)
    now = datetime.now(UTC)
    capabilities = build_broker_capabilities(
        provider="sandbox",
        exchange_id="sandbox",
        is_sandbox=True,
    )
    metadata_patch = metadata if isinstance(metadata, dict) else {}
    existing_metadata = (
        dict(existing.metadata_)
        if existing is not None and isinstance(existing.metadata_, dict)
        else {}
    )
    merged_metadata = dict(existing_metadata)
    merged_metadata["source"] = source
    for key, value in metadata_patch.items():
        if value is None:
            continue
        merged_metadata[str(key)] = value

    if existing is None:
        account = BrokerAccount(
            user_id=user_id,
            provider="sandbox",
            exchange_id="sandbox",
            account_uid=f"sandbox-{uuid4().hex[:16]}",
            mode="paper",
            encrypted_credentials=encrypted_credentials,
            key_fingerprint=key_fingerprint,
            encryption_version=cipher.encryption_version,
            updated_source="api",
            is_default=False,
            is_sandbox=True,
            last_validated_at=now,
            last_validated_status=_SANDBOX_VALIDATION_STATUS,
            last_validation_error_code=None,
            capabilities=capabilities,
            validation_metadata=dict(_SANDBOX_VALIDATION_METADATA),
            metadata_=merged_metadata,
            status="active",
            last_error=None,
        )
        db.add(account)
        await db.flush()
        action = "create"
    else:
        account = existing
        account.exchange_id = "sandbox"
        account.encrypted_credentials = encrypted_credentials
        account.key_fingerprint = key_fingerprint
        account.encryption_version = cipher.encryption_version
        account.updated_source = "api"
        account.is_sandbox = True
        account.last_validated_at = now
        account.last_validated_status = _SANDBOX_VALIDATION_STATUS
        account.last_validation_error_code = None
        account.capabilities = capabilities
        account.validation_metadata = dict(_SANDBOX_VALIDATION_METADATA)
        account.metadata_ = merged_metadata
        account.status = "active"
        account.last_error = None
        action = "update"

    await _rebalance_default_accounts(
        db,
        user_id=user_id,
        mode="paper",
    )
    _append_audit_log(
        db,
        account=account,
        action=action,
        metadata={
            "source": source,
            "last_validated_status": _SANDBOX_VALIDATION_STATUS,
            "upserted_from_create": existing is not None,
        },
    )
    await db.commit()
    await db.refresh(account)
    return account


async def deactivate_builtin_sandbox_account(
    db: AsyncSession,
    *,
    user_id: UUID,
) -> BrokerAccount | None:
    account = await find_builtin_sandbox_account(
        db,
        user_id=user_id,
        active_only=True,
    )
    if account is None:
        return None

    account.status = "inactive"
    account.is_default = False
    account.updated_source = "api"
    await _rebalance_default_accounts(
        db,
        user_id=user_id,
        mode=account.mode,
    )
    _append_audit_log(
        db,
        account=account,
        action="deactivate",
        metadata={"reason": "builtin_sandbox_deactivate"},
    )
    await db.commit()
    await db.refresh(account)
    return account


async def _rebalance_default_accounts(
    db: AsyncSession,
    *,
    user_id: UUID,
    mode: str,
) -> None:
    rows = (
        await db.scalars(
            select(BrokerAccount)
            .where(
                BrokerAccount.user_id == user_id,
                BrokerAccount.mode == mode,
            )
            .order_by(
                BrokerAccount.is_default.desc(),
                BrokerAccount.last_validated_at.desc().nullslast(),
                BrokerAccount.updated_at.desc(),
                BrokerAccount.created_at.desc(),
            )
        )
    ).all()
    active_rows = [
        row for row in rows if str(row.status).strip().lower() == "active"
    ]
    await db.execute(
        update(BrokerAccount)
        .where(
            BrokerAccount.user_id == user_id,
            BrokerAccount.mode == mode,
        )
        .values(is_default=False)
    )
    if not active_rows:
        return
    await db.execute(
        update(BrokerAccount)
        .where(BrokerAccount.id == active_rows[0].id)
        .values(is_default=True)
    )


def _append_audit_log(
    db: AsyncSession,
    *,
    account: BrokerAccount,
    action: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    db.add(
        BrokerAccountAuditLog(
            broker_account_id=account.id,
            user_id=account.user_id,
            action=action,
            source="api",
            metadata_=metadata or {},
        )
    )
