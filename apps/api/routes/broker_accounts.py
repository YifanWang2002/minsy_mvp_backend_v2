"""Broker account management endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import BrokerAccountResponse
from apps.api.schemas.requests import (
    BrokerAccountCreateRequest,
    BrokerAccountCredentialsUpdateRequest,
)
from apps.api.dependencies import get_db
from packages.infra.providers.trading.alpaca_account_probe import (
    AlpacaAccountProbe,
    AlpacaAccountProbeResult,
)
from packages.infra.providers.trading.credentials import (
    CredentialCipher,
    CredentialEncryptionError,
    credential_key_fingerprint,
)
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.broker_account_audit_log import BrokerAccountAuditLog
from packages.infra.db.models.user import User

router = APIRouter(prefix="/broker-accounts", tags=["broker-accounts"])


def _extract_alpaca_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


def _extract_alpaca_credentials(credentials: dict[str, Any]) -> tuple[str, str]:
    api_key = _extract_alpaca_credential_value(
        credentials,
        "APCA-API-KEY-ID",
        "api_key",
        "key",
    )
    api_secret = _extract_alpaca_credential_value(
        credentials,
        "APCA-API-SECRET-KEY",
        "api_secret",
        "secret",
    )
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_CREDENTIALS_MISSING",
                "message": "Alpaca API key and secret are required for validation.",
            },
        )
    return api_key, api_secret


async def _probe_alpaca_credentials(credentials: dict[str, Any]) -> AlpacaAccountProbeResult:
    api_key, api_secret = _extract_alpaca_credentials(credentials)
    probe = AlpacaAccountProbe()
    try:
        return await probe.probe_credentials(api_key=api_key, api_secret=api_secret)
    finally:
        await probe.aclose()


def _raise_probe_failure(result: AlpacaAccountProbeResult) -> None:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "code": "BROKER_ACCOUNT_VALIDATION_FAILED",
            "message": result.message,
            "metadata": result.metadata,
        },
    )


def _append_audit_log(
    db: AsyncSession,
    *,
    account: BrokerAccount,
    action: str,
    source: str = "api",
    metadata: dict[str, Any] | None = None,
) -> None:
    db.add(
        BrokerAccountAuditLog(
            broker_account_id=account.id,
            user_id=account.user_id,
            action=action,
            source=source,
            metadata_=metadata or {},
        )
    )


def _to_response(account: BrokerAccount) -> BrokerAccountResponse:
    return BrokerAccountResponse(
        broker_account_id=account.id,
        user_id=account.user_id,
        provider=account.provider,
        mode=account.mode,
        status=account.status,
        key_fingerprint=account.key_fingerprint,
        encryption_version=account.encryption_version,
        updated_source=account.updated_source,
        last_validated_at=account.last_validated_at,
        last_validated_status=account.last_validated_status,
        validation_metadata=(
            account.validation_metadata if isinstance(account.validation_metadata, dict) else {}
        ),
        metadata=account.metadata_ if isinstance(account.metadata_, dict) else {},
        created_at=account.created_at,
        updated_at=account.updated_at,
    )


@router.post("", response_model=BrokerAccountResponse, status_code=status.HTTP_201_CREATED)
async def create_broker_account(
    payload: BrokerAccountCreateRequest,
    validate: bool = Query(default=True),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    cipher = CredentialCipher()
    encrypted_credentials = cipher.encrypt(payload.credentials)
    key_fingerprint = credential_key_fingerprint(payload.credentials)
    last_validated_at: datetime | None = None
    last_validated_status: str | None = None
    validation_metadata: dict[str, Any] = {}

    if validate and payload.provider == "alpaca":
        probe_result = await _probe_alpaca_credentials(payload.credentials)
        last_validated_at = datetime.now(UTC)
        last_validated_status = probe_result.status
        validation_metadata = probe_result.metadata
        if not probe_result.ok:
            _raise_probe_failure(probe_result)
    elif not validate:
        last_validated_status = "validation_skipped"
        validation_metadata = {"validate_requested": False}

    account = BrokerAccount(
        user_id=user.id,
        provider=payload.provider,
        mode=payload.mode,
        encrypted_credentials=encrypted_credentials,
        key_fingerprint=key_fingerprint,
        encryption_version=cipher.encryption_version,
        updated_source="api",
        last_validated_at=last_validated_at,
        last_validated_status=last_validated_status,
        validation_metadata=validation_metadata,
        metadata_=payload.metadata,
    )
    db.add(account)
    await db.flush()
    _append_audit_log(
        db,
        account=account,
        action="create",
        metadata={
            "validate_requested": validate,
            "last_validated_status": last_validated_status,
        },
    )
    await db.commit()
    await db.refresh(account)
    return _to_response(account)


@router.get("", response_model=list[BrokerAccountResponse])
async def list_broker_accounts(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[BrokerAccountResponse]:
    rows = (
        await db.scalars(
            select(BrokerAccount)
            .where(BrokerAccount.user_id == user.id)
            .order_by(BrokerAccount.created_at.desc()),
        )
    ).all()
    return [_to_response(row) for row in rows]


@router.get("/{broker_account_id}", response_model=BrokerAccountResponse)
async def get_broker_account(
    broker_account_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user.id,
        )
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BROKER_ACCOUNT_NOT_FOUND",
                "message": "Broker account not found.",
            },
        )
    return _to_response(account)


@router.patch("/{broker_account_id}/credentials", response_model=BrokerAccountResponse)
async def rotate_broker_account_credentials(
    broker_account_id: UUID,
    payload: BrokerAccountCredentialsUpdateRequest,
    validate: bool = Query(default=True),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user.id,
        )
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BROKER_ACCOUNT_NOT_FOUND",
                "message": "Broker account not found.",
            },
        )

    last_validated_at: datetime | None = None
    last_validated_status: str | None = None
    validation_metadata: dict[str, Any] = {}
    if validate and account.provider == "alpaca":
        probe_result = await _probe_alpaca_credentials(payload.credentials)
        last_validated_at = datetime.now(UTC)
        last_validated_status = probe_result.status
        validation_metadata = probe_result.metadata
        if not probe_result.ok:
            _raise_probe_failure(probe_result)
    elif not validate:
        last_validated_status = "validation_skipped"
        validation_metadata = {"validate_requested": False}

    cipher = CredentialCipher()
    account.encrypted_credentials = cipher.encrypt(payload.credentials)
    account.key_fingerprint = credential_key_fingerprint(payload.credentials)
    account.encryption_version = cipher.encryption_version
    account.updated_source = "api"
    account.last_validated_at = last_validated_at
    account.last_validated_status = last_validated_status
    account.validation_metadata = validation_metadata
    account.status = "active"
    account.last_error = None
    _append_audit_log(
        db,
        account=account,
        action="update",
        metadata={
            "validate_requested": validate,
            "last_validated_status": last_validated_status,
        },
    )
    await db.commit()
    await db.refresh(account)
    return _to_response(account)


@router.post("/{broker_account_id}/validate", response_model=BrokerAccountResponse)
async def validate_broker_account(
    broker_account_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user.id,
        )
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BROKER_ACCOUNT_NOT_FOUND",
                "message": "Broker account not found.",
            },
        )
    if account.provider != "alpaca":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_PROVIDER_NOT_SUPPORTED",
                "message": "Only Alpaca broker accounts support active key validation.",
            },
        )

    cipher = CredentialCipher()
    try:
        credentials = cipher.decrypt(account.encrypted_credentials)
    except CredentialEncryptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_CREDENTIALS_DECRYPT_FAILED",
                "message": "Failed to decrypt broker account credentials.",
            },
        ) from exc

    probe_result = await _probe_alpaca_credentials(credentials)
    account.last_validated_at = datetime.now(UTC)
    account.last_validated_status = probe_result.status
    account.validation_metadata = probe_result.metadata
    if probe_result.ok:
        account.status = "active"
        account.last_error = None
    else:
        account.status = "error"
        account.last_error = probe_result.message
    _append_audit_log(
        db,
        account=account,
        action="validate",
        metadata={
            "validation_passed": probe_result.ok,
            "last_validated_status": probe_result.status,
            "validation_metadata": probe_result.metadata,
        },
    )

    await db.commit()
    await db.refresh(account)

    if not probe_result.ok:
        _raise_probe_failure(probe_result)

    return _to_response(account)


@router.post("/{broker_account_id}/deactivate", response_model=BrokerAccountResponse)
async def deactivate_broker_account(
    broker_account_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await db.scalar(
        select(BrokerAccount).where(
            BrokerAccount.id == broker_account_id,
            BrokerAccount.user_id == user.id,
        )
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BROKER_ACCOUNT_NOT_FOUND",
                "message": "Broker account not found.",
            },
        )

    account.status = "inactive"
    account.updated_source = "api"
    _append_audit_log(
        db,
        account=account,
        action="deactivate",
        metadata={"reason": "manual_deactivate"},
    )
    await db.commit()
    await db.refresh(account)
    return _to_response(account)
