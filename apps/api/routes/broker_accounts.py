"""Broker account management endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.dependencies import get_db
from apps.api.middleware.auth import get_current_user
from apps.api.schemas.events import BrokerAccountResponse, CcxtExchangeInfo
from apps.api.schemas.requests import (
    BrokerAccountCreateRequest,
    BrokerAccountCredentialsUpdateRequest,
    BuiltinSandboxBrokerAccountRequest,
)
from packages.domain.trading.services.broker_validation_service import (
    BrokerCredentialValidationResult,
    BrokerValidationService,
)
from packages.domain.trading.broker_capability_policy import (
    build_broker_capabilities,
)
from packages.domain.trading.services.ccxt_exchange_catalog import (
    list_supported_ccxt_exchanges,
)
from packages.infra.db.models.broker_account import BrokerAccount
from packages.infra.db.models.broker_account_audit_log import BrokerAccountAuditLog
from packages.infra.db.models.user import User
from packages.infra.trading.broker_account_store import (
    deactivate_builtin_sandbox_account as store_deactivate_builtin_sandbox_account,
)
from packages.infra.trading.broker_account_store import (
    ensure_builtin_sandbox_account as store_ensure_builtin_sandbox_account,
)
from packages.infra.providers.trading.credentials import (
    CredentialCipher,
    CredentialEncryptionError,
    credential_key_fingerprint,
)

router = APIRouter(prefix="/broker-accounts", tags=["broker-accounts"])
_broker_validation_service = BrokerValidationService()


def _raise_validation_failure(result: BrokerCredentialValidationResult) -> None:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "code": "BROKER_ACCOUNT_VALIDATION_FAILED",
            "message": result.message,
            "validation_status": result.status,
            "metadata": result.metadata,
        },
    )


def _raise_integrity_error(exc: IntegrityError) -> None:
    detail = str(getattr(exc, "orig", exc)).lower()
    if "uq_broker_accounts_user_provider_exchange_account_uid_active" in detail:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BROKER_ACCOUNT_IDENTITY_CONFLICT",
                "message": (
                    "An active broker account with the same provider/exchange/account identity "
                    "already exists for this user."
                ),
            },
        ) from exc
    if "uq_broker_accounts_user_mode_default_active" in detail:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "BROKER_ACCOUNT_DEFAULT_CONFLICT",
                "message": "Only one active default broker account is allowed per mode.",
            },
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "BROKER_ACCOUNT_CONSTRAINT_VIOLATION",
            "message": "Broker account update violates database constraints.",
        },
    ) from exc


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


async def _validate_provider_credentials(
    *,
    provider: str,
    credentials: dict[str, Any],
) -> BrokerCredentialValidationResult:
    return await _broker_validation_service.validate_credentials(
        provider=provider,
        credentials=credentials,
    )


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


def _resolve_exchange_id(
    *,
    provider: str,
    payload_exchange_id: str | None,
    credentials: dict[str, Any],
) -> str:
    provider_key = provider.strip().lower()
    if payload_exchange_id is not None and payload_exchange_id.strip():
        return payload_exchange_id.strip().lower()
    if provider_key == "ccxt":
        return _extract_credential_value(
            credentials, "exchange_id", "exchange", "name"
        ).lower()
    if provider_key == "alpaca":
        return "alpaca"
    if provider_key == "sandbox":
        return "sandbox"
    return ""


def _resolve_is_sandbox(
    *,
    provider: str,
    exchange_id: str,
    payload_is_sandbox: bool | None,
    credentials: dict[str, Any],
    validation_metadata: dict[str, Any],
) -> bool:
    if payload_is_sandbox is not None:
        return bool(payload_is_sandbox)
    provider_key = provider.strip().lower()
    if provider_key == "sandbox":
        return True
    sandbox_raw = credentials.get("sandbox")
    if sandbox_raw is not None:
        return str(sandbox_raw).strip().lower() in {"1", "true", "yes", "on"}
    metadata_sandbox = validation_metadata.get("sandbox")
    if metadata_sandbox is not None:
        return str(metadata_sandbox).strip().lower() in {"1", "true", "yes", "on"}
    return provider_key == "ccxt" and exchange_id == "okx"


def _resolve_account_uid(
    *,
    provider: str,
    payload_account_uid: str | None,
    credentials: dict[str, Any],
    validation_metadata: dict[str, Any],
    key_fingerprint: str | None,
    existing_account_uid: str | None = None,
) -> str:
    provider_key = provider.strip().lower()
    if payload_account_uid is not None and payload_account_uid.strip():
        return payload_account_uid.strip()
    if existing_account_uid is not None and existing_account_uid.strip():
        return existing_account_uid.strip()

    credential_uid = _extract_credential_value(
        credentials, "account_uid", "account_id", "uid"
    )
    if credential_uid:
        return credential_uid

    if provider_key == "sandbox":
        return f"sandbox-{uuid4().hex[:16]}"
    if provider_key == "alpaca":
        probe_uid = validation_metadata.get("paper_account_id")
        if isinstance(probe_uid, str) and probe_uid.strip():
            return probe_uid.strip()
    if key_fingerprint:
        return key_fingerprint
    return uuid4().hex


def _resolve_last_validation_error_code(
    *,
    validation_result: BrokerCredentialValidationResult | None,
) -> str | None:
    if validation_result is None or validation_result.ok:
        return None
    metadata = (
        validation_result.metadata
        if isinstance(validation_result.metadata, dict)
        else {}
    )
    error_type = metadata.get("error_type")
    if isinstance(error_type, str) and error_type.strip():
        return error_type.strip()[:64]
    status_code = str(validation_result.status or "").strip()
    return status_code[:64] if status_code else "validation_failed"


async def _rebalance_default_accounts(
    db: AsyncSession,
    *,
    user_id: UUID,
    mode: str,
    preferred_account_id: UUID | None = None,
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
                BrokerAccount.created_at.desc(),
            ),
        )
    ).all()
    active_rows = [row for row in rows if str(row.status).strip().lower() == "active"]
    if not active_rows:
        await db.execute(
            update(BrokerAccount)
            .where(
                BrokerAccount.user_id == user_id,
                BrokerAccount.mode == mode,
            )
            .values(is_default=False)
        )
        return

    target = None
    if preferred_account_id is not None:
        for row in active_rows:
            if row.id == preferred_account_id:
                target = row
                break
    if target is None:
        for row in active_rows:
            if row.is_default:
                target = row
                break
    if target is None:
        target = active_rows[0]
    await db.execute(
        update(BrokerAccount)
        .where(
            BrokerAccount.user_id == user_id,
            BrokerAccount.mode == mode,
        )
        .values(is_default=False)
    )
    await db.execute(
        update(BrokerAccount)
        .where(BrokerAccount.id == target.id)
        .values(is_default=True)
    )


def _logical_group_key(provider: str, exchange_id: str) -> tuple[str, str]:
    provider_key = str(provider).strip().lower()
    if provider_key == "ccxt":
        return provider_key, str(exchange_id).strip().lower()
    return provider_key, ""


async def _find_reusable_broker_account(
    db: AsyncSession,
    *,
    user_id: UUID,
    provider: str,
    mode: str,
    exchange_id: str,
) -> BrokerAccount | None:
    provider_key, exchange_key = _logical_group_key(provider, exchange_id)
    stmt = select(BrokerAccount).where(
        BrokerAccount.user_id == user_id,
        BrokerAccount.provider == provider_key,
        BrokerAccount.mode == mode,
    )
    if exchange_key:
        stmt = stmt.where(BrokerAccount.exchange_id == exchange_key)
    stmt = stmt.order_by(
        BrokerAccount.is_default.desc(),
        BrokerAccount.updated_at.desc(),
        BrokerAccount.created_at.desc(),
    )
    return await db.scalar(stmt)


async def _deactivate_group_siblings(
    db: AsyncSession,
    *,
    user_id: UUID,
    provider: str,
    mode: str,
    exchange_id: str,
    keep_account_id: UUID,
) -> None:
    provider_key, exchange_key = _logical_group_key(provider, exchange_id)
    rows = (
        await db.scalars(
            select(BrokerAccount).where(
                BrokerAccount.user_id == user_id,
                BrokerAccount.provider == provider_key,
                BrokerAccount.mode == mode,
            )
        )
    ).all()
    for row in rows:
        if row.id == keep_account_id:
            continue
        if exchange_key and str(row.exchange_id).strip().lower() != exchange_key:
            continue
        row.status = "inactive"
        row.is_default = False


def _to_response(account: BrokerAccount) -> BrokerAccountResponse:
    return BrokerAccountResponse(
        broker_account_id=account.id,
        user_id=account.user_id,
        provider=account.provider,
        exchange_id=account.exchange_id,
        account_uid=account.account_uid,
        mode=account.mode,
        status=account.status,
        is_default=account.is_default,
        is_sandbox=account.is_sandbox,
        key_fingerprint=account.key_fingerprint,
        encryption_version=account.encryption_version,
        updated_source=account.updated_source,
        last_validated_at=account.last_validated_at,
        last_validated_status=account.last_validated_status,
        last_validation_error_code=account.last_validation_error_code,
        capabilities=account.capabilities
        if isinstance(account.capabilities, dict)
        else {},
        validation_metadata=(
            account.validation_metadata
            if isinstance(account.validation_metadata, dict)
            else {}
        ),
        metadata=account.metadata_ if isinstance(account.metadata_, dict) else {},
        created_at=account.created_at,
        updated_at=account.updated_at,
    )


@router.post(
    "", response_model=BrokerAccountResponse, status_code=status.HTTP_201_CREATED
)
async def create_broker_account(
    payload: BrokerAccountCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    cipher = CredentialCipher()
    encrypted_credentials = cipher.encrypt(payload.credentials)
    key_fingerprint = credential_key_fingerprint(payload.credentials)
    last_validated_at: datetime | None = None
    last_validated_status: str | None = None
    validation_metadata: dict[str, Any] = {}
    validation_result: BrokerCredentialValidationResult | None = None

    validation_result = await _validate_provider_credentials(
        provider=payload.provider,
        credentials=payload.credentials,
    )
    last_validated_at = datetime.now(UTC)
    last_validated_status = validation_result.status
    validation_metadata = validation_result.metadata
    if not validation_result.ok:
        _raise_validation_failure(validation_result)

    exchange_id = _resolve_exchange_id(
        provider=payload.provider,
        payload_exchange_id=payload.exchange_id,
        credentials=payload.credentials,
    )
    is_sandbox = _resolve_is_sandbox(
        provider=payload.provider,
        exchange_id=exchange_id,
        payload_is_sandbox=payload.is_sandbox,
        credentials=payload.credentials,
        validation_metadata=validation_metadata,
    )
    account_uid = _resolve_account_uid(
        provider=payload.provider,
        payload_account_uid=payload.account_uid,
        credentials=payload.credentials,
        validation_metadata=validation_metadata,
        key_fingerprint=key_fingerprint,
    )
    capabilities = build_broker_capabilities(
        provider=payload.provider,
        exchange_id=exchange_id,
        is_sandbox=is_sandbox,
    )
    reusable_account = await _find_reusable_broker_account(
        db,
        user_id=user.id,
        provider=payload.provider,
        mode=payload.mode,
        exchange_id=exchange_id,
    )
    if reusable_account is None:
        account = BrokerAccount(
            user_id=user.id,
            provider=payload.provider,
            exchange_id=exchange_id,
            account_uid=account_uid,
            mode=payload.mode,
            encrypted_credentials=encrypted_credentials,
            key_fingerprint=key_fingerprint,
            encryption_version=cipher.encryption_version,
            updated_source="api",
            is_default=False,
            is_sandbox=is_sandbox,
            last_validated_at=last_validated_at,
            last_validated_status=last_validated_status,
            last_validation_error_code=_resolve_last_validation_error_code(
                validation_result=validation_result
            ),
            capabilities=capabilities,
            validation_metadata=validation_metadata,
            metadata_=payload.metadata,
        )
        db.add(account)
        try:
            await db.flush()
        except IntegrityError as exc:
            await db.rollback()
            _raise_integrity_error(exc)
        audit_action = "create"
    else:
        account = reusable_account
        account.exchange_id = exchange_id
        account.account_uid = account_uid
        account.encrypted_credentials = encrypted_credentials
        account.key_fingerprint = key_fingerprint
        account.encryption_version = cipher.encryption_version
        account.updated_source = "api"
        account.is_sandbox = is_sandbox
        account.last_validated_at = last_validated_at
        account.last_validated_status = last_validated_status
        account.last_validation_error_code = _resolve_last_validation_error_code(
            validation_result=validation_result
        )
        account.capabilities = capabilities
        account.validation_metadata = validation_metadata
        account.metadata_ = payload.metadata
        account.status = "active"
        account.last_error = None
        audit_action = "update"

    await _deactivate_group_siblings(
        db,
        user_id=user.id,
        provider=payload.provider,
        mode=payload.mode,
        exchange_id=exchange_id,
        keep_account_id=account.id,
    )
    await _rebalance_default_accounts(
        db,
        user_id=user.id,
        mode=account.mode,
        preferred_account_id=account.id if payload.is_default else None,
    )
    _append_audit_log(
        db,
        account=account,
        action=audit_action,
        metadata={
            "last_validated_status": last_validated_status,
            "exchange_id": exchange_id,
            "account_uid": account_uid,
            "is_default_requested": payload.is_default,
            "upserted_from_create": reusable_account is not None,
        },
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        _raise_integrity_error(exc)
    await db.refresh(account)
    return _to_response(account)


@router.post(
    "/builtin-sandbox",
    response_model=BrokerAccountResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_builtin_sandbox_broker_account(
    payload: BuiltinSandboxBrokerAccountRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await store_ensure_builtin_sandbox_account(
        db,
        user_id=user.id,
        metadata=payload.to_metadata_patch() if payload is not None else None,
    )
    return _to_response(account)


@router.post(
    "/builtin-sandbox/deactivate",
    response_model=BrokerAccountResponse,
)
async def deactivate_builtin_sandbox_broker_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrokerAccountResponse:
    account = await store_deactivate_builtin_sandbox_account(
        db,
        user_id=user.id,
    )
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "BUILTIN_SANDBOX_BROKER_NOT_FOUND",
                "message": "No active built-in sandbox broker account found.",
            },
        )
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
            .order_by(BrokerAccount.is_default.desc(), BrokerAccount.created_at.desc()),
        )
    ).all()
    return [_to_response(row) for row in rows]


@router.get("/ccxt/exchanges", response_model=list[CcxtExchangeInfo])
async def list_ccxt_exchanges(
    user: User = Depends(get_current_user),
) -> list[CcxtExchangeInfo]:
    _ = user
    return [CcxtExchangeInfo(**item) for item in list_supported_ccxt_exchanges()]


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

    # Keep existing metadata when rotating credentials.
    existing_exchange_id = account.exchange_id
    existing_is_sandbox = account.is_sandbox
    existing_account_uid = account.account_uid

    last_validated_at: datetime | None = None
    last_validated_status: str | None = None
    validation_metadata: dict[str, Any] = {}
    validation_result: BrokerCredentialValidationResult | None = None
    validation_result = await _validate_provider_credentials(
        provider=account.provider,
        credentials=payload.credentials,
    )
    last_validated_at = datetime.now(UTC)
    last_validated_status = validation_result.status
    validation_metadata = validation_result.metadata
    if not validation_result.ok:
        _raise_validation_failure(validation_result)

    next_key_fingerprint = credential_key_fingerprint(payload.credentials)
    exchange_id = _resolve_exchange_id(
        provider=account.provider,
        payload_exchange_id=existing_exchange_id,
        credentials=payload.credentials,
    )
    is_sandbox = _resolve_is_sandbox(
        provider=account.provider,
        exchange_id=exchange_id,
        payload_is_sandbox=existing_is_sandbox,
        credentials=payload.credentials,
        validation_metadata=validation_metadata,
    )
    account_uid = _resolve_account_uid(
        provider=account.provider,
        payload_account_uid=existing_account_uid,
        credentials=payload.credentials,
        validation_metadata=validation_metadata,
        key_fingerprint=next_key_fingerprint,
        existing_account_uid=existing_account_uid,
    )
    capabilities = build_broker_capabilities(
        provider=account.provider,
        exchange_id=exchange_id,
        is_sandbox=is_sandbox,
    )

    cipher = CredentialCipher()
    account.encrypted_credentials = cipher.encrypt(payload.credentials)
    account.key_fingerprint = next_key_fingerprint
    account.encryption_version = cipher.encryption_version
    account.updated_source = "api"
    account.exchange_id = exchange_id
    account.account_uid = account_uid
    account.is_sandbox = is_sandbox
    account.last_validated_at = last_validated_at
    account.last_validated_status = last_validated_status
    account.last_validation_error_code = _resolve_last_validation_error_code(
        validation_result=validation_result
    )
    account.capabilities = capabilities
    account.validation_metadata = validation_metadata
    account.status = "active"
    account.last_error = None
    await _rebalance_default_accounts(
        db,
        user_id=user.id,
        mode=account.mode,
        preferred_account_id=account.id if account.is_default else None,
    )
    _append_audit_log(
        db,
        account=account,
        action="update",
        metadata={
            "last_validated_status": last_validated_status,
            "exchange_id": exchange_id,
            "account_uid": account_uid,
        },
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        _raise_integrity_error(exc)
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

    validation_result = await _validate_provider_credentials(
        provider=account.provider,
        credentials=credentials,
    )
    exchange_id = _resolve_exchange_id(
        provider=account.provider,
        payload_exchange_id=account.exchange_id,
        credentials=credentials,
    )
    is_sandbox = _resolve_is_sandbox(
        provider=account.provider,
        exchange_id=exchange_id,
        payload_is_sandbox=account.is_sandbox,
        credentials=credentials,
        validation_metadata=validation_result.metadata,
    )
    account_uid = _resolve_account_uid(
        provider=account.provider,
        payload_account_uid=account.account_uid,
        credentials=credentials,
        validation_metadata=validation_result.metadata,
        key_fingerprint=account.key_fingerprint,
        existing_account_uid=account.account_uid,
    )
    capabilities = build_broker_capabilities(
        provider=account.provider,
        exchange_id=exchange_id,
        is_sandbox=is_sandbox,
    )

    account.last_validated_at = datetime.now(UTC)
    account.last_validated_status = validation_result.status
    account.last_validation_error_code = _resolve_last_validation_error_code(
        validation_result=validation_result
    )
    account.exchange_id = exchange_id
    account.account_uid = account_uid
    account.is_sandbox = is_sandbox
    account.capabilities = capabilities
    account.validation_metadata = validation_result.metadata
    if validation_result.ok:
        account.status = "active"
        account.last_error = None
    else:
        account.status = "error"
        account.last_error = validation_result.message
    await _rebalance_default_accounts(
        db,
        user_id=user.id,
        mode=account.mode,
        preferred_account_id=account.id
        if validation_result.ok and account.is_default
        else None,
    )
    _append_audit_log(
        db,
        account=account,
        action="validate",
        metadata={
            "validation_passed": validation_result.ok,
            "last_validated_status": validation_result.status,
            "last_validation_error_code": account.last_validation_error_code,
            "exchange_id": exchange_id,
            "account_uid": account_uid,
            "validation_metadata": validation_result.metadata,
        },
    )

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        _raise_integrity_error(exc)
    await db.refresh(account)

    if not validation_result.ok:
        _raise_validation_failure(validation_result)

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
    account.is_default = False
    account.updated_source = "api"
    await _rebalance_default_accounts(
        db,
        user_id=user.id,
        mode=account.mode,
        preferred_account_id=None,
    )
    _append_audit_log(
        db,
        account=account,
        action="deactivate",
        metadata={"reason": "manual_deactivate"},
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        _raise_integrity_error(exc)
    await db.refresh(account)
    return _to_response(account)


@router.post("/{broker_account_id}/set-default", response_model=BrokerAccountResponse)
async def set_default_broker_account(
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
    if str(account.status).strip().lower() != "active":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "BROKER_ACCOUNT_NOT_ACTIVE",
                "message": "Only active broker accounts can be set as default.",
            },
        )

    await _rebalance_default_accounts(
        db,
        user_id=user.id,
        mode=account.mode,
        preferred_account_id=account.id,
    )
    _append_audit_log(
        db,
        account=account,
        action="set_default",
        metadata={"broker_account_id": str(account.id)},
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        _raise_integrity_error(exc)
    await db.refresh(account)
    return _to_response(account)
