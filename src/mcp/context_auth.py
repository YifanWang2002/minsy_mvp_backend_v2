"""Helpers for propagating signed request context into MCP tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Mapping
from uuid import UUID

import jwt

from src.config import settings

MCP_CONTEXT_HEADER = "x-minsy-mcp-context"
MCP_CONTEXT_TOKEN_TYPE = "mcp_context"
MCP_CONTEXT_ISSUER = "minsy-backend"


@dataclass(slots=True, frozen=True)
class McpContextClaims:
    """Verified identity/session context carried in MCP request headers."""

    user_id: UUID
    session_id: UUID | None
    issued_at: datetime
    expires_at: datetime
    trace_id: str | None = None
    phase: str | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_uuid(value: object, *, field_name: str) -> UUID:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid {field_name} in MCP context token.")
    try:
        return UUID(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name} in MCP context token.") from exc


def _parse_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, int | float):
        raise ValueError(f"Invalid {field_name} in MCP context token.")
    return datetime.fromtimestamp(float(value), tz=UTC)


def create_mcp_context_token(
    *,
    user_id: UUID,
    session_id: UUID | None,
    ttl_seconds: int | None = None,
    trace_id: str | None = None,
    phase: str | None = None,
) -> str:
    """Create a short-lived signed token for MCP server-side context resolution."""

    resolved_ttl = ttl_seconds if isinstance(ttl_seconds, int) and ttl_seconds > 0 else 300
    issued_at = _utc_now()
    expires_at = issued_at + timedelta(seconds=resolved_ttl)

    payload: dict[str, object] = {
        "iss": MCP_CONTEXT_ISSUER,
        "type": MCP_CONTEXT_TOKEN_TYPE,
        "sub": str(user_id),
        "sid": str(session_id) if session_id is not None else "",
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    if isinstance(trace_id, str) and trace_id.strip():
        payload["trace_id"] = trace_id.strip()
    if isinstance(phase, str) and phase.strip():
        payload["phase"] = phase.strip()

    return jwt.encode(
        payload,
        settings.effective_mcp_context_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_mcp_context_token(token: str) -> McpContextClaims:
    """Decode and validate one MCP context token."""

    if not isinstance(token, str) or not token.strip():
        raise ValueError("MCP context token is missing.")
    try:
        payload = jwt.decode(
            token.strip(),
            settings.effective_mcp_context_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("MCP context token expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError("Invalid MCP context token.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Invalid MCP context token payload.")

    token_type = str(payload.get("type", "")).strip()
    if token_type != MCP_CONTEXT_TOKEN_TYPE:
        raise ValueError("Invalid MCP context token type.")

    issuer = str(payload.get("iss", "")).strip()
    if issuer != MCP_CONTEXT_ISSUER:
        raise ValueError("Invalid MCP context token issuer.")

    user_id = _parse_uuid(payload.get("sub"), field_name="sub")
    raw_session_id = payload.get("sid")
    session_id: UUID | None = None
    if isinstance(raw_session_id, str) and raw_session_id.strip():
        session_id = _parse_uuid(raw_session_id, field_name="sid")

    issued_at = _parse_timestamp(payload.get("iat"), field_name="iat")
    expires_at = _parse_timestamp(payload.get("exp"), field_name="exp")
    trace_id = (
        payload.get("trace_id").strip()
        if isinstance(payload.get("trace_id"), str) and payload.get("trace_id").strip()
        else None
    )
    phase = (
        payload.get("phase").strip()
        if isinstance(payload.get("phase"), str) and payload.get("phase").strip()
        else None
    )

    return McpContextClaims(
        user_id=user_id,
        session_id=session_id,
        issued_at=issued_at,
        expires_at=expires_at,
        trace_id=trace_id,
        phase=phase,
    )


def extract_mcp_context_token(headers: Mapping[str, str] | None) -> str | None:
    """Read MCP context token from HTTP headers case-insensitively."""

    if headers is None:
        return None
    for key, value in headers.items():
        if str(key).strip().lower() != MCP_CONTEXT_HEADER:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

