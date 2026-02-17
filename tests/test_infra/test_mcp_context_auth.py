from __future__ import annotations

from uuid import uuid4

import pytest

from src.mcp.context_auth import (
    MCP_CONTEXT_HEADER,
    create_mcp_context_token,
    decode_mcp_context_token,
    extract_mcp_context_token,
)


def test_create_and_decode_mcp_context_token_roundtrip() -> None:
    user_id = uuid4()
    session_id = uuid4()
    token = create_mcp_context_token(
        user_id=user_id,
        session_id=session_id,
        ttl_seconds=120,
        trace_id="trace_abc",
        phase="strategy",
        request_id="turn_123",
    )
    claims = decode_mcp_context_token(token)

    assert claims.user_id == user_id
    assert claims.session_id == session_id
    assert claims.trace_id == "trace_abc"
    assert claims.phase == "strategy"
    assert claims.request_id == "turn_123"
    assert claims.expires_at > claims.issued_at


def test_decode_mcp_context_token_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError):
        decode_mcp_context_token("not-a-token")


def test_extract_mcp_context_token_from_headers_case_insensitive() -> None:
    token = "abc123"
    headers = {
        "Content-Type": "application/json",
        "X-Minsy-MCP-Context": token,
    }
    resolved = extract_mcp_context_token(headers)
    assert resolved == token

    missing = extract_mcp_context_token({"content-type": "application/json"})
    assert missing is None

    empty = extract_mcp_context_token({MCP_CONTEXT_HEADER: "   "})
    assert empty is None
