"""Shared helpers for MCP tool implementations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.util.logger import logger


def utc_now_iso() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def to_json(payload: dict[str, Any]) -> str:
    """Serialize MCP tool result payloads consistently."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def log_mcp_tool_result(
    *,
    category: str,
    tool: str,
    ok: bool,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """Emit a compact, consistent log line for each MCP tool response."""
    prefix = f"mcp.{category} tool={tool} ok={ok}"
    if ok:
        logger.info(prefix)
        return

    resolved_code = error_code or "UNKNOWN_ERROR"
    resolved_message = (error_message or "").strip()
    if resolved_message:
        logger.warning(
            "%s error_code=%s error_message=%s",
            prefix,
            resolved_code,
            resolved_message,
        )
        return
    logger.warning("%s error_code=%s", prefix, resolved_code)
