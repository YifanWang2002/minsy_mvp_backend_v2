"""Shared helpers for MCP tool implementations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from packages.infra.observability.logger import logger


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
    error_context: dict[str, Any] | None = None,
) -> None:
    """Emit a compact, consistent log line for each MCP tool response."""
    prefix = f"mcp.{category} tool={tool} ok={ok}"
    if ok:
        logger.info(prefix)
        return

    resolved_code = error_code or "UNKNOWN_ERROR"
    resolved_message = (error_message or "").strip()
    context_text = ""
    if isinstance(error_context, dict) and error_context:
        try:
            context_text = json.dumps(
                error_context,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            context_text = str(error_context)
    if resolved_message:
        if context_text:
            logger.warning(
                "%s error_code=%s error_message=%s error_context=%s",
                prefix,
                resolved_code,
                resolved_message,
                context_text,
            )
            return
        logger.warning(
            "%s error_code=%s error_message=%s",
            prefix,
            resolved_code,
            resolved_message,
        )
        return
    if context_text:
        logger.warning("%s error_code=%s error_context=%s", prefix, resolved_code, context_text)
        return
    logger.warning("%s error_code=%s", prefix, resolved_code)
