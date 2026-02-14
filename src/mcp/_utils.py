"""Shared helpers for MCP tool implementations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def to_json(payload: dict[str, Any]) -> str:
    """Serialize MCP tool result payloads consistently."""
    return json.dumps(payload, ensure_ascii=False, default=str)
