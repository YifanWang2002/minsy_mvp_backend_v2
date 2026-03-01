"""Helpers for compact `[SESSION STATE]` blocks."""

from __future__ import annotations

from typing import Any, Iterable


def compact_state_block(*, items: Iterable[tuple[str, Any]]) -> str:
    parts: list[str] = []
    for raw_key, raw_value in items:
        key = str(raw_key).strip()
        if not key:
            continue
        value = _compact_state_value(raw_value)
        parts.append(f"{key}={value}")
    body = " | ".join(parts) if parts else "empty=true"
    return f"[SESSION STATE]\n{body}\n[/SESSION STATE]\n\n"


def _compact_state_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    compact = " ".join(text.split())
    if not compact:
        return "none"
    # Keep separators stable for parser/model readability.
    return compact.replace("|", "/")
