"""Deployment phase prompt builders and contracts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_SKILLS_DIR = Path(__file__).parent
_DEPLOYMENT_SKILLS_MD = _SKILLS_DIR / "deployment" / "skills.md"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"

REQUIRED_FIELDS: list[str] = [
    "selected_broker_account_id",
    "deployment_confirmation_status",
]
VALID_STATUS_VALUES: set[str] = {"ready", "deployed", "blocked"}
VALID_BROKER_READINESS_VALUES: set[str] = {
    "unknown",
    "no_broker",
    "needs_choice",
    "ready",
    "blocked",
}
VALID_CONFIRMATION_VALUES: set[str] = {
    "pending",
    "confirmed",
    "needs_changes",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}


@lru_cache(maxsize=8)
def _load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_template(template: str, values: dict[str, str]) -> str:
    output = template
    for key, value in values.items():
        output = output.replace(f"{{{{{key}}}}}", value)
    return output


def _language_display(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


@lru_cache(maxsize=16)
def _build_deployment_static_instructions_cached(*, language: str) -> str:
    template = _load_md(_DEPLOYMENT_SKILLS_MD)
    ui_knowledge = _load_md(_UTILS_SKILLS_MD)
    return _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": ui_knowledge,
        },
    )


def build_deployment_static_instructions(*, language: str = "en") -> str:
    normalized_language = language.strip().lower() if isinstance(language, str) else "en"
    return _build_deployment_static_instructions_cached(language=normalized_language or "en")


def build_deployment_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, Any] | None = None,
    deployment_state: dict[str, Any] | None = None,
    phase_stage: str | None = None,
) -> str:
    fields = dict(collected_fields or {})
    runtime_state = dict(deployment_state or {})
    missing = list(missing_fields or [])
    has_missing = bool(missing)
    next_missing = missing[0] if missing else "none"
    missing_str = ", ".join(missing) if missing else "none - all collected"
    collected = (
        ", ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        or "none"
    )
    runtime_json = json.dumps(runtime_state, ensure_ascii=True, sort_keys=True)
    stage = str(phase_stage or "").strip() or "deployment"

    return (
        "[SESSION STATE]\n"
        f"- deployment_phase_stage: {stage}\n"
        f"- already_collected: {collected}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        f"- deployment_runtime_state: {runtime_json}\n"
        "[/SESSION STATE]\n\n"
    )
