"""Deployment phase prompt builders and contracts."""

from __future__ import annotations

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
    runtime_summary = _summarize_runtime_state(runtime_state)
    stage = str(phase_stage or "").strip() or "deployment"

    return (
        "[SESSION STATE]\n"
        f"- deployment_phase_stage: {stage}\n"
        f"- already_collected: {collected}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        f"- deployment_runtime_summary: {runtime_summary}\n"
        "[/SESSION STATE]\n\n"
    )


def _summarize_runtime_state(runtime_state: dict[str, Any]) -> str:
    accounts_count = len(runtime_state.get("broker_accounts", [])) if isinstance(
        runtime_state.get("broker_accounts"), list
    ) else 0
    choices_count = len(runtime_state.get("available_broker_choices", [])) if isinstance(
        runtime_state.get("available_broker_choices"), list
    ) else 0
    latest_deployment = (
        runtime_state.get("latest_deployment")
        if isinstance(runtime_state.get("latest_deployment"), dict)
        else {}
    )
    latest_status = str(latest_deployment.get("status", "")).strip().lower() or "none"
    latest_deployment_id = str(
        latest_deployment.get("deployment_id") or latest_deployment.get("id") or ""
    ).strip() or "none"
    auto_execute_pending = bool(runtime_state.get("auto_execute_pending"))
    capital_resolution = (
        runtime_state.get("capital_resolution")
        if isinstance(runtime_state.get("capital_resolution"), dict)
        else {}
    )
    resolved_amount = str(capital_resolution.get("resolved_amount", "")).strip() or "none"
    source = str(capital_resolution.get("source", "")).strip() or "none"
    return (
        f"accounts={accounts_count}, "
        f"choices={choices_count}, "
        f"latest_status={latest_status}, "
        f"latest_deployment_id={latest_deployment_id}, "
        f"auto_execute_pending={str(auto_execute_pending).lower()}, "
        f"capital_resolution_source={source}, "
        f"capital_resolution_amount={resolved_amount}"
    )
