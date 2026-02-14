"""Stress-test phase prompt builders and contracts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent
_STRESS_SKILLS_MD = _SKILLS_DIR / "stress_test" / "skills.md"
_STRESS_STAGE_DIR = _SKILLS_DIR / "stress_test" / "stages"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"

REQUIRED_FIELDS: list[str] = [
    "strategy_id",
    "backtest_job_id",
    "backtest_status",
]

VALID_STATUS_VALUES: set[str] = {"pending", "running", "done", "failed"}
VALID_DECISION_VALUES: set[str] = {"hold", "iterate", "deploy"}

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


def _load_optional_stage_addendum(stage: str | None) -> str:
    if not isinstance(stage, str) or not stage.strip():
        return ""
    stage_path = _STRESS_STAGE_DIR / f"{stage.strip()}.md"
    if not stage_path.is_file():
        return ""
    return _load_md(stage_path)


def _normalize_phase_stage(stage: str | None) -> str:
    if not isinstance(stage, str):
        return ""
    return stage.strip()


@lru_cache(maxsize=32)
def _build_stress_test_static_instructions_cached(*, language: str, phase_stage: str) -> str:
    template = _load_md(_STRESS_SKILLS_MD)
    ui_knowledge = _load_md(_UTILS_SKILLS_MD)
    rendered = _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": ui_knowledge,
        },
    )
    stage_addendum = _load_optional_stage_addendum(phase_stage)
    if stage_addendum:
        rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
    return rendered


def build_stress_test_static_instructions(
    *,
    language: str = "en",
    phase_stage: str | None = None,
) -> str:
    normalized_language = language.strip().lower() if isinstance(language, str) else "en"
    normalized_stage = _normalize_phase_stage(phase_stage)
    return _build_stress_test_static_instructions_cached(
        language=normalized_language or "en",
        phase_stage=normalized_stage,
    )


def build_stress_test_dynamic_state(
    *,
    collected_fields: dict[str, str] | None = None,
) -> str:
    fields = dict(collected_fields or {})
    missing = [field for field in REQUIRED_FIELDS if not fields.get(field)]
    has_missing = bool(missing)
    next_missing = missing[0] if missing else "none"
    missing_str = ", ".join(missing) if missing else "none - all collected"
    collected = ", ".join(f"{key}={value}" for key, value in fields.items() if value) or "none"
    decision = str(fields.get("stress_test_decision", "")).strip().lower() or "hold"

    return (
        "[SESSION STATE]\n"
        f"- already_collected: {collected}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        f"- stress_test_decision: {decision}\n"
        "- stress_test_decision_options: hold, iterate, deploy\n"
        "[/SESSION STATE]\n\n"
    )
