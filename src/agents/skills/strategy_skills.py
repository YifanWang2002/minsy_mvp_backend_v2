"""Strategy phase prompt builders and contracts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent
_STRATEGY_SKILLS_MD = _SKILLS_DIR / "strategy" / "skills.md"
_STRATEGY_STAGE_DIR = _SKILLS_DIR / "strategy" / "stages"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"

REQUIRED_FIELDS: list[str] = [
    "strategy_id",
]

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}


@lru_cache(maxsize=16)
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
    stage_path = _STRATEGY_STAGE_DIR / f"{stage.strip()}.md"
    if not stage_path.is_file():
        return ""
    return _load_md(stage_path)


def build_strategy_static_instructions(
    *,
    language: str = "en",
    phase_stage: str | None = None,
) -> str:
    """Build static strategy instructions from markdown templates."""

    template = _load_md(_STRATEGY_SKILLS_MD)
    ui_knowledge = _load_md(_UTILS_SKILLS_MD)
    stage_addendum = _load_optional_stage_addendum(phase_stage)

    rendered = _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": ui_knowledge,
        },
    )
    if stage_addendum:
        rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
    return rendered


def build_strategy_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
    pre_strategy_fields: dict[str, str] | None = None,
) -> str:
    """Build `[SESSION STATE]` block for strategy phase."""

    fields = dict(collected_fields or {})
    pre = dict(pre_strategy_fields or {})
    if missing_fields is None:
        missing_fields = [field for field in REQUIRED_FIELDS if not fields.get(field)]

    has_missing = bool(missing_fields)
    next_missing = missing_fields[0] if missing_fields else "none"
    missing_str = ", ".join(missing_fields) if missing_fields else "none - all collected"

    collected = ", ".join(f"{key}={value}" for key, value in fields.items() if value) or "none"
    pre_scope = ", ".join(f"{key}={value}" for key, value in pre.items() if value) or "none"

    return (
        "[SESSION STATE]\n"
        f"- pre_strategy_scope: {pre_scope}\n"
        f"- already_collected: {collected}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        "[/SESSION STATE]\n\n"
    )
