"""KYC prompt builder – static instructions + dynamic session state."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent
_KYC_SKILLS_MD = _SKILLS_DIR / "kyc" / "skills.md"
_KYC_STAGE_DIR = _SKILLS_DIR / "kyc" / "stages"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"

# Canonical field list & allowed enum values – shared with handlers.
REQUIRED_FIELDS: list[str] = [
    "trading_years_bucket",
    "risk_tolerance",
    "return_expectation",
]

VALID_VALUES: dict[str, set[str]] = {
    "trading_years_bucket": {"years_0_1", "years_1_3", "years_3_5", "years_5_plus"},
    "risk_tolerance": {"conservative", "moderate", "aggressive", "very_aggressive"},
    "return_expectation": {"capital_preservation", "balanced_growth", "growth", "high_growth"},
}


@lru_cache(maxsize=2)
def _load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}


def _language_display(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def _render_template(template: str, values: dict[str, str]) -> str:
    output = template
    for key, value in values.items():
        output = output.replace(f"{{{{{key}}}}}", value)
    return output


def _load_optional_stage_addendum(stage: str | None) -> str:
    if not isinstance(stage, str) or not stage.strip():
        return ""
    stage_path = _KYC_STAGE_DIR / f"{stage.strip()}.md"
    if not stage_path.is_file():
        return ""
    return _load_md(stage_path)


def build_kyc_static_instructions(*, language: str = "en", phase_stage: str | None = None) -> str:
    """Return static KYC instructions from markdown templates."""
    template = _load_md(_KYC_SKILLS_MD)
    genui_knowledge = _load_md(_UTILS_SKILLS_MD)
    stage_addendum = _load_optional_stage_addendum(phase_stage)

    rendered = _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": genui_knowledge,
        },
    )
    if stage_addendum:
        rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
    return rendered


def build_kyc_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
) -> str:
    """Return a short ``[SESSION STATE]`` block to prepend to the user message."""
    if missing_fields is None:
        missing_fields = list(REQUIRED_FIELDS)
    missing_str = ", ".join(missing_fields) if missing_fields else "none – all collected"

    collected_str = "none"
    if collected_fields:
        parts = [f"{k}={v}" for k, v in collected_fields.items() if v]
        if parts:
            collected_str = ", ".join(parts)

    has_missing = bool(missing_fields)
    next_missing = missing_fields[0] if missing_fields else "none"

    return (
        "[SESSION STATE]\n"
        f"- already_collected: {collected_str}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        "[/SESSION STATE]\n\n"
    )


def build_kyc_system_prompt(
    *,
    language: str = "en",
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
) -> str:
    """Legacy convenience wrapper for callers that need a single prompt blob."""
    static = build_kyc_static_instructions(language=language)
    dynamic = build_kyc_dynamic_state(
        missing_fields=missing_fields,
        collected_fields=collected_fields,
    )
    return static + "\n" + dynamic
