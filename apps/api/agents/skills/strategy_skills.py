"""Strategy phase prompt builders and contracts."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

_SKILLS_DIR = Path(__file__).parent
_STRATEGY_SKILLS_MD = _SKILLS_DIR / "strategy" / "skills.md"
_STRATEGY_STAGE_DIR = _SKILLS_DIR / "strategy" / "stages"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"
_STRATEGY_PATCH_SKILLS_MD = _SKILLS_DIR / "strategy_patch" / "skills.md"
# After refactor: DSL assets moved from src/engine/ to packages/domain/strategy/
_DOMAIN_STRATEGY_ASSETS_DIR = (
    Path(__file__).resolve().parents[4] / "packages" / "domain" / "strategy" / "assets"
)
_DSL_SPEC_MD = _DOMAIN_STRATEGY_ASSETS_DIR / "DSL_SPEC.md"
_DSL_SCHEMA_JSON = _DOMAIN_STRATEGY_ASSETS_DIR / "strategy_dsl_schema.json"

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


def _load_optional_md(path: Path) -> str:
    if not path.is_file():
        return ""
    return _load_md(path)


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


def _normalize_phase_stage(stage: str | None) -> str:
    if not isinstance(stage, str):
        return ""
    return stage.strip()


def _normalize_prompt_profile(profile: str | None) -> str:
    if not isinstance(profile, str):
        return "full_bootstrap"
    normalized = profile.strip().lower()
    if normalized == "compact":
        return normalized
    if normalized == "full_bootstrap":
        return normalized
    return "full_bootstrap"


@lru_cache(maxsize=32)
def _build_strategy_static_instructions_cached(
    *,
    language: str,
    phase_stage: str,
    prompt_profile: str,
) -> str:
    template = _load_md(_STRATEGY_SKILLS_MD)
    ui_knowledge = _load_md(_UTILS_SKILLS_MD)
    patch_knowledge = (
        _load_md(_STRATEGY_PATCH_SKILLS_MD) if phase_stage == "artifact_ops" else ""
    )
    stage_addendum = _load_optional_stage_addendum(phase_stage)
    include_full_dsl = prompt_profile == "full_bootstrap"
    dsl_spec = _load_optional_md(_DSL_SPEC_MD) if include_full_dsl else ""
    dsl_schema = _load_optional_md(_DSL_SCHEMA_JSON) if include_full_dsl else ""

    rendered = _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": ui_knowledge,
        },
    )
    if stage_addendum:
        rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
    if patch_knowledge:
        rendered = rendered.rstrip() + "\n\n" + patch_knowledge.strip() + "\n"
    if dsl_spec:
        rendered = rendered.rstrip() + "\n\n[DSL SPEC]\n" + dsl_spec.strip() + "\n"
    if dsl_schema:
        rendered = (
            rendered.rstrip()
            + "\n\n[DSL JSON SCHEMA]\n```json\n"
            + dsl_schema.strip()
            + "\n```\n"
        )
    return rendered


def build_strategy_static_instructions(
    *,
    language: str = "en",
    phase_stage: str | None = None,
    prompt_profile: str | None = None,
) -> str:
    """Build static strategy instructions from markdown templates."""
    normalized_language = (
        language.strip().lower() if isinstance(language, str) else "en"
    )
    normalized_stage = _normalize_phase_stage(phase_stage)
    normalized_prompt_profile = _normalize_prompt_profile(prompt_profile)
    return _build_strategy_static_instructions_cached(
        language=normalized_language or "en",
        phase_stage=normalized_stage,
        prompt_profile=normalized_prompt_profile,
    )


def build_strategy_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
    pre_strategy_fields: dict[str, str] | None = None,
    pre_strategy_runtime: dict[str, Any] | None = None,
    session_id: str | None = None,
    choice_selection: dict[str, Any] | None = None,
    trade_snapshot_request: dict[str, Any] | None = None,
    pending_trade_patch: dict[str, Any] | None = None,
) -> str:
    """Build `[SESSION STATE]` block for strategy phase."""

    fields = dict(collected_fields or {})
    pre = dict(pre_strategy_fields or {})
    if missing_fields is None:
        missing_fields = [field for field in REQUIRED_FIELDS if not fields.get(field)]

    has_missing = bool(missing_fields)
    next_missing = missing_fields[0] if missing_fields else "none"
    missing_str = (
        ", ".join(missing_fields) if missing_fields else "none - all collected"
    )

    collected = (
        ", ".join(f"{key}={value}" for key, value in fields.items() if value) or "none"
    )
    pre_scope = (
        ", ".join(f"{key}={value}" for key, value in pre.items() if value) or "none"
    )
    pre_runtime = dict(pre_strategy_runtime or {})
    pre_strategy_instrument_data_status = (
        str(pre_runtime.get("instrument_data_status", "")).strip() or "unknown"
    )
    pre_strategy_instrument_data_symbol = (
        str(pre_runtime.get("instrument_data_symbol", "")).strip() or "none"
    )
    pre_strategy_instrument_data_market = (
        str(pre_runtime.get("instrument_data_market", "")).strip() or "none"
    )
    pre_strategy_instrument_available_locally = bool(
        pre_runtime.get("instrument_available_locally")
    )
    pre_strategy_family_choice = (
        str(pre.get("strategy_family_choice", "")).strip() or "none"
    )
    timeframe_plan = pre_runtime.get("timeframe_plan")
    if isinstance(timeframe_plan, dict):
        pre_strategy_timeframe_primary = (
            str(timeframe_plan.get("primary", "")).strip() or "none"
        )
    else:
        pre_strategy_timeframe_primary = "none"
    pre_strategy_market_regime_summary = (
        str(pre_runtime.get("regime_summary_short", "")).strip() or "none"
    )
    strategy_id = str(fields.get("strategy_id", "")).strip() or "none"
    strategy_market = str(fields.get("strategy_market", "")).strip() or "none"
    strategy_primary_symbol = (
        str(fields.get("strategy_primary_symbol", "")).strip() or "none"
    )
    raw_tickers_csv = str(fields.get("strategy_tickers_csv", "")).strip()
    if raw_tickers_csv:
        strategy_tickers_csv = raw_tickers_csv
    else:
        raw_tickers = fields.get("strategy_tickers")
        if isinstance(raw_tickers, list):
            strategy_tickers_csv = (
                ",".join(str(item).strip() for item in raw_tickers if str(item).strip())
                or "none"
            )
        else:
            strategy_tickers_csv = "none"
    strategy_timeframe = str(fields.get("strategy_timeframe", "")).strip() or "none"
    tool_compat_session_id = (
        session_id.strip()
        if isinstance(session_id, str) and session_id.strip()
        else "none"
    )
    normalized_choice_selection = _normalize_choice_selection(choice_selection)
    choice_selection_present = normalized_choice_selection is not None
    choice_selection_json = (
        _compact_json(normalized_choice_selection)
        if normalized_choice_selection is not None
        else "none"
    )
    normalized_trade_snapshot_request = _normalize_trade_snapshot_request(
        trade_snapshot_request
    )
    trade_snapshot_request_present = normalized_trade_snapshot_request is not None
    trade_snapshot_request_json = (
        _compact_json(normalized_trade_snapshot_request)
        if normalized_trade_snapshot_request is not None
        else "none"
    )
    normalized_pending_trade_patch = _normalize_pending_trade_patch(pending_trade_patch)
    pending_trade_patch_present = normalized_pending_trade_patch is not None
    pending_trade_patch_summary = _summarize_pending_trade_patch(
        normalized_pending_trade_patch
    )
    pending_trade_patch_json = (
        _compact_json(normalized_pending_trade_patch)
        if normalized_pending_trade_patch is not None
        else "none"
    )

    return (
        "[SESSION STATE]\n"
        f"- pre_strategy_scope: {pre_scope}\n"
        f"- pre_strategy_instrument_data_status: {pre_strategy_instrument_data_status}\n"
        f"- pre_strategy_instrument_data_symbol: {pre_strategy_instrument_data_symbol}\n"
        f"- pre_strategy_instrument_data_market: {pre_strategy_instrument_data_market}\n"
        f"- pre_strategy_instrument_available_locally: {str(pre_strategy_instrument_available_locally).lower()}\n"
        f"- pre_strategy_strategy_family_choice: {pre_strategy_family_choice}\n"
        f"- pre_strategy_timeframe_primary: {pre_strategy_timeframe_primary}\n"
        f"- pre_strategy_market_regime_summary: {pre_strategy_market_regime_summary}\n"
        f"- already_collected: {collected}\n"
        f"- confirmed_strategy_id: {strategy_id}\n"
        f"- strategy_market: {strategy_market}\n"
        f"- strategy_primary_symbol: {strategy_primary_symbol}\n"
        f"- strategy_tickers_csv: {strategy_tickers_csv}\n"
        f"- strategy_timeframe: {strategy_timeframe}\n"
        f"- tool_compat_session_id: {tool_compat_session_id}\n"
        f"- structured_choice_selection_present: {str(choice_selection_present).lower()}\n"
        f"- structured_choice_selection_json: {choice_selection_json}\n"
        f"- trade_snapshot_request_present: {str(trade_snapshot_request_present).lower()}\n"
        f"- trade_snapshot_request_json: {trade_snapshot_request_json}\n"
        f"- pending_trade_patch_present: {str(pending_trade_patch_present).lower()}\n"
        f"- pending_trade_patch_summary: {pending_trade_patch_summary}\n"
        f"- pending_trade_patch_json: {pending_trade_patch_json}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        "[/SESSION STATE]\n\n"
    )


def _compact_json(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "none"
    try:
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except Exception:  # noqa: BLE001
        return "none"


def _normalize_choice_selection(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    choice_id = payload.get("choice_id")
    selected_option_id = payload.get("selected_option_id")
    if not isinstance(choice_id, str) or not choice_id.strip():
        return None
    if not isinstance(selected_option_id, str) or not selected_option_id.strip():
        return None
    normalized: dict[str, Any] = {
        "choice_id": choice_id.strip(),
        "selected_option_id": selected_option_id.strip(),
    }
    label = payload.get("selected_option_label")
    if isinstance(label, str) and label.strip():
        normalized["selected_option_label"] = label.strip()
    return normalized


def _normalize_trade_snapshot_request(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    job_id = payload.get("job_id")
    trade_index = payload.get("trade_index")
    if not isinstance(job_id, str) or not job_id.strip():
        return None
    try:
        normalized_trade_index = int(trade_index)
    except (TypeError, ValueError):
        return None
    if normalized_trade_index < 0:
        return None
    normalized: dict[str, Any] = {
        "job_id": job_id.strip(),
        "trade_index": normalized_trade_index,
    }
    trade_uid = payload.get("trade_uid")
    if isinstance(trade_uid, str) and trade_uid.strip():
        normalized["trade_uid"] = trade_uid.strip()
    visible_keys = payload.get("visible_indicator_keys")
    if isinstance(visible_keys, list):
        normalized["visible_indicator_keys"] = [
            str(item).strip()
            for item in visible_keys
            if isinstance(item, str) and item.strip()
        ]
    filters = payload.get("filters")
    if isinstance(filters, dict):
        normalized["filters"] = {
            key: value.strip()
            for key, value in filters.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
    lookback_bars = payload.get("lookback_bars")
    lookforward_bars = payload.get("lookforward_bars")
    if isinstance(lookback_bars, int) and lookback_bars >= 0:
        normalized["lookback_bars"] = lookback_bars
    if isinstance(lookforward_bars, int) and lookforward_bars >= 0:
        normalized["lookforward_bars"] = lookforward_bars
    user_prompt = payload.get("user_prompt")
    if isinstance(user_prompt, str) and user_prompt.strip():
        normalized["user_prompt"] = user_prompt.strip()
    return normalized


def _normalize_pending_trade_patch(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    strategy_id = payload.get("strategy_id")
    patch_ops = payload.get("patch_ops")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        return None
    if not isinstance(patch_ops, list) or not patch_ops:
        return None
    normalized_ops = [dict(item) for item in patch_ops if isinstance(item, dict)]
    if not normalized_ops:
        return None
    normalized: dict[str, Any] = {
        "strategy_id": strategy_id.strip(),
        "patch_ops": normalized_ops,
    }
    source_trade = payload.get("source_trade")
    if isinstance(source_trade, dict):
        normalized["source_trade"] = dict(source_trade)
    backtest_request = payload.get("backtest_request")
    if isinstance(backtest_request, dict):
        normalized["backtest_request"] = dict(backtest_request)
    return normalized


def _summarize_pending_trade_patch(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "none"
    strategy_id = str(payload.get("strategy_id", "")).strip() or "none"
    patch_ops = payload.get("patch_ops")
    patch_count = len(patch_ops) if isinstance(patch_ops, list) else 0
    source_trade = payload.get("source_trade")
    source_summary = "none"
    if isinstance(source_trade, dict):
        job_id = str(source_trade.get("job_id", "")).strip() or "none"
        trade_index = source_trade.get("trade_index")
        source_summary = f"job_id={job_id},trade_index={trade_index}"
    return f"strategy_id={strategy_id},patch_ops={patch_count},source={source_summary}"
