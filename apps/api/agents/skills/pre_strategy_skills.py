"""Pre-strategy prompt builders and dynamic enum contracts."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.domain.market_data.data import DataLoader

_SKILLS_DIR = Path(__file__).parent
_PRE_STRATEGY_SKILLS_MD = _SKILLS_DIR / "pre_strategy" / "skills.md"
_PRE_STRATEGY_STAGE_DIR = _SKILLS_DIR / "pre_strategy" / "stages"
_UTILS_SKILLS_MD = _SKILLS_DIR / "utils" / "skills.md"
_TRADINGVIEW_SKILLS_MD = _SKILLS_DIR / "utils" / "tradingview.md"

_PREFERRED_MARKET_ORDER: tuple[str, ...] = (
    "us_stocks",
    "crypto",
    "forex",
    "futures",
)

REQUIRED_FIELDS: list[str] = [
    "target_market",
    "target_instrument",
    "opportunity_frequency_bucket",
    "holding_period_bucket",
    "strategy_family_choice",
]

_OPPORTUNITY_FREQUENCY_VALUES: set[str] = {
    "few_per_month",
    "few_per_week",
    "daily",
    "multiple_per_day",
}

_HOLDING_PERIOD_VALUES: set[str] = {
    "intraday_scalp",
    "intraday",
    "swing_days",
    "position_weeks_plus",
}

_STRATEGY_FAMILY_VALUES: set[str] = {
    "trend_continuation",
    "mean_reversion",
    "volatility_regime",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}

_SYNTHETIC_SYMBOL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[A-Z0-9]+BENCH[0-9A-F]{6,}$"),
)


def _is_user_selectable_symbol(symbol: str) -> bool:
    """Hide synthetic benchmark artifacts from end-user instrument pickers."""
    normalized = symbol.strip().upper()
    if not normalized:
        return False
    return all(pattern.fullmatch(normalized) is None for pattern in _SYNTHETIC_SYMBOL_PATTERNS)


@lru_cache(maxsize=4)
def _load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_market_catalog() -> dict[str, tuple[str, ...]]:
    """Return dynamic market->symbols catalog from local parquet data."""
    loader = DataLoader()
    catalog: dict[str, tuple[str, ...]] = {}

    available_markets = loader.get_available_markets()
    ordered_markets = sorted(
        available_markets,
        key=lambda market: (
            _PREFERRED_MARKET_ORDER.index(market)
            if market in _PREFERRED_MARKET_ORDER
            else len(_PREFERRED_MARKET_ORDER),
            market,
        ),
    )

    for market in ordered_markets:
        symbols = sorted(
            {
                item.strip().upper()
                for item in loader.get_available_symbols(market)
                if (
                    isinstance(item, str)
                    and item.strip()
                    and _is_user_selectable_symbol(item)
                )
            }
        )
        if symbols:
            catalog[market] = tuple(symbols)
    return catalog


def get_pre_strategy_market_instrument_map() -> dict[str, tuple[str, ...]]:
    """Return a copy of dynamic market->instrument mapping."""
    return dict(_load_market_catalog())


def get_pre_strategy_valid_values() -> dict[str, set[str]]:
    """Return valid enum values for pre-strategy fields."""
    market_catalog = get_pre_strategy_market_instrument_map()
    return {
        "target_market": set(market_catalog.keys()),
        "target_instrument": {
            instrument
            for instruments in market_catalog.values()
            for instrument in instruments
        },
        "opportunity_frequency_bucket": set(_OPPORTUNITY_FREQUENCY_VALUES),
        "holding_period_bucket": set(_HOLDING_PERIOD_VALUES),
        "strategy_family_choice": set(_STRATEGY_FAMILY_VALUES),
    }


def normalize_market_value(market: str) -> str:
    """Normalize market id, respecting DataLoader aliases when possible."""
    raw = market.strip()
    if not raw:
        return ""
    try:
        return DataLoader().normalize_market(raw)
    except Exception:  # noqa: BLE001
        return raw.lower()


def normalize_instrument_value(instrument: str) -> str:
    return instrument.strip().upper()


def get_market_for_instrument(instrument: str) -> str | None:
    normalized_instrument = normalize_instrument_value(instrument)
    if not normalized_instrument:
        return None
    for market, instruments in get_pre_strategy_market_instrument_map().items():
        if normalized_instrument in instruments:
            return market
    return None


def get_instruments_for_market(market: str | None) -> tuple[str, ...]:
    if not isinstance(market, str) or not market.strip():
        return ()
    market_key = normalize_market_value(market)
    return get_pre_strategy_market_instrument_map().get(market_key, ())


def format_instrument_label(
    *,
    market: str | None,
    instrument: str,
) -> str:
    """Build a user-facing label from a local symbol."""
    symbol = normalize_instrument_value(instrument)
    market_key = normalize_market_value(market or "")

    if market_key == "forex" and len(symbol) == 6 and symbol.isalpha():
        return f"{symbol[:3]}/{symbol[3:]}"
    if market_key == "crypto":
        if symbol.endswith("USDT") and len(symbol) > 4:
            return f"{symbol[:-4]}/USDT"
        if symbol.endswith("USD") and len(symbol) > 3:
            return f"{symbol[:-3]}/USD"
    return symbol


def get_market_data_symbol_for_market_instrument(
    *,
    market: str | None,
    instrument: str | None,
) -> str:
    """Convert local symbol into market-data tool symbol by market rules."""
    if not isinstance(instrument, str) or not instrument.strip():
        return "none - select target_instrument first"

    symbol = normalize_instrument_value(instrument)
    market_key = normalize_market_value(market or "")

    if market_key == "us_stocks":
        return symbol.replace(".", "-")

    if market_key == "crypto":
        compact = symbol.replace("-", "").replace("/", "")
        if compact.endswith("USDT"):
            return f"{compact[:-4]}-USD"
        if compact.endswith("USD"):
            return f"{compact[:-3]}-USD"
        return f"{compact}-USD"

    if market_key == "forex":
        compact = symbol.replace("-", "").replace("/", "").replace("=X", "")
        if len(compact) == 6 and compact.isalpha():
            return f"{compact}=X"
        return symbol

    if market_key == "futures":
        if symbol.endswith("=F"):
            return symbol
        return f"{symbol}=F"

    return symbol


def get_tradingview_symbol_for_market_instrument(
    *,
    market: str | None,
    instrument: str | None,
) -> str:
    """Convert local symbol into TradingView chart symbol by market rules."""
    if not isinstance(instrument, str) or not instrument.strip():
        return "none - select target_instrument first"

    symbol = normalize_instrument_value(instrument)
    market_key = normalize_market_value(market or "")

    if market_key == "crypto":
        compact = symbol.replace("-", "").replace("/", "")
        if compact.endswith("USD"):
            compact = f"{compact[:-3]}USDT"
        if not compact.endswith("USDT"):
            compact = f"{compact}USDT"
        return f"BINANCE:{compact}"

    if market_key == "forex":
        compact = symbol.replace("-", "").replace("/", "").replace("=X", "")
        if len(compact) == 6 and compact.isalpha():
            return f"FX:{compact}"
        return symbol

    if market_key == "futures":
        compact = symbol.replace("=F", "").rstrip("!")
        futures_to_proxy: dict[str, str] = {
            "ES": "SPY",
            "NQ": "QQQ",
            "GC": "XAUUSD",
            "CL": "USOIL",
            "RTY": "IWM",
            "YM": "DIA",
        }
        return futures_to_proxy.get(compact, compact)

    return symbol


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
    stage_path = _PRE_STRATEGY_STAGE_DIR / f"{stage.strip()}.md"
    if not stage_path.is_file():
        return ""
    return _load_md(stage_path)


def _normalize_phase_stage(stage: str | None) -> str:
    if not isinstance(stage, str):
        return ""
    return stage.strip()


@lru_cache(maxsize=32)
def _build_pre_strategy_static_instructions_cached(*, language: str, phase_stage: str) -> str:
    template = _load_md(_PRE_STRATEGY_SKILLS_MD)
    genui_knowledge = _load_md(_UTILS_SKILLS_MD)
    tradingview_knowledge = _load_md(_TRADINGVIEW_SKILLS_MD)
    stage_addendum = _load_optional_stage_addendum(phase_stage)

    rendered = _render_template(
        template,
        {
            "LANG_NAME": _language_display(language),
            "GENUI_KNOWLEDGE": genui_knowledge,
            "TRADINGVIEW_KNOWLEDGE": tradingview_knowledge,
        },
    )
    if stage_addendum:
        rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
    return rendered


def build_pre_strategy_static_instructions(
    *,
    language: str = "en",
    phase_stage: str | None = None,
) -> str:
    """Return static instructions for pre-strategy collection."""
    normalized_language = language.strip().lower() if isinstance(language, str) else "en"
    normalized_stage = _normalize_phase_stage(phase_stage)
    return _build_pre_strategy_static_instructions_cached(
        language=normalized_language or "en",
        phase_stage=normalized_stage,
    )


def build_pre_strategy_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
    kyc_profile: dict[str, str] | None = None,
    runtime_state: dict[str, Any] | None = None,
) -> str:
    """Return dynamic state for pre-strategy turns."""
    if missing_fields is None:
        missing_fields = list(REQUIRED_FIELDS)

    market_catalog = get_pre_strategy_market_instrument_map()

    pre_collected_str = "none"
    if collected_fields:
        parts = [f"{k}={v}" for k, v in collected_fields.items() if v]
        if parts:
            pre_collected_str = ", ".join(parts)

    kyc_str = "none"
    if kyc_profile:
        parts = [f"{k}={v}" for k, v in kyc_profile.items() if v]
        if parts:
            kyc_str = ", ".join(parts)

    has_missing = bool(missing_fields)
    next_missing = missing_fields[0] if missing_fields else "none"
    missing_str = ", ".join(missing_fields) if missing_fields else "none - all collected"

    target_market_raw = (collected_fields or {}).get("target_market")
    target_market = (
        normalize_market_value(target_market_raw)
        if isinstance(target_market_raw, str) and target_market_raw.strip()
        else ""
    )
    target_instrument_raw = (collected_fields or {}).get("target_instrument")
    target_instrument = (
        normalize_instrument_value(target_instrument_raw)
        if isinstance(target_instrument_raw, str) and target_instrument_raw.strip()
        else ""
    )

    allowed_instruments = market_catalog.get(target_market, ())
    allowed_instruments_str = _summarize_instrument_candidates(
        allowed_instruments,
        empty_fallback="none - select target_market first",
    )
    mapped_market_data_symbol = get_market_data_symbol_for_market_instrument(
        market=target_market,
        instrument=target_instrument,
    )
    mapped_tradingview_symbol = get_tradingview_symbol_for_market_instrument(
        market=target_market,
        instrument=target_instrument,
    )
    runtime = dict(runtime_state or {})
    instrument_data_status = str(
        runtime.get("instrument_data_status", "")
    ).strip() or "not_applicable"
    instrument_data_symbol = str(
        runtime.get("instrument_data_symbol", "")
    ).strip() or "none"
    instrument_data_market = str(
        runtime.get("instrument_data_market", "")
    ).strip() or "none"
    instrument_available_locally = bool(runtime.get("instrument_available_locally"))
    strategy_family_choice = (
        str((collected_fields or {}).get("strategy_family_choice", "")).strip() or "none"
    )
    timeframe_plan = runtime.get("timeframe_plan")
    if isinstance(timeframe_plan, dict):
        timeframe_primary = str(timeframe_plan.get("primary", "")).strip() or "none"
        timeframe_secondary = str(timeframe_plan.get("secondary", "")).strip() or "none"
        timeframe_mapping_reason = (
            str(timeframe_plan.get("mapping_reason", "")).strip() or "none"
        )
    else:
        timeframe_primary = "none"
        timeframe_secondary = "none"
        timeframe_mapping_reason = "none"
    regime_snapshot_status = (
        str(runtime.get("regime_snapshot_status", "")).strip() or "pending"
    )
    regime_summary_short = (
        str(runtime.get("regime_summary_short", "")).strip() or "none"
    )
    regime_snapshot_id = (
        str(runtime.get("regime_snapshot_id", "")).strip() or "none"
    )
    regime_family_scores_raw = runtime.get("regime_family_scores")
    if isinstance(regime_family_scores_raw, dict) and regime_family_scores_raw:
        score_parts = []
        for key in (
            "trend_continuation",
            "mean_reversion",
            "volatility_regime",
            "recommended_family",
            "confidence",
        ):
            if key not in regime_family_scores_raw:
                continue
            score_parts.append(f"{key}={regime_family_scores_raw.get(key)}")
        regime_family_scores = ", ".join(score_parts) if score_parts else "none"
        regime_evidence_for = _compact_family_evidence(
            regime_family_scores_raw.get("evidence_for")
        )
        regime_evidence_against = _compact_family_evidence(
            regime_family_scores_raw.get("evidence_against")
        )
    else:
        regime_family_scores = "none"
        regime_evidence_for = "none"
        regime_evidence_against = "none"
    regime_primary_features = _compact_primary_features(
        runtime.get("regime_primary_features")
    )
    regime_probe_required = (
        "true"
        if all(
            isinstance((collected_fields or {}).get(key), str)
            and str((collected_fields or {}).get(key)).strip()
            for key in (
                "target_market",
                "target_instrument",
                "opportunity_frequency_bucket",
                "holding_period_bucket",
            )
        )
        and regime_snapshot_status != "ready"
        else "false"
    )
    download_lookback_days = 730
    download_default_timeframe = "1m"
    download_eta_hint_minutes = 2

    market_list_str = (
        ", ".join(market_catalog.keys())
        if market_catalog
        else "none - no local market parquet data found"
    )
    should_include_market_list = next_missing in {"target_market", "target_instrument"}
    available_markets_line = (
        market_list_str
        if should_include_market_list
        else "omitted - not required for current next_missing_field"
    )

    return (
        "[SESSION STATE]\n"
        "- phase: pre_strategy\n"
        f"- kyc_profile: {kyc_str}\n"
        f"- already_collected: {pre_collected_str}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        f"- available_markets: {available_markets_line}\n"
        f"- target_market: {target_market or 'none'}\n"
        f"- target_instrument: {target_instrument or 'none'}\n"
        f"- strategy_family_choice: {strategy_family_choice}\n"
        f"- allowed_instruments_for_target_market: {allowed_instruments_str}\n"
        f"- mapped_market_data_symbol_for_target_instrument: {mapped_market_data_symbol}\n"
        f"- mapped_tradingview_symbol_for_target_instrument: {mapped_tradingview_symbol}\n"
        f"- instrument_data_status: {instrument_data_status}\n"
        f"- instrument_data_symbol: {instrument_data_symbol}\n"
        f"- instrument_data_market: {instrument_data_market}\n"
        f"- instrument_available_locally: {str(instrument_available_locally).lower()}\n"
        f"- timeframe_plan_primary: {timeframe_primary}\n"
        f"- timeframe_plan_secondary: {timeframe_secondary}\n"
        f"- timeframe_plan_mapping_reason: {timeframe_mapping_reason}\n"
        f"- regime_snapshot_status: {regime_snapshot_status}\n"
        f"- regime_summary_short: {regime_summary_short}\n"
        f"- regime_family_scores: {regime_family_scores}\n"
        f"- regime_evidence_for: {regime_evidence_for}\n"
        f"- regime_evidence_against: {regime_evidence_against}\n"
        f"- regime_primary_features: {regime_primary_features}\n"
        f"- regime_snapshot_id: {regime_snapshot_id}\n"
        f"- regime_probe_required: {regime_probe_required}\n"
        "- download_requires_user_confirmation: true\n"
        f"- download_lookback_days: {download_lookback_days}\n"
        f"- download_default_timeframe: {download_default_timeframe}\n"
        f"- download_eta_hint_minutes: {download_eta_hint_minutes}\n"
        "[/SESSION STATE]\n\n"
    )


def _summarize_instrument_candidates(
    instruments: tuple[str, ...],
    *,
    empty_fallback: str,
    sample_size: int = 16,
) -> str:
    if not instruments:
        return empty_fallback
    head = list(instruments[:sample_size])
    sample = ", ".join(head)
    if len(instruments) <= sample_size:
        return sample
    return f"{sample} ... (+{len(instruments) - sample_size} more)"


def _compact_family_evidence(value: Any) -> str:
    if not isinstance(value, dict):
        return "none"
    parts: list[str] = []
    for family in ("trend_continuation", "mean_reversion", "volatility_regime"):
        entries = value.get(family)
        if not isinstance(entries, list) or not entries:
            continue
        first = next(
            (
                str(item).strip()
                for item in entries
                if isinstance(item, str) and item.strip()
            ),
            "",
        )
        if not first:
            continue
        parts.append(f"{family}={first}")
    return " | ".join(parts) if parts else "none"


def _compact_primary_features(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"

    preferred_order = (
        "adx_n",
        "chop_n",
        "er_n",
        "vol_percentile_n",
        "atr_percentile_n",
        "distance_to_recent_midrange_n",
        "return_autocorr_n",
        "zscore_n",
        "bb_pos_n",
    )
    fields: list[str] = []
    for key in preferred_order:
        if key not in value:
            continue
        fields.append(f"{key}={value.get(key)}")
    if not fields:
        keys = sorted(str(key) for key in value.keys())[:6]
        fields = [f"{key}={value.get(key)}" for key in keys]
    return ", ".join(fields) if fields else "none"
