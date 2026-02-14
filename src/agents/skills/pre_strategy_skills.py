"""Pre-strategy prompt builders and dynamic enum contracts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.engine import DataLoader

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

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}


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
                if isinstance(item, str) and item.strip()
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


def get_yfinance_symbol_for_market_instrument(
    *,
    market: str | None,
    instrument: str | None,
) -> str:
    """Convert local symbol into yfinance-oriented symbol by market rules."""
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
        compact = symbol.replace("=F", "")
        if compact.endswith("1!"):
            return compact
        return f"{compact}1!"

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


def build_pre_strategy_static_instructions(
    *,
    language: str = "en",
    phase_stage: str | None = None,
) -> str:
    """Return static instructions for pre-strategy collection."""
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


def build_pre_strategy_dynamic_state(
    *,
    missing_fields: list[str] | None = None,
    collected_fields: dict[str, str] | None = None,
    kyc_profile: dict[str, str] | None = None,
    symbol_newly_provided_this_turn_hint: bool = False,
    inferred_instrument_from_user_message: str | None = None,
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

    selected_market_raw = (collected_fields or {}).get("target_market")
    selected_market = (
        normalize_market_value(selected_market_raw)
        if isinstance(selected_market_raw, str) and selected_market_raw.strip()
        else ""
    )
    selected_instrument_raw = (collected_fields or {}).get("target_instrument")
    selected_instrument = (
        normalize_instrument_value(selected_instrument_raw)
        if isinstance(selected_instrument_raw, str) and selected_instrument_raw.strip()
        else ""
    )

    allowed_instruments = market_catalog.get(selected_market, ())
    allowed_instruments_str = (
        ", ".join(allowed_instruments)
        if allowed_instruments
        else "none - select target_market first"
    )
    mapped_yf_symbol = get_yfinance_symbol_for_market_instrument(
        market=selected_market,
        instrument=selected_instrument,
    )
    mapped_tradingview_symbol = get_tradingview_symbol_for_market_instrument(
        market=selected_market,
        instrument=selected_instrument,
    )

    market_list_str = (
        ", ".join(market_catalog.keys())
        if market_catalog
        else "none - no local market parquet data found"
    )
    market_symbol_catalog_str = (
        "; ".join(
            f"{market}=[{', '.join(instruments)}]"
            for market, instruments in market_catalog.items()
        )
        if market_catalog
        else "none"
    )

    instrument_mappings_str = (
        ", ".join(
            f"{instrument}:{get_yfinance_symbol_for_market_instrument(market=market, instrument=instrument)}"
            f"|{get_tradingview_symbol_for_market_instrument(market=market, instrument=instrument)}"
            for market, instruments in market_catalog.items()
            for instrument in instruments
        )
        if market_catalog
        else "none"
    )

    return (
        "[SESSION STATE]\n"
        "- phase: pre_strategy\n"
        f"- kyc_profile: {kyc_str}\n"
        f"- already_collected: {pre_collected_str}\n"
        f"- still_missing: {missing_str}\n"
        f"- has_missing_fields: {str(has_missing).lower()}\n"
        f"- next_missing_field: {next_missing}\n"
        f"- available_markets: {market_list_str}\n"
        f"- market_symbol_catalog: {market_symbol_catalog_str}\n"
        f"- selected_target_market: {selected_market or 'none'}\n"
        f"- selected_target_instrument: {selected_instrument or 'none'}\n"
        f"- allowed_instruments_for_selected_market: {allowed_instruments_str}\n"
        f"- mapped_yfinance_symbol_for_selected_instrument: {mapped_yf_symbol}\n"
        f"- mapped_tradingview_symbol_for_selected_instrument: {mapped_tradingview_symbol}\n"
        f"- symbol_newly_provided_this_turn_hint: {str(symbol_newly_provided_this_turn_hint).lower()}\n"
        f"- inferred_instrument_from_user_message: "
        f"{normalize_instrument_value(inferred_instrument_from_user_message) if inferred_instrument_from_user_message else 'none'}\n"
        f"- instrument_symbol_map (instrument:yfinance|tradingview): {instrument_mappings_str}\n"
        "- symbol_format_rules: "
        "for check_symbol_available/get_quote use yfinance-style conversion "
        "(stock=TICKER, crypto=BASE-USD, forex=PAIR=X, futures=SYMBOL=F); "
        "for tradingview_chart use (stock=TICKER, crypto=BINANCE:BASEUSDT, "
        "forex=FX:PAIR, futures=SYMBOL1!).\n"
        "[/SESSION STATE]\n\n"
    )
