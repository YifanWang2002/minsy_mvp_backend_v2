"""Pre-strategy phase handler – collects market/instrument/frequency/holding fields."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from src.agents.phases import Phase
from src.agents.skills.pre_strategy_skills import (
    REQUIRED_FIELDS,
    build_pre_strategy_dynamic_state,
    build_pre_strategy_static_instructions,
    format_instrument_label,
    get_instruments_for_market,
    get_market_data_symbol_for_market_instrument,
    get_market_for_instrument,
    get_pre_strategy_market_instrument_map,
    get_pre_strategy_valid_values,
    normalize_instrument_value,
    normalize_market_value,
)
from src.config import settings

_MARKET_DISPLAY_LABELS: dict[str, str] = {
    "us_stocks": "US Stocks",
    "crypto": "Crypto",
    "forex": "Forex",
    "futures": "Futures",
}
_OPPORTUNITY_FREQUENCY_ORDER: tuple[str, ...] = (
    "few_per_month",
    "few_per_week",
    "daily",
    "multiple_per_day",
)
_HOLDING_PERIOD_ORDER: tuple[str, ...] = (
    "intraday_scalp",
    "intraday",
    "swing_days",
    "position_weeks_plus",
)
_MARKET_OPTION_SUBTITLES: dict[str, str] = {
    "us_stocks": "US equities and ETFs",
    "crypto": "Major crypto spot pairs",
    "forex": "Major FX currency pairs",
    "futures": "Global index futures",
}
_OPPORTUNITY_FREQUENCY_LABELS: dict[str, str] = {
    "few_per_month": "Few per month",
    "few_per_week": "Few per week",
    "daily": "Daily",
    "multiple_per_day": "Multiple per day",
}
_OPPORTUNITY_FREQUENCY_SUBTITLES: dict[str, str] = {
    "few_per_month": "1-3 setups each month",
    "few_per_week": "A handful of setups weekly",
    "daily": "At least one setup per day",
    "multiple_per_day": "Several intraday setups",
}
_HOLDING_PERIOD_LABELS: dict[str, str] = {
    "intraday_scalp": "Intraday Scalp",
    "intraday": "Intraday",
    "swing_days": "Swing Days",
    "position_weeks_plus": "Position Weeks+",
}
_HOLDING_PERIOD_SUBTITLES: dict[str, str] = {
    "intraday_scalp": "Seconds to minutes",
    "intraday": "Open and close same day",
    "swing_days": "Hold for a few days",
    "position_weeks_plus": "Hold for weeks or longer",
}
_FALLBACK_QUESTION_BY_FIELD: dict[str, str] = {
    "target_market": "Which market do you want to trade?",
    "target_instrument": "Which instrument do you want to focus on?",
    "opportunity_frequency_bucket": "How often do you want trade opportunities?",
    "holding_period_bucket": "What holding period style do you prefer?",
}
_FALLBACK_SUBTITLE_BY_FIELD: dict[str, str] = {
    "target_market": "Choose one market so I can scope symbols correctly.",
    "target_instrument": "Pick one primary symbol for setup design.",
    "opportunity_frequency_bucket": "This controls signal cadence and strategy style.",
    "holding_period_bucket": "This controls entry and exit horizon.",
}


class PreStrategyHandler:
    """Implements :class:`PhaseHandler` for the pre-strategy phase."""

    # -- protocol properties -------------------------------------------

    @property
    def phase_name(self) -> str:
        return Phase.PRE_STRATEGY.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return get_pre_strategy_valid_values()

    # -- prompt --------------------------------------------------------

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.PRE_STRATEGY.value, {})
        profile = dict(phase_data.get("profile", {}))
        missing = self._compute_missing(profile)

        kyc_data = ctx.session_artifacts.get(Phase.KYC.value, {})
        kyc_profile = dict(kyc_data.get("profile", {}))
        inferred_instrument = _infer_instrument_from_message(
            user_message=user_message,
            target_market=profile.get("target_market"),
        )
        should_hint_symbol_snapshot = bool(
            not profile.get("target_instrument")
            and inferred_instrument is not None
        )

        instructions = build_pre_strategy_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        state_block = build_pre_strategy_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
            kyc_profile=kyc_profile,
            symbol_newly_provided_this_turn_hint=should_hint_symbol_snapshot,
            inferred_instrument_from_user_message=inferred_instrument,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=(
                [
                    {
                        "type": "mcp",
                        "server_label": "market_data",
                        "server_url": settings.mcp_server_url,
                        "allowed_tools": [
                            "check_symbol_available",
                            "get_symbol_quote",
                        ],
                        "require_approval": "never",
                    }
                ]
                if should_hint_symbol_snapshot
                else None
            ),
            model=settings.openai_response_model,
            reasoning={"effort": "none"},
        )

    # -- post-process --------------------------------------------------

    async def post_process(
        self,
        ctx: PhaseContext,
        raw_patches: list[dict[str, Any]],
        db: AsyncSession,
    ) -> PostProcessResult:
        artifacts = ctx.session_artifacts
        phase_data = artifacts.setdefault(
            Phase.PRE_STRATEGY.value,
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)},
        )
        profile = dict(phase_data.get("profile", {}))

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        profile = self._sanitize_profile(profile)
        missing = self._compute_missing(profile)
        completed = not missing

        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        result = PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=completed,
        )
        if completed:
            result.next_phase = Phase.STRATEGY.value
            result.transition_reason = "pre_strategy_completed_to_strategy"
        return result

    # -- genui ---------------------------------------------------------

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        if payload.get("type") != "choice_prompt":
            return payload
        choice_id = str(payload.get("choice_id", "")).strip()
        if not choice_id:
            return payload

        options = payload.get("options")
        option_list = options if isinstance(options, list) else []
        result = dict(payload)

        if choice_id == "target_market":
            market_order = tuple(get_pre_strategy_market_instrument_map().keys())
            allowed_markets = set(market_order)
            filtered = [
                option
                for option in option_list
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
                and normalize_market_value(option.get("id")) in allowed_markets
            ]
            if len(filtered) < 2:
                filtered = [
                    {
                        "id": market,
                        "label": _MARKET_DISPLAY_LABELS.get(
                            market, market.replace("_", " ").title()
                        ),
                        "subtitle": market,
                    }
                    for market in market_order
                ]
            result["options"] = filtered
            return result

        if choice_id == "target_instrument":
            phase_data = ctx.session_artifacts.get(Phase.PRE_STRATEGY.value, {})
            profile = phase_data.get("profile", {})
            market_raw = profile.get("target_market")
            if not isinstance(market_raw, str):
                return payload

            market = normalize_market_value(market_raw)
            allowed_order = get_instruments_for_market(market)
            if not allowed_order:
                return payload
            allowed = set(allowed_order)
            filtered = [
                option
                for option in option_list
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
                and normalize_instrument_value(option.get("id")) in allowed
            ]
            if len(filtered) < 2:
                filtered = [
                    {
                        "id": instrument,
                        "label": format_instrument_label(
                            market=market, instrument=instrument
                        ),
                        "subtitle": instrument,
                    }
                    for instrument in allowed_order
                ]
            result["options"] = filtered
            return result

        if choice_id == "opportunity_frequency_bucket":
            allowed = set(_OPPORTUNITY_FREQUENCY_ORDER)
            filtered = [
                option
                for option in option_list
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
                and option.get("id") in allowed
            ]
            if len(filtered) < 2:
                filtered = [
                    {"id": value, "label": value.replace("_", " ").title(), "subtitle": value}
                    for value in _OPPORTUNITY_FREQUENCY_ORDER
                ]
            result["options"] = filtered
            return result

        if choice_id == "holding_period_bucket":
            allowed = set(_HOLDING_PERIOD_ORDER)
            filtered = [
                option
                for option in option_list
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
                and option.get("id") in allowed
            ]
            if len(filtered) < 2:
                filtered = [
                    {"id": value, "label": value.replace("_", " ").title(), "subtitle": value}
                    for value in _HOLDING_PERIOD_ORDER
                ]
            result["options"] = filtered
            return result

        return payload

    def build_fallback_choice_prompt(
        self,
        *,
        missing_fields: list[str],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        target_field = next(
            (field for field in missing_fields if field in REQUIRED_FIELDS),
            None,
        )
        if target_field is None:
            return None

        options = self._build_fallback_options_for_field(
            target_field=target_field,
            ctx=ctx,
        )
        if len(options) < 2:
            return None

        payload: dict[str, Any] = {
            "type": "choice_prompt",
            "choice_id": target_field,
            "question": _FALLBACK_QUESTION_BY_FIELD.get(
                target_field,
                "Please choose one option.",
            ),
            "options": options,
        }
        subtitle = _FALLBACK_SUBTITLE_BY_FIELD.get(target_field)
        if isinstance(subtitle, str) and subtitle.strip():
            payload["subtitle"] = subtitle.strip()
        return payload

    # -- artifacts init ------------------------------------------------

    def init_artifacts(self) -> dict[str, Any]:
        return {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)}

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        if ctx.language == "zh":
            return (
                "接下来进入策略准备阶段。告诉我你想交易的市场和标的，"
                "比如“美股 us_stocks + SPY”或“加密 crypto + BTCUSD”。"
            )
        return (
            "Next, let's define your strategy scope. "
            "Tell me the market and instrument you want to trade "
            "(for example: us_stocks + SPY)."
        )

    # -- internal helpers -----------------------------------------------

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not _has_value(profile.get(f))]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, str]:
        valid_values = self.valid_values
        validated: dict[str, str] = {}

        for field in REQUIRED_FIELDS:
            raw_value = patch.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            value = _normalize_field_value(field, raw_value)
            if not value:
                continue
            if _is_field_value_allowed(field=field, value=value, valid_values=valid_values):
                validated[field] = value
        return validated

    def _sanitize_profile(self, profile: dict[str, Any]) -> dict[str, str]:
        """Ensure instrument is valid for selected market; infer market if possible."""
        valid_values = self.valid_values
        cleaned: dict[str, str] = {}

        for field in REQUIRED_FIELDS:
            raw_value = profile.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            value = _normalize_field_value(field, raw_value)
            if not value:
                continue
            if _is_field_value_allowed(field=field, value=value, valid_values=valid_values):
                cleaned[field] = value

        market = cleaned.get("target_market")
        instrument = cleaned.get("target_instrument")

        if market is None and instrument is not None:
            inferred_market = get_market_for_instrument(instrument)
            if inferred_market is not None:
                cleaned["target_market"] = inferred_market
                market = inferred_market

        if (
            market is not None
            and instrument is not None
            and instrument not in set(get_instruments_for_market(market))
        ):
            cleaned.pop("target_instrument", None)

        return cleaned

    def _build_fallback_options_for_field(
        self,
        *,
        target_field: str,
        ctx: PhaseContext,
    ) -> list[dict[str, str]]:
        if target_field == "target_market":
            market_order = tuple(get_pre_strategy_market_instrument_map().keys())
            return [
                {
                    "id": market,
                    "label": _MARKET_DISPLAY_LABELS.get(
                        market,
                        market.replace("_", " ").title(),
                    ),
                    "subtitle": _MARKET_OPTION_SUBTITLES.get(market, market),
                }
                for market in market_order
            ]

        if target_field == "target_instrument":
            phase_data = ctx.session_artifacts.get(Phase.PRE_STRATEGY.value, {})
            profile = phase_data.get("profile", {})
            market_raw = profile.get("target_market")
            if not isinstance(market_raw, str):
                return []
            market = normalize_market_value(market_raw)
            allowed_order = get_instruments_for_market(market)
            return [
                {
                    "id": instrument,
                    "label": format_instrument_label(
                        market=market,
                        instrument=instrument,
                    ),
                    "subtitle": instrument,
                }
                for instrument in allowed_order
            ]

        if target_field == "opportunity_frequency_bucket":
            return [
                {
                    "id": value,
                    "label": _OPPORTUNITY_FREQUENCY_LABELS.get(
                        value,
                        value.replace("_", " ").title(),
                    ),
                    "subtitle": _OPPORTUNITY_FREQUENCY_SUBTITLES.get(value, value),
                }
                for value in _OPPORTUNITY_FREQUENCY_ORDER
            ]

        if target_field == "holding_period_bucket":
            return [
                {
                    "id": value,
                    "label": _HOLDING_PERIOD_LABELS.get(
                        value,
                        value.replace("_", " ").title(),
                    ),
                    "subtitle": _HOLDING_PERIOD_SUBTITLES.get(value, value),
                }
                for value in _HOLDING_PERIOD_ORDER
            ]

        return []


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _normalize_field_value(field: str, value: str) -> str:
    if field == "target_market":
        return normalize_market_value(value)
    if field == "target_instrument":
        return normalize_instrument_value(value)
    return value.strip()


def _is_field_value_allowed(
    *,
    field: str,
    value: str,
    valid_values: dict[str, set[str]],
) -> bool:
    allowed = valid_values.get(field, set())
    if not allowed and field in {"target_market", "target_instrument"}:
        return True
    return value in allowed


def _infer_instrument_from_message(
    *,
    user_message: str,
    target_market: Any,
) -> str | None:
    market_catalog = get_pre_strategy_market_instrument_map()
    if not market_catalog:
        return None

    target_market_key = (
        normalize_market_value(target_market)
        if isinstance(target_market, str) and target_market.strip()
        else ""
    )

    if target_market_key and target_market_key in market_catalog:
        candidates: list[tuple[str, str]] = [
            (target_market_key, instrument)
            for instrument in market_catalog[target_market_key]
        ]
    else:
        candidates = [
            (market, instrument)
            for market, instruments in market_catalog.items()
            for instrument in instruments
        ]

    for market, instrument in candidates:
        for alias in _instrument_aliases(market=market, instrument=instrument):
            if _message_contains_alias(user_message=user_message, alias=alias):
                return instrument

        market_data_symbol = get_market_data_symbol_for_market_instrument(
            market=market,
            instrument=instrument,
        )
        if _message_contains_alias(user_message=user_message, alias=market_data_symbol):
            return instrument

    return None


def _instrument_aliases(*, market: str, instrument: str) -> tuple[str, ...]:
    symbol = normalize_instrument_value(instrument)
    aliases: set[str] = {symbol}

    if market == "crypto":
        if symbol.endswith("USD") and len(symbol) > 3:
            base = symbol[:-3]
            aliases.update(
                {
                    base,
                    f"{base}USD",
                    f"{base}USDT",
                    f"{base}/USD",
                    f"{base}/USDT",
                    f"{base}-USD",
                    f"{base}-USDT",
                }
            )
        elif symbol.endswith("USDT") and len(symbol) > 4:
            base = symbol[:-4]
            aliases.update(
                {
                    base,
                    f"{base}USD",
                    f"{base}USDT",
                    f"{base}/USD",
                    f"{base}/USDT",
                }
            )

    if market == "forex" and len(symbol) == 6 and symbol.isalpha():
        aliases.add(f"{symbol[:3]}/{symbol[3:]}")
        aliases.add(f"{symbol[:3]}-{symbol[3:]}")

    return tuple(sorted(aliases, key=len, reverse=True))


def _message_contains_alias(*, user_message: str, alias: str) -> bool:
    alias_clean = re.sub(r"[^a-z0-9]+", "", alias.lower())
    if not alias_clean:
        return False

    lower = user_message.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if len(alias_clean) <= 3:
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias_clean)}(?![a-z0-9])",
                lower,
            )
            is not None
        )
    return alias_clean in compact
