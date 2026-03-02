"""Pre-strategy phase handler – collects market/instrument/frequency/holding fields."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from apps.api.agents.phases import Phase
from apps.api.agents.skills.pre_strategy_skills import (
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
from packages.domain.market_data.data import DataLoader
from packages.shared_settings.schema.settings import settings

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
_PRE_STRATEGY_FALLBACK_MARKET_DATA_TOOLS: tuple[str, ...] = (
    "check_symbol_available",
    "get_symbol_data_coverage",
    "market_data_detect_missing_ranges",
    "market_data_fetch_missing_ranges",
    "market_data_get_sync_job",
)
_CRYPTO_QUOTE_SUFFIXES: tuple[str, ...] = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR")
_DATA_RESOLUTION_AWAITING_USER_CHOICE = "awaiting_user_choice"
_DATA_RESOLUTION_DOWNLOAD_STARTED = "download_started"
_DATA_RESOLUTION_LOCAL_READY = "local_ready"


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
        phase_data = ctx.session_artifacts.setdefault(
            Phase.PRE_STRATEGY.value,
            self.init_artifacts(),
        )
        profile = self._sanitize_profile(dict(phase_data.get("profile", {})))
        runtime_state = self._compute_runtime_state(
            profile=profile,
            previous_runtime_state=dict(phase_data.get("runtime", {})),
            turn_context=ctx.turn_context,
        )
        phase_data["profile"] = profile
        phase_data["runtime"] = runtime_state
        missing = self._compute_missing(profile)
        phase_data["missing_fields"] = missing

        kyc_data = ctx.session_artifacts.get(Phase.KYC.value, {})
        kyc_profile = dict(kyc_data.get("profile", {}))

        instructions = build_pre_strategy_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        state_block = build_pre_strategy_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
            kyc_profile=kyc_profile,
            runtime_state=runtime_state,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=_build_pre_strategy_tools(),
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
            {"profile": {}, "missing_fields": list(REQUIRED_FIELDS), "runtime": {}},
        )
        profile = dict(phase_data.get("profile", {}))
        runtime_state = dict(phase_data.get("runtime", {}))

        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        profile = self._sanitize_profile(profile)
        runtime_state = self._compute_runtime_state(
            profile=profile,
            previous_runtime_state=runtime_state,
            turn_context=ctx.turn_context,
        )
        missing = self._compute_missing(profile)
        completed = not missing and not self._requires_data_resolution(runtime_state)

        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing
        phase_data["runtime"] = runtime_state

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
            current_instrument = profile.get("target_instrument")
            current_instrument_value = (
                _normalize_target_instrument_value(current_instrument, target_market=market)
                if isinstance(current_instrument, str) and current_instrument.strip()
                else ""
            )
            allowed = set(allowed_order)
            filtered = [
                option
                for option in option_list
                if isinstance(option, dict)
                and isinstance(option.get("id"), str)
                and (
                    normalize_instrument_value(option.get("id")) in allowed
                    or (
                        current_instrument_value
                        and _normalize_target_instrument_value(
                            option.get("id"),
                            target_market=market,
                        )
                        == current_instrument_value
                    )
                )
            ]
            if not filtered and current_instrument_value:
                filtered = [
                    {
                        "id": current_instrument_value,
                        "label": format_instrument_label(
                            market=market,
                            instrument=current_instrument_value,
                        ),
                        "subtitle": current_instrument_value,
                    }
                ]
            if not filtered:
                return None
            if len(filtered) < 2 and any(
                normalize_instrument_value(option.get("id", "")) in allowed
                for option in filtered
            ):
                filtered = [
                    *filtered,
                    *[
                        {
                            "id": instrument,
                            "label": format_instrument_label(
                                market=market, instrument=instrument
                            ),
                            "subtitle": instrument,
                        }
                        for instrument in allowed_order
                    ],
                ]
            deduped: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for option in filtered:
                option_id = str(option.get("id", "")).strip()
                if not option_id or option_id in seen_ids:
                    continue
                seen_ids.add(option_id)
                deduped.append(option)
            filtered = deduped
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
        return {"profile": {}, "missing_fields": list(REQUIRED_FIELDS), "runtime": {}}

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
        raw_market = patch.get("target_market")
        market_value = ""
        if isinstance(raw_market, str) and raw_market.strip():
            market_value = _normalize_field_value("target_market", raw_market)
            if market_value and _is_field_value_allowed(
                field="target_market",
                value=market_value,
                valid_values=valid_values,
            ):
                validated["target_market"] = market_value
            else:
                market_value = ""

        raw_instrument = patch.get("target_instrument")
        if isinstance(raw_instrument, str) and raw_instrument.strip():
            instrument_value = _normalize_target_instrument_value(
                raw_instrument,
                target_market=market_value or None,
            )
            if instrument_value and _is_target_instrument_allowed(
                value=instrument_value,
                target_market=market_value or None,
                valid_values=valid_values,
            ):
                validated["target_instrument"] = instrument_value
                if "target_market" not in validated:
                    inferred_market = _infer_market_for_instrument(instrument_value)
                    if inferred_market and _is_field_value_allowed(
                        field="target_market",
                        value=inferred_market,
                        valid_values=valid_values,
                    ):
                        validated["target_market"] = inferred_market

        for field in ("opportunity_frequency_bucket", "holding_period_bucket"):
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

        raw_market = profile.get("target_market")
        if isinstance(raw_market, str) and raw_market.strip():
            market_value = _normalize_field_value("target_market", raw_market)
            if market_value and _is_field_value_allowed(
                field="target_market",
                value=market_value,
                valid_values=valid_values,
            ):
                cleaned["target_market"] = market_value

        raw_instrument = profile.get("target_instrument")
        target_market = cleaned.get("target_market")
        if isinstance(raw_instrument, str) and raw_instrument.strip():
            instrument_value = _normalize_target_instrument_value(
                raw_instrument,
                target_market=target_market,
            )
            inferred_market = _infer_market_for_instrument(instrument_value) if instrument_value else None
            market_for_validation = target_market or inferred_market
            if instrument_value and _is_target_instrument_allowed(
                value=instrument_value,
                target_market=market_for_validation,
                valid_values=valid_values,
            ):
                cleaned["target_instrument"] = instrument_value
                if target_market is None and inferred_market is not None:
                    cleaned["target_market"] = inferred_market

        for field in ("opportunity_frequency_bucket", "holding_period_bucket"):
            raw_value = profile.get(field)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            value = _normalize_field_value(field, raw_value)
            if not value:
                continue
            if _is_field_value_allowed(field=field, value=value, valid_values=valid_values):
                cleaned[field] = value

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
            return []

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

    def _compute_runtime_state(
        self,
        *,
        profile: dict[str, str],
        previous_runtime_state: dict[str, Any],
        turn_context: dict[str, Any],
    ) -> dict[str, Any]:
        market = profile.get("target_market")
        instrument = profile.get("target_instrument")
        if not market or not instrument:
            return {}

        normalized_market = normalize_market_value(market)
        normalized_instrument = _normalize_target_instrument_value(
            instrument,
            target_market=normalized_market,
        )
        if not normalized_market or not normalized_instrument:
            return {}

        local_available = _is_symbol_available_locally(
            market=normalized_market,
            symbol=normalized_instrument,
        )
        previous_status = ""
        previous_market = ""
        previous_instrument = ""
        if isinstance(previous_runtime_state, dict):
            previous_status = str(
                previous_runtime_state.get("instrument_data_status", "")
            ).strip()
            previous_market = str(
                previous_runtime_state.get("instrument_data_market", "")
            ).strip()
            previous_instrument = str(
                previous_runtime_state.get("instrument_data_symbol", "")
            ).strip()

        same_symbol_as_previous = (
            previous_market == normalized_market
            and previous_instrument == normalized_instrument
        )
        download_started_this_turn = _turn_started_download_for_symbol(
            turn_context=turn_context,
            market=normalized_market,
            symbol=normalized_instrument,
        )

        if local_available:
            status = _DATA_RESOLUTION_LOCAL_READY
        elif download_started_this_turn:
            status = _DATA_RESOLUTION_DOWNLOAD_STARTED
        elif (
            same_symbol_as_previous
            and previous_status == _DATA_RESOLUTION_DOWNLOAD_STARTED
        ):
            status = _DATA_RESOLUTION_DOWNLOAD_STARTED
        else:
            status = _DATA_RESOLUTION_AWAITING_USER_CHOICE

        return {
            "instrument_data_status": status,
            "instrument_data_market": normalized_market,
            "instrument_data_symbol": normalized_instrument,
            "instrument_available_locally": local_available,
        }

    @staticmethod
    def _requires_data_resolution(runtime_state: dict[str, Any]) -> bool:
        status = str(runtime_state.get("instrument_data_status", "")).strip()
        return status == _DATA_RESOLUTION_AWAITING_USER_CHOICE


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
        return _normalize_target_instrument_value(value, target_market=None)
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


def _is_target_instrument_allowed(
    *,
    value: str,
    target_market: str | None,
    valid_values: dict[str, set[str]],
) -> bool:
    if value in valid_values.get("target_instrument", set()):
        return True
    market = normalize_market_value(target_market) if isinstance(target_market, str) and target_market.strip() else ""
    if not market:
        market = _infer_market_for_instrument(value) or ""
    if not market:
        return False
    return _instrument_matches_market(value=value, target_market=market)


def _build_pre_strategy_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "mcp",
            "server_label": "market_data",
            "server_url": settings.market_data_mcp_server_url,
            "allowed_tools": list(_PRE_STRATEGY_FALLBACK_MARKET_DATA_TOOLS),
            "require_approval": "never",
        }
    ]


def _infer_market_for_instrument(instrument: str) -> str | None:
    normalized = normalize_instrument_value(instrument)
    if not normalized:
        return None

    catalog_market = get_market_for_instrument(normalized)
    if catalog_market is not None:
        return catalog_market

    if _instrument_matches_market(value=normalized, target_market="crypto"):
        return "crypto"
    if _instrument_matches_market(value=normalized, target_market="forex"):
        return "forex"
    if _instrument_matches_market(value=normalized, target_market="us_stocks"):
        return "us_stocks"
    if _instrument_matches_market(value=normalized, target_market="futures"):
        return "futures"
    return None


def _normalize_target_instrument_value(
    raw_value: str,
    *,
    target_market: str | None,
) -> str:
    raw = raw_value.strip()
    if not raw:
        return ""

    market = normalize_market_value(target_market) if isinstance(target_market, str) and target_market.strip() else ""
    normalized = normalize_instrument_value(raw)
    if not normalized:
        return ""

    inferred_market = market or _infer_market_for_instrument(normalized) or ""
    if inferred_market == "crypto":
        compact = re.sub(r"[^A-Z0-9]+", "", normalized)
        for quote in _CRYPTO_QUOTE_SUFFIXES:
            if compact.endswith(quote) and len(compact) > len(quote):
                return f"{compact[:-len(quote)]}USD"
        if re.fullmatch(r"[A-Z0-9]{2,12}", compact):
            return f"{compact}USD"
        return ""

    if inferred_market == "forex":
        compact = re.sub(r"[^A-Z]+", "", normalized.replace("=X", ""))
        if len(compact) == 6 and compact.isalpha():
            return compact
        return ""

    if inferred_market == "us_stocks":
        compact = normalized.replace("/", "").replace(" ", "")
        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", compact):
            return compact
        return ""

    if inferred_market == "futures":
        compact = normalized.replace("/", "").replace(" ", "")
        if compact.endswith("=F") and re.fullmatch(r"[A-Z0-9]{1,6}=F", compact):
            return compact
        if re.fullmatch(r"[A-Z0-9]{1,6}", compact):
            return compact
        return ""

    return normalized


def _instrument_matches_market(*, value: str, target_market: str) -> bool:
    normalized = normalize_instrument_value(value)
    market = normalize_market_value(target_market)
    if not normalized or not market:
        return False

    if market == "crypto":
        return any(
            normalized.endswith(quote) and len(normalized) > len(quote)
            for quote in _CRYPTO_QUOTE_SUFFIXES
        )
    if market == "forex":
        compact = normalized.replace("=X", "")
        return len(compact) == 6 and compact.isalpha()
    if market == "us_stocks":
        return re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized) is not None
    if market == "futures":
        compact = normalized.removesuffix("=F")
        return re.fullmatch(r"[A-Z0-9]{1,6}", compact) is not None
    return False


def _is_symbol_available_locally(*, market: str, symbol: str) -> bool:
    if not market or not symbol:
        return False
    loader = DataLoader()
    try:
        market_key = loader.normalize_market(market)
    except ValueError:
        return False
    symbol_key = normalize_instrument_value(symbol)
    return symbol_key in set(loader.get_available_symbols(market_key))


def _turn_started_download_for_symbol(
    *,
    turn_context: dict[str, Any],
    market: str,
    symbol: str,
) -> bool:
    tool_calls_raw = turn_context.get("mcp_tool_calls")
    if not isinstance(tool_calls_raw, list):
        return False

    expected_market = normalize_market_value(market)
    expected_symbol = normalize_instrument_value(symbol)
    for call in tool_calls_raw:
        if not isinstance(call, dict):
            continue
        if str(call.get("status", "")).strip().lower() != "success":
            continue
        name = str(call.get("name") or call.get("tool_name") or "").strip()
        if name not in {"market_data_fetch_missing_ranges", "market_data_get_sync_job"}:
            continue
        if _mcp_call_targets_symbol(
            call=call,
            expected_market=expected_market,
            expected_symbol=expected_symbol,
        ):
            return True
    return False


def _mcp_call_targets_symbol(
    *,
    call: dict[str, Any],
    expected_market: str,
    expected_symbol: str,
) -> bool:
    arguments_raw = call.get("arguments")
    if not isinstance(arguments_raw, str) or not arguments_raw.strip():
        return False
    try:
        arguments = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(arguments, dict):
        return False

    market_raw = arguments.get("market")
    symbol_raw = arguments.get("symbol")
    if not isinstance(market_raw, str) or not isinstance(symbol_raw, str):
        return False
    return (
        normalize_market_value(market_raw) == expected_market
        and normalize_instrument_value(symbol_raw) == expected_symbol
    )
