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
    get_market_for_instrument,
    get_pre_strategy_market_instrument_map,
    get_pre_strategy_valid_values,
    get_yfinance_symbol_for_market_instrument,
    normalize_instrument_value,
    normalize_market_value,
)
from src.config import settings


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
            selected_market=profile.get("target_market"),
        )
        should_force_snapshot_call = bool(
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
            symbol_newly_provided_this_turn_hint=should_force_snapshot_call,
            inferred_instrument_from_user_message=inferred_instrument,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
            tools=_build_pre_strategy_tools() if should_force_snapshot_call else None,
            tool_choice=_build_pre_strategy_tool_choice() if should_force_snapshot_call else None,
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
        if payload.get("choice_id") != "target_instrument":
            return payload

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

        options = payload.get("options")
        if not isinstance(options, list):
            return payload

        filtered = [
            option
            for option in options
            if isinstance(option, dict)
            and isinstance(option.get("id"), str)
            and normalize_instrument_value(option.get("id")) in allowed
        ]

        if len(filtered) < 2:
            filtered = [
                {
                    "id": instrument,
                    "label": format_instrument_label(market=market, instrument=instrument),
                    "subtitle": instrument,
                }
                for instrument in allowed_order
            ]

        result = dict(payload)
        result["options"] = filtered
        return result

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


def _build_pre_strategy_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "mcp",
            "server_label": "yfinance",
            "server_url": settings.yfinance_mcp_server_url,
            "allowed_tools": ["check_symbol_available", "get_quote"],
            "require_approval": "never",
        }
    ]


def _build_pre_strategy_tool_choice() -> dict[str, str]:
    return {
        "type": "mcp",
        "server_label": "yfinance",
        "name": "check_symbol_available",
    }


def _infer_instrument_from_message(
    *,
    user_message: str,
    selected_market: Any,
) -> str | None:
    market_catalog = get_pre_strategy_market_instrument_map()
    if not market_catalog:
        return None

    selected_market_key = (
        normalize_market_value(selected_market)
        if isinstance(selected_market, str) and selected_market.strip()
        else ""
    )

    if selected_market_key and selected_market_key in market_catalog:
        candidates: list[tuple[str, str]] = [
            (selected_market_key, instrument)
            for instrument in market_catalog[selected_market_key]
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

        yf_symbol = get_yfinance_symbol_for_market_instrument(
            market=market,
            instrument=instrument,
        )
        if _message_contains_alias(user_message=user_message, alias=yf_symbol):
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
