"""KYC phase handler – collects 3 risk-profile fields."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)
from apps.api.agents.phases import Phase
from apps.api.agents.skills.kyc_skills import (
    REQUIRED_FIELDS,
    VALID_VALUES,
    build_kyc_dynamic_state,
    build_kyc_static_instructions,
)
from packages.shared_settings.schema.settings import settings
from packages.infra.db.models.user import UserProfile

_KYC_FIELD_CHOICE_IDS: dict[str, str] = {
    "trading_years_bucket": "kyc_trading_years_bucket",
    "risk_tolerance": "kyc_risk_tolerance",
    "return_expectation": "kyc_return_expectation",
}
_KYC_QUESTIONS: dict[str, str] = {
    "trading_years_bucket": "How many years of trading experience do you have?",
    "risk_tolerance": "What is your risk tolerance?",
    "return_expectation": "What return expectation do you target?",
}
_KYC_SUBTITLES: dict[str, str] = {
    "trading_years_bucket": "Pick the closest experience bucket.",
    "risk_tolerance": "Choose the risk level that fits your style.",
    "return_expectation": "Choose the return goal you are aiming for.",
}
_KYC_OPTION_META: dict[str, dict[str, tuple[str, str]]] = {
    "trading_years_bucket": {
        "years_0_1": ("0-1 years", "New or early-stage experience"),
        "years_1_3": ("1-3 years", "Some market cycle exposure"),
        "years_3_5": ("3-5 years", "Solid intermediate experience"),
        "years_5_plus": ("5+ years", "Extensive market experience"),
    },
    "risk_tolerance": {
        "conservative": ("Conservative", "Protect capital first"),
        "moderate": ("Moderate", "Balanced risk and return"),
        "aggressive": ("Aggressive", "Higher risk for higher return"),
        "very_aggressive": ("Very aggressive", "Maximum risk tolerance"),
    },
    "return_expectation": {
        "capital_preservation": ("Capital preservation", "Prioritize downside protection"),
        "balanced_growth": ("Balanced growth", "Steady long-term compounding"),
        "growth": ("Growth", "Target stronger account growth"),
        "high_growth": ("High growth", "Pursue maximum growth potential"),
    },
}


class KYCHandler:
    """Implements :class:`PhaseHandler` for the KYC phase."""

    # -- protocol properties -------------------------------------------

    @property
    def phase_name(self) -> str:
        return Phase.KYC.value

    @property
    def required_fields(self) -> list[str]:
        return list(REQUIRED_FIELDS)

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return dict(VALID_VALUES)

    # -- prompt --------------------------------------------------------

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        phase_data = ctx.session_artifacts.get(Phase.KYC.value, {})
        profile = dict(phase_data.get("profile", {}))
        missing = self._compute_missing(profile)

        instructions = build_kyc_static_instructions(
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        state_block = build_kyc_dynamic_state(
            missing_fields=missing,
            collected_fields=profile,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=state_block + user_message,
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
        phase_data = artifacts.setdefault(Phase.KYC.value, {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)})
        profile = dict(phase_data.get("profile", {}))

        # Apply validated patches
        for patch in raw_patches:
            validated = self._validate_patch(patch)
            if validated:
                profile.update(validated)

        missing = self._compute_missing(profile)
        completed = not missing

        # Persist to artifacts
        phase_data["profile"] = profile
        phase_data["missing_fields"] = missing

        # Side-effect: persist to UserProfile
        kyc_status = await self._update_user_profile(
            db=db,
            user_id=ctx.user_id,
            kyc_collected=profile,
            completed=completed,
        )

        result = PostProcessResult(
            artifacts=artifacts,
            missing_fields=missing,
            completed=completed,
            phase_status={"kyc_status": kyc_status},
        )
        if completed:
            result.next_phase = Phase.PRE_STRATEGY.value
            result.transition_reason = "kyc_completed_to_pre_strategy"
        return result

    # -- genui ---------------------------------------------------------

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        del ctx
        if payload.get("type") != "choice_prompt":
            return payload

        choice_id_raw = payload.get("choice_id")
        choice_id = choice_id_raw.strip() if isinstance(choice_id_raw, str) else ""
        if not choice_id:
            return payload

        target_field = self._resolve_kyc_field_from_choice_id(choice_id)
        if target_field is None:
            return payload

        allowed = set(VALID_VALUES.get(target_field, set()))
        raw_options = payload.get("options")
        option_list = raw_options if isinstance(raw_options, list) else []
        filtered = [
            option
            for option in option_list
            if isinstance(option, dict)
            and isinstance(option.get("id"), str)
            and option.get("id") in allowed
        ]
        if len(filtered) < 2:
            filtered = self._build_options_for_field(target_field)

        result = dict(payload)
        result["options"] = filtered
        if not isinstance(result.get("question"), str) or not result["question"].strip():
            result["question"] = _KYC_QUESTIONS.get(
                target_field,
                "Please choose one option.",
            )
        if not isinstance(result.get("subtitle"), str) or not result["subtitle"].strip():
            subtitle = _KYC_SUBTITLES.get(target_field)
            if isinstance(subtitle, str) and subtitle.strip():
                result["subtitle"] = subtitle
        return result

    def build_fallback_choice_prompt(
        self,
        *,
        missing_fields: list[str],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        del ctx
        target_field = next(
            (field for field in missing_fields if field in REQUIRED_FIELDS),
            None,
        )
        if target_field is None:
            return None

        options = self._build_options_for_field(target_field)
        if len(options) < 2:
            return None

        payload: dict[str, Any] = {
            "type": "choice_prompt",
            "choice_id": _KYC_FIELD_CHOICE_IDS.get(target_field, target_field),
            "question": _KYC_QUESTIONS.get(target_field, "Please choose one option."),
            "options": options,
        }
        subtitle = _KYC_SUBTITLES.get(target_field)
        if isinstance(subtitle, str) and subtitle.strip():
            payload["subtitle"] = subtitle
        return payload

    # -- artifacts init ------------------------------------------------

    def init_artifacts(self) -> dict[str, Any]:
        return {"profile": {}, "missing_fields": list(REQUIRED_FIELDS)}

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        if ctx.language == "zh":
            return "接下来我们先完成 KYC 画像：我会快速确认你的交易经验、风险偏好和收益预期。"
        return (
            "Next, we will complete your KYC profile: "
            "trading experience, risk tolerance, and return expectation."
        )

    # -- helpers (public for orchestrator backward-compat) ---------------

    def build_profile_from_user_profile(self, user_profile: UserProfile | None) -> dict[str, str]:
        """Extract KYC fields from a persisted UserProfile."""
        if user_profile is None:
            return {}
        output: dict[str, str] = {}
        for field in REQUIRED_FIELDS:
            value = getattr(user_profile, field, None)
            if isinstance(value, str) and value in VALID_VALUES.get(field, set()):
                output[field] = value
        return output

    def is_profile_complete(self, user_profile: UserProfile | None) -> bool:
        """Return True if all KYC fields are present and valid."""
        if user_profile is None:
            return False
        if user_profile.kyc_status != "complete":
            return False
        profile = self.build_profile_from_user_profile(user_profile)
        return not self._compute_missing(profile)

    # -- internal helpers -----------------------------------------------

    def _compute_missing(self, profile: dict[str, Any]) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not _has_value(profile.get(f))]

    def _validate_patch(self, patch: dict[str, Any]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for field in REQUIRED_FIELDS:
            value = patch.get(field)
            if isinstance(value, str) and value in VALID_VALUES.get(field, set()):
                validated[field] = value
        return validated

    async def _update_user_profile(
        self,
        *,
        db: AsyncSession,
        user_id: UUID,
        kyc_collected: dict[str, Any],
        completed: bool,
    ) -> str:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        profile = await db.scalar(stmt)
        if profile is None:
            profile = UserProfile(
                user_id=user_id,
                trading_years_bucket=None,
                risk_tolerance=None,
                return_expectation=None,
                kyc_status="incomplete",
            )
            db.add(profile)
            await db.flush()

        for field in REQUIRED_FIELDS:
            value = kyc_collected.get(field)
            if isinstance(value, str) and value in VALID_VALUES.get(field, set()):
                setattr(profile, field, value)

        profile.kyc_status = "complete" if completed else "incomplete"
        return profile.kyc_status

    def _resolve_kyc_field_from_choice_id(self, choice_id: str) -> str | None:
        normalized = choice_id.strip().lower()
        if not normalized:
            return None
        if normalized in REQUIRED_FIELDS:
            return normalized
        if normalized.startswith("kyc_"):
            normalized = normalized[4:]
        if normalized in REQUIRED_FIELDS:
            return normalized
        return None

    def _build_options_for_field(self, field: str) -> list[dict[str, str]]:
        options = _KYC_OPTION_META.get(field)
        if not isinstance(options, dict):
            return []
        return [
            {"id": option_id, "label": label, "subtitle": subtitle}
            for option_id, (label, subtitle) in options.items()
        ]


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
