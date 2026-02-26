"""Orchestrator mixin extracted from legacy implementation."""

from __future__ import annotations

from .shared import *  # noqa: F403


class FallbackMixin:
    def _maybe_apply_stop_criteria_placeholder(
        self,
        *,
        session: Session,
        phase: str,
        phase_turn_count: int,
        language: str,
        assistant_text: str,
    ) -> tuple[str, str | None]:
        if phase not in {Phase.STRATEGY.value, Phase.STRESS_TEST.value}:
            return assistant_text, None
        if phase_turn_count < _STOP_CRITERIA_TURN_LIMIT:
            return assistant_text, None

        metadata = dict(session.metadata_ or {})
        raw_alerted = metadata.get("stop_criteria_alerted_phases")
        alerted_phases = (
            {
                item.strip()
                for item in raw_alerted
                if isinstance(item, str) and item.strip()
            }
            if isinstance(raw_alerted, list)
            else set()
        )

        if phase in alerted_phases:
            return assistant_text, None

        alerted_phases.add(phase)
        metadata["stop_criteria_alerted_phases"] = sorted(alerted_phases)
        metadata["stop_criteria_placeholder"] = {
            "enabled": True,
            "max_turns_per_phase": _STOP_CRITERIA_TURN_LIMIT,
            "performance_threshold_todo": True,
            "last_triggered_phase": phase,
            "last_triggered_at": datetime.now(UTC).isoformat(),
        }
        session.metadata_ = metadata

        if language == "zh":
            hint = (
                "提示：当前策略迭代轮次已较多。你可以考虑更换策略方向或重置一次。"
                "（占位逻辑：后续将接入真实绩效阈值判断）"
            )
        else:
            hint = (
                "Hint: this strategy has gone through many iterations. "
                "Consider trying a new strategy direction or restarting once. "
                "(Placeholder logic: real performance-threshold checks will be added later.)"
            )

        if not assistant_text.strip():
            return hint, hint
        return f"{assistant_text}\n\n{hint}", f"\n\n{hint}"

    @staticmethod
    def _build_empty_turn_fallback_text(
        *,
        phase: str,
        missing_fields: list[str],
        language: str,
    ) -> str:
        field = missing_fields[0] if missing_fields else ""
        is_zh = isinstance(language, str) and language.strip().lower().startswith("zh")

        if is_zh:
            prompts = {
                "trading_years_bucket": "我这轮没有收到可显示的回复。请告诉我你的交易经验年限（例如：5年以上）。",
                "risk_tolerance": "我这轮没有收到可显示的回复。请告诉我你的风险偏好（保守/中等/激进/非常激进）。",
                "return_expectation": "我这轮没有收到可显示的回复。请告诉我你的收益预期（保本/平衡增长/增长/高增长）。",
                "target_market": "我这轮没有收到可显示的回复。请告诉我你想交易的市场（如：美股、加密、外汇、期货）。",
                "target_instrument": "我这轮没有收到可显示的回复。请告诉我你想交易的标的（例如：SPY 或 GBPUSD）。",
                "opportunity_frequency_bucket": "我这轮没有收到可显示的回复。请告诉我你期望的机会频率（每月少量/每周几次/每日/每日多次）。",
                "holding_period_bucket": "我这轮没有收到可显示的回复。请告诉我你的持仓周期（超短线/日内/波段数天/持有数周以上）。",
            }
            if field in prompts:
                return prompts[field]
            return "我这轮没有收到可显示的回复。请再发送一次你的答案，我们继续。"

        prompts = {
            "trading_years_bucket": "I did not receive a displayable reply this turn. Please share your trading experience bucket (for example: 5+ years).",
            "risk_tolerance": "I did not receive a displayable reply this turn. Please share your risk tolerance (conservative/moderate/aggressive/very aggressive).",
            "return_expectation": "I did not receive a displayable reply this turn. Please share your return expectation (capital preservation/balanced growth/growth/high growth).",
            "target_market": "I did not receive a displayable reply this turn. Please tell me your target market (us_stocks/crypto/forex/futures).",
            "target_instrument": "I did not receive a displayable reply this turn. Please tell me your target instrument (for example: SPY or GBPUSD).",
            "opportunity_frequency_bucket": "I did not receive a displayable reply this turn. Please share your opportunity frequency (few_per_month/few_per_week/daily/multiple_per_day).",
            "holding_period_bucket": "I did not receive a displayable reply this turn. Please share your holding period bucket (intraday_scalp/intraday/swing_days/position_weeks_plus).",
        }
        if field in prompts:
            return prompts[field]
        if phase == Phase.STRATEGY.value:
            return "I did not receive a displayable reply this turn. Please restate your strategy request and I will continue."
        return "I did not receive a displayable reply this turn. Please resend your answer and we can continue."
