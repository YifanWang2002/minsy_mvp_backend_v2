"""Test data factories for orchestrator testing.

Provides pre-built scripts and configurations for common test scenarios.
"""

from __future__ import annotations

from typing import Any

from .scripted_user import ScriptedReply, ScriptedUser, ConditionalUser
from .observation_types import TurnObservation


# =============================================================================
# KYC Phase Scripts
# =============================================================================

def create_kyc_completion_script_en() -> list[ScriptedReply]:
    """Create a script that completes the KYC phase in English."""
    return [
        ScriptedReply(
            message="I want to create a trading strategy",
            metadata={"intent": "start_conversation"},
        ),
        ScriptedReply(
            message="I have 3-5 years of trading experience",
            metadata={"field": "trading_years_bucket", "value": "years_3_5"},
        ),
        ScriptedReply(
            message="My risk tolerance is moderate",
            metadata={"field": "risk_tolerance", "value": "moderate"},
        ),
        ScriptedReply(
            message="I expect 15-25% annual returns",
            metadata={"field": "return_expectation", "value": "return_15_25"},
        ),
    ]


def create_kyc_completion_script_zh() -> list[ScriptedReply]:
    """Create a script that completes the KYC phase in Chinese."""
    return [
        ScriptedReply(
            message="我想创建一个交易策略",
            metadata={"intent": "start_conversation"},
        ),
        ScriptedReply(
            message="我有3-5年的交易经验",
            metadata={"field": "trading_years_bucket", "value": "years_3_5"},
        ),
        ScriptedReply(
            message="我的风险承受能力是中等",
            metadata={"field": "risk_tolerance", "value": "moderate"},
        ),
        ScriptedReply(
            message="我期望15-25%的年化收益",
            metadata={"field": "return_expectation", "value": "return_15_25"},
        ),
    ]


def create_kyc_user(language: str = "en") -> ScriptedUser:
    """Create a ScriptedUser that completes KYC."""
    if language.startswith("zh"):
        return ScriptedUser(create_kyc_completion_script_zh())
    return ScriptedUser(create_kyc_completion_script_en())


# =============================================================================
# Pre-Strategy Phase Scripts
# =============================================================================

def create_pre_strategy_script_us_stocks() -> list[ScriptedReply]:
    """Create a script for US stocks pre-strategy setup."""
    return [
        ScriptedReply(
            message="I want to trade US stocks",
            metadata={"field": "target_market", "value": "us_stock"},
        ),
        ScriptedReply(
            message="I'll focus on AAPL",
            metadata={"field": "target_instrument", "value": "AAPL"},
        ),
        ScriptedReply(
            message="I want to trade daily",
            metadata={"field": "opportunity_frequency_bucket", "value": "daily"},
        ),
        ScriptedReply(
            message="I'll hold positions for a few days",
            metadata={"field": "holding_period_bucket", "value": "days"},
        ),
    ]


def create_pre_strategy_script_crypto() -> list[ScriptedReply]:
    """Create a script for crypto pre-strategy setup."""
    return [
        ScriptedReply(
            message="I want to trade cryptocurrency",
            metadata={"field": "target_market", "value": "crypto"},
        ),
        ScriptedReply(
            message="I'll trade BTC/USDT",
            metadata={"field": "target_instrument", "value": "BTC/USDT"},
        ),
        ScriptedReply(
            message="I want to trade multiple times per day",
            metadata={"field": "opportunity_frequency_bucket", "value": "intraday"},
        ),
        ScriptedReply(
            message="I'll hold for hours",
            metadata={"field": "holding_period_bucket", "value": "hours"},
        ),
    ]


# =============================================================================
# Full Workflow Scripts
# =============================================================================

def create_full_workflow_script_to_strategy() -> list[ScriptedReply]:
    """Create a script that goes from KYC through to strategy phase."""
    kyc = create_kyc_completion_script_en()
    pre_strategy = create_pre_strategy_script_us_stocks()
    return kyc + pre_strategy


def create_full_workflow_script_to_strategy_zh() -> list[ScriptedReply]:
    """Create a Chinese script that goes from KYC through to strategy phase."""
    return [
        # KYC
        ScriptedReply("我想创建一个交易策略"),
        ScriptedReply("我有3-5年的交易经验"),
        ScriptedReply("中等风险承受能力"),
        ScriptedReply("期望15-25%年化收益"),
        # Pre-Strategy
        ScriptedReply("我想交易美股"),
        ScriptedReply("我要交易苹果股票 AAPL"),
        ScriptedReply("每天交易"),
        ScriptedReply("持仓几天"),
    ]


# =============================================================================
# Conditional Users
# =============================================================================

def create_kyc_conditional_user() -> ConditionalUser:
    """Create a conditional user that responds to KYC questions."""

    def decide(turn: TurnObservation) -> ScriptedReply | None:
        text = turn.cleaned_text.lower()

        # Check for completion
        if turn.phase != "kyc":
            return None

        # Respond based on what's being asked
        if "experience" in text or "trading" in text and "years" in text:
            return ScriptedReply("3-5 years")
        if "risk" in text:
            return ScriptedReply("moderate risk tolerance")
        if "return" in text or "expect" in text:
            return ScriptedReply("15-25% annual returns")

        # Check missing fields
        if turn.missing_fields:
            field = turn.missing_fields[0]
            responses = {
                "trading_years_bucket": "3-5 years of experience",
                "risk_tolerance": "moderate risk",
                "return_expectation": "15-25% returns",
            }
            if field in responses:
                return ScriptedReply(responses[field])

        # Default continuation
        return ScriptedReply("continue")

    return ConditionalUser(decide, max_turns=10)


def create_adaptive_user(
    phase_responses: dict[str, list[str]],
    *,
    max_turns: int = 20,
) -> ConditionalUser:
    """Create a user that adapts responses based on current phase.

    Args:
        phase_responses: Dict mapping phase names to lists of responses.
            Responses are used in order for each phase.
        max_turns: Maximum turns before stopping.
    """
    phase_indices: dict[str, int] = {}

    def decide(turn: TurnObservation) -> ScriptedReply | None:
        phase = turn.phase
        if phase not in phase_responses:
            return None

        responses = phase_responses[phase]
        idx = phase_indices.get(phase, 0)

        if idx >= len(responses):
            return None

        phase_indices[phase] = idx + 1
        return ScriptedReply(responses[idx])

    return ConditionalUser(decide, max_turns=max_turns)


# =============================================================================
# Test Scenario Builders
# =============================================================================

class ScenarioBuilder:
    """Builder for creating complex test scenarios."""

    def __init__(self) -> None:
        self._replies: list[ScriptedReply] = []
        self._metadata: dict[str, Any] = {}

    def add_message(
        self,
        message: str,
        *,
        runtime_policy: dict[str, Any] | None = None,
        delay_ms: int = 0,
        **metadata: Any,
    ) -> "ScenarioBuilder":
        """Add a message to the scenario."""
        self._replies.append(
            ScriptedReply(
                message=message,
                runtime_policy=runtime_policy,
                delay_ms=delay_ms,
                metadata=metadata,
            )
        )
        return self

    def add_kyc_completion(self, language: str = "en") -> "ScenarioBuilder":
        """Add KYC completion messages."""
        if language.startswith("zh"):
            self._replies.extend(create_kyc_completion_script_zh())
        else:
            self._replies.extend(create_kyc_completion_script_en())
        return self

    def add_pre_strategy_us_stocks(self) -> "ScenarioBuilder":
        """Add pre-strategy messages for US stocks."""
        self._replies.extend(create_pre_strategy_script_us_stocks())
        return self

    def add_pre_strategy_crypto(self) -> "ScenarioBuilder":
        """Add pre-strategy messages for crypto."""
        self._replies.extend(create_pre_strategy_script_crypto())
        return self

    def with_metadata(self, **metadata: Any) -> "ScenarioBuilder":
        """Add metadata to the scenario."""
        self._metadata.update(metadata)
        return self

    def build(self) -> ScriptedUser:
        """Build the ScriptedUser."""
        return ScriptedUser(self._replies)

    def build_replies(self) -> list[ScriptedReply]:
        """Get the raw replies list."""
        return list(self._replies)


def scenario() -> ScenarioBuilder:
    """Create a new scenario builder."""
    return ScenarioBuilder()


# =============================================================================
# Quick Access Functions
# =============================================================================

def quick_kyc_user(language: str = "en") -> ScriptedUser:
    """Quick access to a KYC completion user."""
    return create_kyc_user(language)


def quick_full_workflow_user(language: str = "en") -> ScriptedUser:
    """Quick access to a full workflow user (KYC + pre-strategy)."""
    if language.startswith("zh"):
        return ScriptedUser(create_full_workflow_script_to_strategy_zh())
    return ScriptedUser(create_full_workflow_script_to_strategy())
