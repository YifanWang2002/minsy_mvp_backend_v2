#!/usr/bin/env python
"""Unit tests for the orchestrator test harness components.

These tests verify the harness infrastructure without making real API calls.
Run with: uv run pytest tests/test_agents/harness/test_harness_unit.py -v
"""

from __future__ import annotations

import pytest
from datetime import datetime, UTC
from uuid import uuid4

from tests.test_agents.harness import (
    TurnObservation,
    ConversationObservation,
    TurnObserver,
    ScriptedReply,
    ScriptedUser,
    ConditionalUser,
    TestReporter,
)
from tests.test_agents.harness.factories import (
    create_kyc_user,
    scenario,
    ScenarioBuilder,
)


class TestScriptedUser:
    """Tests for ScriptedUser."""

    def test_sequential_replies(self):
        """Test that replies are returned in order."""
        user = ScriptedUser([
            ScriptedReply("Hello"),
            ScriptedReply("World"),
        ])

        assert user.next_reply().message == "Hello"
        assert user.next_reply().message == "World"
        assert user.next_reply() is None

    def test_remaining_count(self):
        """Test remaining count tracking."""
        user = ScriptedUser([
            ScriptedReply("A"),
            ScriptedReply("B"),
            ScriptedReply("C"),
        ])

        assert user.remaining_count == 3
        user.next_reply()
        assert user.remaining_count == 2
        user.next_reply()
        assert user.remaining_count == 1
        user.next_reply()
        assert user.remaining_count == 0

    def test_inject_reply(self):
        """Test injecting a reply at current position."""
        user = ScriptedUser([
            ScriptedReply("A"),
            ScriptedReply("C"),
        ])

        user.next_reply()  # Consume A
        user.inject_reply(ScriptedReply("B"))

        assert user.next_reply().message == "B"
        assert user.next_reply().message == "C"

    def test_reset(self):
        """Test resetting the user."""
        user = ScriptedUser([
            ScriptedReply("A"),
            ScriptedReply("B"),
        ])

        user.next_reply()
        user.next_reply()
        assert user.is_exhausted

        user.reset()
        assert not user.is_exhausted
        assert user.next_reply().message == "A"


class TestConditionalUser:
    """Tests for ConditionalUser."""

    def test_decision_function(self):
        """Test that decision function is called correctly."""
        decisions = []

        def decide(turn):
            decisions.append(turn.phase)
            if turn.phase == "kyc":
                return ScriptedReply("KYC response")
            return None

        user = ConditionalUser(decide)

        # Create a mock turn observation
        turn = TurnObservation(
            turn_number=1,
            timestamp=datetime.now(UTC),
            user_message="test",
            phase="kyc",
            phase_stage=None,
            phase_turn_count=1,
            session_state_snapshot={},
            instructions="",
            instructions_sent=True,
            enriched_input="",
            tools=[],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=None,
            reasoning_config=None,
            raw_response_text="",
            cleaned_text="",
            extracted_patches=[],
            extracted_genui=[],
            mcp_tool_calls=[],
            artifacts_before={},
            artifacts_after={},
            missing_fields=[],
            phase_transition=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            stream_start_time=None,
            stream_end_time=None,
            stream_error=None,
            stream_error_detail=None,
            response_id=None,
            assistant_message_id=None,
        )

        reply = user.decide_reply(turn)
        assert reply is not None
        assert reply.message == "KYC response"
        assert decisions == ["kyc"]

    def test_max_turns_limit(self):
        """Test that max_turns limit is enforced."""
        user = ConditionalUser(
            lambda t: ScriptedReply("continue"),
            max_turns=3,
        )

        turn = TurnObservation(
            turn_number=1,
            timestamp=datetime.now(UTC),
            user_message="test",
            phase="kyc",
            phase_stage=None,
            phase_turn_count=1,
            session_state_snapshot={},
            instructions="",
            instructions_sent=True,
            enriched_input="",
            tools=[],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=None,
            reasoning_config=None,
            raw_response_text="",
            cleaned_text="",
            extracted_patches=[],
            extracted_genui=[],
            mcp_tool_calls=[],
            artifacts_before={},
            artifacts_after={},
            missing_fields=[],
            phase_transition=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            stream_start_time=None,
            stream_end_time=None,
            stream_error=None,
            stream_error_detail=None,
            response_id=None,
            assistant_message_id=None,
        )

        assert user.decide_reply(turn) is not None
        assert user.decide_reply(turn) is not None
        assert user.decide_reply(turn) is not None
        assert user.decide_reply(turn) is None  # Exceeded max_turns


class TestTurnObserver:
    """Tests for TurnObserver."""

    def test_turn_counting(self):
        """Test that turns are counted correctly."""
        observer = TurnObserver(uuid4(), uuid4())

        assert observer.current_turn_number == 0

        observer.start_turn("Hello", {})
        assert observer.current_turn_number == 1

        observer.start_turn("World", {})
        assert observer.current_turn_number == 2

    def test_capture_preparation(self):
        """Test capturing preparation data."""
        observer = TurnObserver(uuid4(), uuid4())
        observer.start_turn("Hello", {"key": "value"})

        observer.capture_preparation(
            turn_id="turn_1",
            user_message_id=uuid4(),
            phase_before="kyc",
            phase_turn_count=1,
            prompt_user_message="Hello",
            artifacts={"key": "value"},
            instructions="System instructions",
            enriched_input="[STATE] Hello",
            tools=[{"name": "tool1"}],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=1000,
            reasoning=None,
            phase_stage=None,
        )

        # Preparation should be captured (internal state)
        assert observer._current_preparation is not None
        assert observer._current_preparation.phase_before == "kyc"


class TestConversationObservation:
    """Tests for ConversationObservation."""

    def test_add_turn(self):
        """Test adding turns to observation."""
        obs = ConversationObservation(
            session_id=uuid4(),
            user_id=uuid4(),
            started_at=datetime.now(UTC),
        )

        turn = TurnObservation(
            turn_number=1,
            timestamp=datetime.now(UTC),
            user_message="test",
            phase="kyc",
            phase_stage=None,
            phase_turn_count=1,
            session_state_snapshot={},
            instructions="",
            instructions_sent=True,
            enriched_input="",
            tools=[],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=None,
            reasoning_config=None,
            raw_response_text="",
            cleaned_text="response",
            extracted_patches=[],
            extracted_genui=[],
            mcp_tool_calls=[],
            artifacts_before={},
            artifacts_after={},
            missing_fields=[],
            phase_transition=None,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000,
            stream_start_time=None,
            stream_end_time=None,
            stream_error=None,
            stream_error_detail=None,
            response_id=None,
            assistant_message_id=None,
        )

        obs.add_turn(turn)

        assert len(obs.turns) == 1
        assert obs.total_tokens == 150
        assert "kyc" in obs.phases_visited

    def test_phase_transitions_tracking(self):
        """Test that phase transitions are tracked."""
        obs = ConversationObservation(
            session_id=uuid4(),
            user_id=uuid4(),
            started_at=datetime.now(UTC),
        )

        turn = TurnObservation(
            turn_number=1,
            timestamp=datetime.now(UTC),
            user_message="test",
            phase="kyc",
            phase_stage=None,
            phase_turn_count=1,
            session_state_snapshot={},
            instructions="",
            instructions_sent=True,
            enriched_input="",
            tools=[],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=None,
            reasoning_config=None,
            raw_response_text="",
            cleaned_text="",
            extracted_patches=[],
            extracted_genui=[],
            mcp_tool_calls=[],
            artifacts_before={},
            artifacts_after={},
            missing_fields=[],
            phase_transition=("kyc", "pre_strategy"),
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            stream_start_time=None,
            stream_end_time=None,
            stream_error=None,
            stream_error_detail=None,
            response_id=None,
            assistant_message_id=None,
        )

        obs.add_turn(turn)

        assert len(obs.phase_transitions) == 1
        assert obs.phase_transitions[0] == ("kyc", "pre_strategy")


class TestScenarioBuilder:
    """Tests for ScenarioBuilder."""

    def test_add_message(self):
        """Test adding individual messages."""
        user = (
            scenario()
            .add_message("Hello")
            .add_message("World")
            .build()
        )

        assert user.total_count == 2
        assert user.next_reply().message == "Hello"
        assert user.next_reply().message == "World"

    def test_add_kyc_completion(self):
        """Test adding KYC completion script."""
        user = scenario().add_kyc_completion("en").build()

        assert user.total_count == 4  # KYC has 4 messages

    def test_chaining(self):
        """Test method chaining."""
        user = (
            scenario()
            .add_message("Start")
            .add_kyc_completion("en")
            .add_message("End")
            .build()
        )

        assert user.total_count == 6  # 1 + 4 + 1


class TestFactories:
    """Tests for factory functions."""

    def test_create_kyc_user_en(self):
        """Test creating English KYC user."""
        user = create_kyc_user("en")
        assert user.total_count == 4

    def test_create_kyc_user_zh(self):
        """Test creating Chinese KYC user."""
        user = create_kyc_user("zh")
        assert user.total_count == 4


class TestTestReporter:
    """Tests for TestReporter."""

    def test_markdown_generation(self):
        """Test markdown report generation."""
        obs = ConversationObservation(
            session_id=uuid4(),
            user_id=uuid4(),
            started_at=datetime.now(UTC),
        )

        turn = TurnObservation(
            turn_number=1,
            timestamp=datetime.now(UTC),
            user_message="Hello",
            phase="kyc",
            phase_stage=None,
            phase_turn_count=1,
            session_state_snapshot={},
            instructions="System prompt",
            instructions_sent=True,
            enriched_input="[STATE] Hello",
            tools=[],
            tool_choice=None,
            model="gpt-4",
            max_output_tokens=None,
            reasoning_config=None,
            raw_response_text="Hi there!",
            cleaned_text="Hi there!",
            extracted_patches=[],
            extracted_genui=[],
            mcp_tool_calls=[],
            artifacts_before={},
            artifacts_after={},
            missing_fields=[],
            phase_transition=None,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=1000,
            stream_start_time=None,
            stream_end_time=None,
            stream_error=None,
            stream_error_detail=None,
            response_id=None,
            assistant_message_id=None,
        )

        obs.add_turn(turn)
        obs.finalize(datetime.now(UTC))

        reporter = TestReporter(obs)
        md = reporter.to_markdown()

        assert "# Orchestrator Test Report" in md
        assert "## Summary" in md
        assert "Turn 1" in md
        assert "Hello" in md

    def test_token_breakdown(self):
        """Test token breakdown calculation."""
        obs = ConversationObservation(
            session_id=uuid4(),
            user_id=uuid4(),
            started_at=datetime.now(UTC),
        )

        for i in range(3):
            turn = TurnObservation(
                turn_number=i + 1,
                timestamp=datetime.now(UTC),
                user_message=f"Message {i}",
                phase="kyc",
                phase_stage=None,
                phase_turn_count=i + 1,
                session_state_snapshot={},
                instructions="",
                instructions_sent=True,
                enriched_input="",
                tools=[],
                tool_choice=None,
                model="gpt-4",
                max_output_tokens=None,
                reasoning_config=None,
                raw_response_text="",
                cleaned_text="",
                extracted_patches=[],
                extracted_genui=[],
                mcp_tool_calls=[],
                artifacts_before={},
                artifacts_after={},
                missing_fields=[],
                phase_transition=None,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                latency_ms=1000,
                stream_start_time=None,
                stream_end_time=None,
                stream_error=None,
                stream_error_detail=None,
                response_id=None,
                assistant_message_id=None,
            )
            obs.add_turn(turn)

        obs.finalize(datetime.now(UTC))

        reporter = TestReporter(obs)
        breakdown = reporter.get_token_breakdown()

        assert breakdown["total"]["total"] == 450
        assert breakdown["by_phase"]["kyc"]["total"] == 450
        assert breakdown["by_phase"]["kyc"]["turns"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
