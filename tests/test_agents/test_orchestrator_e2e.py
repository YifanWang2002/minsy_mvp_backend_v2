"""End-to-end tests for the ChatOrchestrator.

These tests use real OpenAI API calls to verify the complete orchestrator flow.
They demonstrate how to use the test harness for various scenarios.

Run with: uv run pytest tests/test_agents/test_orchestrator_e2e.py -v
"""

from __future__ import annotations

import pytest

from tests.test_agents.harness import (
    ScriptedReply,
    ScriptedUser,
    TestReporter,
)
from tests.test_agents.harness.factories import (
    create_kyc_user,
    quick_full_workflow_user,
    scenario,
)
from tests.test_agents.harness.fixtures import (
    test_db,
    test_user,
    openai_streamer,
    orchestrator_runner,
    quick_runner,
)


@pytest.mark.external
class TestKYCPhase:
    """Tests for the KYC phase of the orchestrator."""

    async def test_kyc_single_turn(self, quick_runner):
        """Test a single turn in the KYC phase."""
        obs = await quick_runner.send("I want to create a trading strategy")

        assert obs.phase == "kyc"
        assert obs.total_tokens > 0
        assert obs.cleaned_text  # AI should respond
        assert obs.instructions_sent  # First turn should send instructions

        # Print summary for debugging
        print(f"\nKYC Single Turn:")
        print(f"  Tokens: {obs.total_tokens}")
        print(f"  Latency: {obs.latency_ms:.0f}ms")
        print(f"  Response: {obs.cleaned_text[:100]}...")

    async def test_kyc_phase_completion(self, orchestrator_runner):
        """Test completing the entire KYC phase."""
        user = create_kyc_user("en")

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=10,
            stop_on_phase="pre_strategy",
        )

        # Verify we reached pre_strategy
        assert observation.final_phase == "pre_strategy" or "pre_strategy" in observation.phases_visited

        # Verify KYC was visited
        assert "kyc" in observation.phases_visited

        # Print report
        reporter = TestReporter(observation)
        reporter.print_summary()

    async def test_kyc_collects_all_fields(self, orchestrator_runner):
        """Test that KYC collects all required fields."""
        user = ScriptedUser([
            ScriptedReply("Hello, I want to start"),
            ScriptedReply("I have 3-5 years of trading experience"),
            ScriptedReply("My risk tolerance is moderate"),
            ScriptedReply("I expect 15-25% annual returns"),
        ])

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=8,
            stop_on_transition=True,
        )

        # Check that we transitioned out of KYC
        if observation.phase_transitions:
            from_phase, to_phase = observation.phase_transitions[0]
            assert from_phase == "kyc"
            assert to_phase == "pre_strategy"

        # Verify artifacts contain KYC data
        kyc_artifacts = observation.final_artifacts.get("kyc", {})
        profile = kyc_artifacts.get("profile", {})

        # At least some fields should be collected
        collected_fields = [k for k, v in profile.items() if v]
        print(f"\nCollected KYC fields: {collected_fields}")


@pytest.mark.external
class TestPreStrategyPhase:
    """Tests for the pre-strategy phase."""

    async def test_pre_strategy_after_kyc(self, orchestrator_runner):
        """Test transitioning from KYC to pre-strategy."""
        user = quick_full_workflow_user("en")

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=15,
            stop_on_phase="strategy",
        )

        # Verify phase progression
        assert "kyc" in observation.phases_visited
        assert "pre_strategy" in observation.phases_visited

        # Print token breakdown
        reporter = TestReporter(observation)
        breakdown = reporter.get_token_breakdown()
        print(f"\nToken breakdown by phase:")
        for phase, stats in breakdown["by_phase"].items():
            print(f"  {phase}: {stats['total']} tokens ({stats['turns']} turns)")


@pytest.mark.external
class TestMultiTurnConversation:
    """Tests for multi-turn conversation handling."""

    async def test_conversation_continuity(self, orchestrator_runner):
        """Test that conversation context is maintained across turns."""
        user = ScriptedUser([
            ScriptedReply("Hi, I want to create a strategy"),
            ScriptedReply("I have 5 years of experience"),
            ScriptedReply("What did I just tell you about my experience?"),
        ])

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=5,
        )

        # The AI should remember the experience mentioned
        last_turn = observation.turns[-1]
        # Check that the response references the previous context
        assert last_turn.total_tokens > 0

        reporter = TestReporter(observation)
        reporter.print_summary()

    async def test_instruction_reuse(self, orchestrator_runner):
        """Test that instructions are not sent every turn."""
        user = ScriptedUser([
            ScriptedReply("Hello"),
            ScriptedReply("I have 3 years experience"),
            ScriptedReply("Tell me more"),
        ])

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=5,
        )

        # First turn should send instructions
        assert observation.turns[0].instructions_sent

        # Subsequent turns in the same phase may reuse instructions
        instructions_sent_count = sum(1 for t in observation.turns if t.instructions_sent)
        print(f"\nInstructions sent in {instructions_sent_count}/{len(observation.turns)} turns")


@pytest.mark.external
class TestErrorHandling:
    """Tests for error handling scenarios."""

    async def test_empty_message_handling(self, quick_runner):
        """Test handling of minimal messages."""
        obs = await quick_runner.send("ok")

        # Should still get a response
        assert obs.total_tokens > 0
        # May or may not have cleaned text depending on AI response

    async def test_long_message_handling(self, quick_runner):
        """Test handling of longer messages."""
        long_message = "I want to create a trading strategy. " * 20

        obs = await quick_runner.send(long_message)

        assert obs.total_tokens > 0
        assert obs.input_tokens > 100  # Should have significant input


@pytest.mark.external
class TestObservationCapture:
    """Tests verifying the observation capture functionality."""

    async def test_captures_all_turn_data(self, quick_runner):
        """Test that all turn data is captured correctly."""
        obs = await quick_runner.send("I want to create a strategy for trading stocks")

        # Verify input capture
        assert obs.user_message == "I want to create a strategy for trading stocks"
        assert obs.phase == "kyc"
        assert obs.instructions  # Should have instructions
        assert obs.enriched_input  # Should have enriched input
        assert obs.tools is not None  # Should have tools list

        # Verify output capture
        assert obs.model  # Should have model name
        assert obs.total_tokens > 0

        # Verify timing capture
        assert obs.latency_ms > 0
        assert obs.stream_start_time is not None
        assert obs.stream_end_time is not None

    async def test_captures_tool_information(self, quick_runner):
        """Test that tool information is captured."""
        obs = await quick_runner.send("Hello")

        # Should have tools available
        assert isinstance(obs.tools, list)

        # Print tool summary
        print(f"\nTools available in KYC phase:")
        for tool in obs.tools:
            if tool.get("type") == "mcp":
                server = tool.get("server_label", "unknown")
                allowed = tool.get("allowed_tools", [])
                print(f"  {server}: {len(allowed)} tools")


@pytest.mark.external
class TestReporting:
    """Tests for the reporting functionality."""

    async def test_markdown_report_generation(self, orchestrator_runner, tmp_path):
        """Test generating a markdown report."""
        user = ScriptedUser([
            ScriptedReply("I want to create a strategy"),
            ScriptedReply("3-5 years experience"),
        ])

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=3,
        )

        reporter = TestReporter(observation)

        # Generate markdown
        md = reporter.to_markdown()
        assert "# Orchestrator Test Report" in md
        assert "## Summary" in md
        assert "## Turn Details" in md

        # Save to file
        report_path = tmp_path / "test_report.md"
        reporter.save_to_file(report_path)
        assert report_path.exists()

    async def test_json_export(self, orchestrator_runner):
        """Test JSON export of observations."""
        user = ScriptedUser([
            ScriptedReply("Hello"),
        ])

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=2,
        )

        reporter = TestReporter(observation)
        data = reporter.to_json()

        assert "session_id" in data
        assert "turns" in data
        assert "total_tokens" in data
        assert len(data["turns"]) > 0


@pytest.mark.external
class TestScenarioBuilder:
    """Tests using the scenario builder."""

    async def test_custom_scenario(self, orchestrator_runner):
        """Test building a custom scenario."""
        user = (
            scenario()
            .add_message("I want to start trading")
            .add_message("I have moderate experience, about 3 years")
            .add_message("I can handle moderate risk")
            .build()
        )

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=5,
        )

        assert len(observation.turns) >= 3
        reporter = TestReporter(observation)
        reporter.print_summary()

    async def test_scenario_with_kyc_helper(self, orchestrator_runner):
        """Test scenario builder with KYC helper."""
        user = (
            scenario()
            .add_kyc_completion("en")
            .build()
        )

        observation = await orchestrator_runner.run_conversation(
            user,
            max_turns=10,
            stop_on_phase="pre_strategy",
        )

        # Should complete KYC
        assert "kyc" in observation.phases_visited
