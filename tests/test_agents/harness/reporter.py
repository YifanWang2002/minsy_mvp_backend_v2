"""Test report generator for orchestrator observations.

Generates human-readable reports from ConversationObservation data,
useful for debugging, documentation, and test result analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .observation_types import ConversationObservation, TurnObservation


class ObservationReporter:
    """Generates test reports from conversation observations.

    Usage:
        observation = await runner.run_conversation(user)
        reporter = ObservationReporter(observation)

        # Print summary to console
        reporter.print_summary()

        # Generate full markdown report
        md = reporter.to_markdown()

        # Save to file
        reporter.save_to_file(Path("test_report.md"))
    """

    __test__ = False  # Tell pytest not to collect this class

    def __init__(self, observation: ConversationObservation) -> None:
        self._observation = observation

    def print_summary(self) -> None:
        """Print a concise summary to console."""
        obs = self._observation
        print("\n" + "=" * 60)
        print("ORCHESTRATOR TEST SUMMARY")
        print("=" * 60)
        print(f"Session ID: {obs.session_id}")
        print(f"Total Turns: {len(obs.turns)}")
        print(f"Total Tokens: {obs.total_tokens:,} (in: {obs.total_input_tokens:,}, out: {obs.total_output_tokens:,})")
        print(f"Total Duration: {obs.total_duration_ms / 1000:.2f}s")
        print(f"Phases Visited: {' -> '.join(obs.phases_visited)}")
        print(f"Final Phase: {obs.final_phase}")

        if obs.phase_transitions:
            print(f"Phase Transitions: {len(obs.phase_transitions)}")
            for from_p, to_p in obs.phase_transitions:
                print(f"  {from_p} -> {to_p}")

        if obs.errors:
            print(f"\nErrors: {len(obs.errors)}")
            for err in obs.errors:
                print(f"  Turn {err['turn']}: {err['error']}")

        print("\nTurn Summary:")
        for turn in obs.turns:
            phase_info = f"[{turn.phase}]"
            if turn.phase_stage:
                phase_info += f"/{turn.phase_stage}"
            transition = ""
            if turn.phase_transition:
                transition = f" -> {turn.phase_transition[1]}"
            tokens = f"{turn.total_tokens:,} tokens"
            latency = f"{turn.latency_ms:.0f}ms"
            print(f"  #{turn.turn_number}: {phase_info}{transition} | {tokens} | {latency}")
            print(f"       User: {turn.user_message[:50]}{'...' if len(turn.user_message) > 50 else ''}")
            print(f"       AI: {turn.cleaned_text[:50]}{'...' if len(turn.cleaned_text) > 50 else ''}")

        print("=" * 60 + "\n")

    def to_markdown(self, *, include_full_instructions: bool = False) -> str:
        """Generate a full markdown report."""
        obs = self._observation
        lines: list[str] = []

        # Header
        lines.append("# Orchestrator Test Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Session ID**: `{obs.session_id}`")
        lines.append(f"- **User ID**: `{obs.user_id}`")
        lines.append(f"- **Total Turns**: {len(obs.turns)}")
        lines.append(f"- **Total Tokens**: {obs.total_tokens:,} (input: {obs.total_input_tokens:,}, output: {obs.total_output_tokens:,})")
        lines.append(f"- **Total Duration**: {obs.total_duration_ms / 1000:.2f}s")
        lines.append(f"- **Phases Visited**: {' → '.join(obs.phases_visited)}")
        lines.append(f"- **Final Phase**: {obs.final_phase}")
        lines.append("")

        # Phase transitions
        if obs.phase_transitions:
            lines.append("### Phase Transitions")
            lines.append("")
            for i, (from_p, to_p) in enumerate(obs.phase_transitions, 1):
                lines.append(f"{i}. `{from_p}` → `{to_p}`")
            lines.append("")

        # Errors
        if obs.errors:
            lines.append("### Errors")
            lines.append("")
            for err in obs.errors:
                lines.append(f"- **Turn {err['turn']}**: {err['error']}")
            lines.append("")

        # Turn details
        lines.append("---")
        lines.append("")
        lines.append("## Turn Details")
        lines.append("")

        for turn in obs.turns:
            lines.extend(self._format_turn_markdown(turn, include_full_instructions))
            lines.append("")

        # Final artifacts
        lines.append("---")
        lines.append("")
        lines.append("## Final Artifacts")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(obs.final_artifacts, indent=2, ensure_ascii=False))
        lines.append("```")

        return "\n".join(lines)

    def _format_turn_markdown(
        self,
        turn: TurnObservation,
        include_full_instructions: bool,
    ) -> list[str]:
        """Format a single turn as markdown."""
        lines: list[str] = []

        # Turn header
        phase_info = turn.phase
        if turn.phase_stage:
            phase_info += f" / {turn.phase_stage}"
        transition = ""
        if turn.phase_transition:
            transition = f" → **{turn.phase_transition[1]}**"

        lines.append(f"### Turn {turn.turn_number} ({phase_info}){transition}")
        lines.append("")

        # Metrics
        lines.append(f"**Metrics**: {turn.total_tokens:,} tokens (in: {turn.input_tokens:,}, out: {turn.output_tokens:,}) | {turn.latency_ms:.0f}ms")
        lines.append("")

        # User message
        lines.append("#### User Message")
        lines.append("")
        lines.append(f"> {turn.user_message}")
        lines.append("")

        # What was sent to AI
        lines.append("#### Sent to AI")
        lines.append("")
        lines.append(f"- **Model**: `{turn.model}`")
        lines.append(f"- **Instructions Sent**: {'Yes' if turn.instructions_sent else 'No (reused from previous turn)'}")
        lines.append(f"- **Max Output Tokens**: {turn.max_output_tokens or 'default'}")
        lines.append(f"- **Tools**: {len(turn.tools)} tools")

        if turn.tools:
            tool_names = []
            for tool in turn.tools:
                if tool.get("type") == "mcp":
                    server = tool.get("server_label", "unknown")
                    allowed = tool.get("allowed_tools", [])
                    tool_names.append(f"{server}: {', '.join(allowed[:3])}{'...' if len(allowed) > 3 else ''}")
            if tool_names:
                lines.append("")
                for tn in tool_names:
                    lines.append(f"  - {tn}")
        lines.append("")

        # Instructions (optionally full)
        if include_full_instructions:
            lines.append("##### Instructions")
            lines.append("")
            lines.append("```")
            lines.append(turn.instructions[:2000] + ("..." if len(turn.instructions) > 2000 else ""))
            lines.append("```")
            lines.append("")

        # Enriched input
        lines.append("##### Enriched Input")
        lines.append("")
        lines.append("```")
        # Truncate for readability
        enriched = turn.enriched_input
        if len(enriched) > 500:
            enriched = enriched[:500] + "..."
        lines.append(enriched)
        lines.append("```")
        lines.append("")

        # AI Response
        lines.append("#### AI Response")
        lines.append("")

        if turn.stream_error:
            lines.append(f"**Error**: {turn.stream_error}")
            lines.append("")

        lines.append("##### Cleaned Text")
        lines.append("")
        lines.append(turn.cleaned_text or "*[empty]*")
        lines.append("")

        # GenUI payloads
        if turn.extracted_genui:
            lines.append("##### GenUI Payloads")
            lines.append("")
            for genui in turn.extracted_genui:
                genui_type = genui.get("type", "unknown")
                lines.append(f"- **{genui_type}**")
                # Show key fields based on type
                if genui_type == "choice_prompt":
                    question = genui.get("question", "")
                    lines.append(f"  - Question: {question[:100]}")
                    options = genui.get("options", [])
                    if options:
                        lines.append(f"  - Options: {len(options)} choices")
                elif genui_type == "tradingview_chart":
                    lines.append(f"  - Symbol: {genui.get('symbol')}")
            lines.append("")

        # MCP Tool Calls
        if turn.mcp_tool_calls:
            lines.append("##### MCP Tool Calls")
            lines.append("")
            for call in turn.mcp_tool_calls:
                tool_name = call.get("tool_name", call.get("name", "unknown"))
                status = call.get("status", "unknown")
                lines.append(f"- `{tool_name}` ({status})")
            lines.append("")

        # Missing fields
        if turn.missing_fields:
            lines.append(f"**Missing Fields**: {', '.join(turn.missing_fields)}")
            lines.append("")

        return lines

    def to_json(self) -> dict[str, Any]:
        """Export as JSON-serializable dictionary."""
        return self._observation.to_dict()

    def save_to_file(self, path: Path, *, format: str = "markdown") -> None:
        """Save report to file.

        Args:
            path: Output file path
            format: "markdown" or "json"
        """
        if format == "json":
            content = json.dumps(self.to_json(), indent=2, ensure_ascii=False)
        else:
            content = self.to_markdown()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def get_token_breakdown(self) -> dict[str, Any]:
        """Get detailed token usage breakdown."""
        obs = self._observation
        by_phase: dict[str, dict[str, int]] = {}

        for turn in obs.turns:
            phase = turn.phase
            if phase not in by_phase:
                by_phase[phase] = {"input": 0, "output": 0, "total": 0, "turns": 0}
            by_phase[phase]["input"] += turn.input_tokens
            by_phase[phase]["output"] += turn.output_tokens
            by_phase[phase]["total"] += turn.total_tokens
            by_phase[phase]["turns"] += 1

        return {
            "total": {
                "input": obs.total_input_tokens,
                "output": obs.total_output_tokens,
                "total": obs.total_tokens,
            },
            "by_phase": by_phase,
            "average_per_turn": {
                "input": obs.total_input_tokens / len(obs.turns) if obs.turns else 0,
                "output": obs.total_output_tokens / len(obs.turns) if obs.turns else 0,
                "total": obs.total_tokens / len(obs.turns) if obs.turns else 0,
            },
        }

    def get_latency_breakdown(self) -> dict[str, Any]:
        """Get detailed latency breakdown."""
        obs = self._observation
        latencies = [t.latency_ms for t in obs.turns]

        if not latencies:
            return {"total_ms": 0, "average_ms": 0, "min_ms": 0, "max_ms": 0}

        return {
            "total_ms": obs.total_duration_ms,
            "average_ms": obs.total_duration_ms / len(obs.turns),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "by_turn": [
                {"turn": t.turn_number, "latency_ms": t.latency_ms}
                for t in obs.turns
            ],
        }
