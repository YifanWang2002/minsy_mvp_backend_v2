from __future__ import annotations

from uuid import uuid4

from apps.api.agents.handler_protocol import PhaseContext, PromptPieces, RuntimePolicy
from apps.api.agents.phases import Phase
from apps.api.orchestration import ChatOrchestrator
from apps.api.orchestration.types import _TurnPreparation


class _FakeImage:
    def __init__(self) -> None:
        self.data = b"\x89PNG\r\n\x1a\nfake"
        self._format = "png"


def _build_preparation(*, artifacts: dict[str, object]) -> _TurnPreparation:
    prompt = PromptPieces(
        instructions="test instructions",
        enriched_input="[SESSION STATE]\n- phase: pre_strategy\n[/SESSION STATE]\n",
        tools=None,
    )
    return _TurnPreparation(
        turn_id="turn_test",
        user_message_id=uuid4(),
        phase_before=Phase.PRE_STRATEGY.value,
        phase_turn_count=1,
        prompt_user_message="test",
        handler=object(),
        artifacts=artifacts,
        pre_strategy_instrument_before="SPY",
        ctx=PhaseContext(
            user_id=uuid4(),
            session_artifacts=artifacts,
            session_id=uuid4(),
            language="zh",
            runtime_policy=RuntimePolicy(),
            turn_context={},
        ),
        prompt=prompt,
        tools=[],
    )


def test_build_openai_input_payload_attaches_pre_strategy_chart(monkeypatch) -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    def _fake_render(*, snapshot_id: str, timeframe: str, bars: int):  # noqa: ANN202
        assert snapshot_id == "snapshot-123"
        assert timeframe == "primary"
        assert bars > 0
        return [_FakeImage(), "Chart timeframe=15m. summary."]

    monkeypatch.setattr(
        "apps.mcp.domains.market_data.tools.pre_strategy_render_candlestick",
        _fake_render,
    )

    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {
                "target_market": "us_stocks",
                "target_instrument": "SPY",
                "opportunity_frequency_bucket": "daily",
                "holding_period_bucket": "intraday",
            },
            "missing_fields": ["strategy_family_choice"],
            "runtime": {
                "regime_snapshot_status": "ready",
                "regime_snapshot_id": "snapshot-123",
            },
        }
    }

    payload = orchestrator._build_openai_input_payload(
        preparation=_build_preparation(artifacts=artifacts),
    )

    assert isinstance(payload, list)
    assert payload[0]["role"] == "user"
    content = payload[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert content[0]["text"].startswith("[SESSION STATE]")
    assert content[1]["type"] == "input_text"
    assert "AUTO_GENERATED_REGIME_CHART_CONTEXT" in content[1]["text"]
    assert content[2]["type"] == "input_image"
    assert content[2]["image_url"].startswith("data:image/png;base64,")


def test_build_openai_input_payload_skips_when_not_final_choice_step(monkeypatch) -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    def _should_not_render(**kwargs):  # noqa: ANN001, ANN202
        raise AssertionError("chart render should not be called")

    monkeypatch.setattr(
        "apps.mcp.domains.market_data.tools.pre_strategy_render_candlestick",
        _should_not_render,
    )

    artifacts = {
        Phase.PRE_STRATEGY.value: {
            "profile": {},
            "missing_fields": ["holding_period_bucket", "strategy_family_choice"],
            "runtime": {
                "regime_snapshot_status": "ready",
                "regime_snapshot_id": "snapshot-123",
            },
        }
    }

    payload = orchestrator._build_openai_input_payload(
        preparation=_build_preparation(artifacts=artifacts),
    )

    assert payload is None

