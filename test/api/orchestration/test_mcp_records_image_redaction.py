from __future__ import annotations

from apps.api.orchestration import ChatOrchestrator


def test_mcp_records_redacts_pre_strategy_chart_base64_output() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]

    records = {
        "call_1": {
            "name": "pre_strategy_render_candlestick",
            "status": "success",
            "arguments": '{"snapshot_id":"abc"}',
            "output": [
                {
                    "type": "image",
                    "format": "png",
                    "data": "A" * 5000,
                },
                "alt text",
            ],
        }
    }

    persisted = orchestrator._build_persistable_mcp_tool_calls(
        records=records,
        order=["call_1"],
    )

    assert len(persisted) == 1
    output = persisted[0]["output"]
    assert isinstance(output, list)
    image_payload = output[0]
    assert isinstance(image_payload, dict)
    assert image_payload.get("type") == "image"
    assert "data" not in image_payload
    assert image_payload.get("data_redacted") is True
    assert isinstance(image_payload.get("data_sha256"), str)
