from __future__ import annotations

from apps.api.orchestration import ChatOrchestrator


def test_redact_stream_request_kwargs_redacts_input_payload_data_url() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    data_url = "data:image/png;base64," + ("A" * 2048)

    redacted = orchestrator._redact_stream_request_kwargs_for_trace(
        {
            "model": "gpt-5.2",
            "input_text": "plain text input",
            "input_payload": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        }
    )

    payload = redacted["input_payload"][0]["content"][1]
    assert payload["type"] == "input_image"
    assert isinstance(payload["image_url"], str)
    assert payload["image_url"].startswith("[redacted_data_url len=")
    assert "AAAA" not in payload["image_url"]

