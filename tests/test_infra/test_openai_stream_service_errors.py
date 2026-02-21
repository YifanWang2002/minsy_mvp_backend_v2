from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from src.services.openai_stream_service import (
    OpenAIResponsesEventStreamer,
    _build_stream_error_diagnostics,
    _build_stream_error_payload,
    _is_retryable_mcp_listing_error,
)


def _build_api_status_error(
    *,
    status_code: int,
    body: object,
    request_id: str = "req_test_123",
) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        status_code,
        request=request,
        headers={"x-request-id": request_id},
    )
    return APIStatusError("Error code: 424", response=response, body=body)


def test_mcp_list_tools_diagnostics_expose_root_cause() -> None:
    exc = _build_api_status_error(
        status_code=424,
        body={
            "error": {
                "message": (
                    "Error retrieving tool list from mcp server: "
                    "dial tcp 1.2.3.4:443: i/o timeout"
                ),
                "code": "MCP_LIST_TOOLS_FAILED",
                "type": "mcp_error",
            }
        },
    )

    diagnostics = _build_stream_error_diagnostics(exc)

    assert diagnostics["category"] == "mcp_list_tools_fetch_failed"
    assert diagnostics["status_code"] == 424
    assert diagnostics["request_id"] == "req_test_123"
    assert diagnostics["upstream_error_code"] == "MCP_LIST_TOOLS_FAILED"
    assert diagnostics["upstream_error_type"] == "mcp_error"
    assert "tool list" in diagnostics["upstream_message"].lower()
    assert _is_retryable_mcp_listing_error(exc, diagnostics=diagnostics)

    payload = _build_stream_error_payload(
        exc=exc,
        retryable=True,
        attempt=3,
        diagnostics=diagnostics,
    )
    assert payload["class"] == "APIStatusError"
    assert payload["retryable"] is True
    assert payload["attempt"] == 3
    assert "tool list" in payload["message"].lower()
    assert payload["diagnostics"]["category"] == "mcp_list_tools_fetch_failed"


def test_non_mcp_424_is_not_classified_as_list_tools_error() -> None:
    exc = _build_api_status_error(
        status_code=424,
        body={
            "error": {
                "message": "Payment dependency failed.",
                "code": "PAYMENT_FAILED",
                "type": "business_error",
            }
        },
    )

    diagnostics = _build_stream_error_diagnostics(exc)

    assert diagnostics["category"] == "upstream_client_error"
    assert not _is_retryable_mcp_listing_error(exc, diagnostics=diagnostics)


def test_network_error_is_classified_as_network() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    exc = APIConnectionError(request=request)

    diagnostics = _build_stream_error_diagnostics(exc)

    assert diagnostics["category"] == "network"
    assert not _is_retryable_mcp_listing_error(exc, diagnostics=diagnostics)


@pytest.mark.asyncio
async def test_stream_error_reports_to_sentry_without_changing_payload() -> None:
    exc = _build_api_status_error(
        status_code=424,
        body={
            "error": {
                "message": "Payment dependency failed.",
                "code": "PAYMENT_FAILED",
                "type": "business_error",
            }
        },
        request_id="req_payload_unchanged",
    )

    class _FailingContext:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
            return False

    class _FakeResponses:
        @staticmethod
        def stream(**kwargs):  # noqa: ANN003
            return _FailingContext()

    streamer = OpenAIResponsesEventStreamer.__new__(OpenAIResponsesEventStreamer)
    streamer.client = type("_FakeClient", (), {"responses": _FakeResponses()})()

    with patch("src.services.openai_stream_service.capture_exception_with_context") as mocked_capture:
        events = [
            item
            async for item in streamer.stream_events(
                model="gpt-5.2",
                input_text="hello",
            )
        ]

    assert len(events) == 1
    assert events[0]["type"] == "response.stream_error"
    assert events[0]["error"]["class"] == "APIStatusError"
    assert events[0]["error"]["diagnostics"]["category"] == "upstream_client_error"
    assert events[0]["error"]["diagnostics"]["request_id"] == "req_payload_unchanged"

    mocked_capture.assert_called_once()
    called_exc = mocked_capture.call_args.args[0]
    called_tags = mocked_capture.call_args.kwargs["tags"]
    assert isinstance(called_exc, APIStatusError)
    assert called_tags["category"] == "upstream_client_error"
    assert called_tags["status_code"] == 424
    assert called_tags["request_id"] == "req_payload_unchanged"
