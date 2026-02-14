from __future__ import annotations

import httpx
from openai import APIConnectionError, APIStatusError

from src.services.openai_stream_service import (
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
