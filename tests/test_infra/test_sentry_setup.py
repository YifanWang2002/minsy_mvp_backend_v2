from __future__ import annotations

from src.observability.sentry_setup import sentry_before_send


def test_sentry_before_send_redacts_sensitive_headers_and_fields() -> None:
    event = {
        "request": {
            "headers": {
                "Authorization": "Bearer secret-token",
                "x-minsy-mcp-context": "context-token",
                "X-Request-Id": "req-123",
            },
            "data": {
                "access_token": "abc",
                "nested": {
                    "refreshToken": "xyz",
                    "safe_field": "safe",
                },
            },
        },
        "extra": {
            "api_key": "key-123",
            "non_sensitive": "keep",
        },
        "breadcrumbs": {
            "values": [
                {
                    "category": "http",
                    "data": {
                        "authorization": "Bearer in-breadcrumb",
                        "path": "/api/v1/chat/send-openai-stream",
                    },
                }
            ]
        },
    }

    sanitized = sentry_before_send(event, {})

    assert sanitized["request"]["headers"]["Authorization"] == "[REDACTED]"
    assert sanitized["request"]["headers"]["x-minsy-mcp-context"] == "[REDACTED]"
    assert sanitized["request"]["headers"]["X-Request-Id"] == "req-123"
    assert sanitized["request"]["data"]["access_token"] == "[REDACTED]"
    assert sanitized["request"]["data"]["nested"]["refreshToken"] == "[REDACTED]"
    assert sanitized["request"]["data"]["nested"]["safe_field"] == "safe"
    assert sanitized["extra"]["api_key"] == "[REDACTED]"
    assert sanitized["extra"]["non_sensitive"] == "keep"
    breadcrumb_data = sanitized["breadcrumbs"]["values"][0]["data"]
    assert breadcrumb_data["authorization"] == "[REDACTED]"
    assert breadcrumb_data["path"] == "/api/v1/chat/send-openai-stream"


def test_sentry_before_send_does_not_mutate_original_event() -> None:
    event = {
        "request": {
            "headers": {
                "Authorization": "Bearer not-redacted-in-source",
            }
        }
    }

    sanitized = sentry_before_send(event, {})

    assert event["request"]["headers"]["Authorization"] == "Bearer not-redacted-in-source"
    assert sanitized["request"]["headers"]["Authorization"] == "[REDACTED]"

