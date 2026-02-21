"""Capture non-2xx FastAPI HTTP responses into Sentry with request context."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.config import settings
from src.observability.sentry_setup import capture_message_with_context


def _setting_bool(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def _setting_int(name: str, default: int) -> int:
    value = getattr(settings, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _setting_str_list(name: str, default: list[str]) -> list[str]:
    value = getattr(settings, name, default)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return []
        return [item.strip() for item in normalized.split(",") if item.strip()]
    return list(default)


def _status_family(status_code: int) -> str:
    if status_code < 100:
        return "unknown"
    return f"{status_code // 100}xx"


def _normalize_path(path: str) -> str:
    normalized = path.strip() or "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def _is_excluded_path(path: str) -> bool:
    normalized_path = _normalize_path(path)
    for raw in _setting_str_list(
        "sentry_http_status_exclude_paths",
        ["/api/v1/health", "/api/v1/status"],
    ):
        candidate = _normalize_path(raw)
        if normalized_path == candidate:
            return True
        if candidate != "/" and normalized_path.startswith(f"{candidate}/"):
            return True
    return False


def should_capture_http_status_event(
    *,
    status_code: int,
    path: str,
    method: str,
) -> bool:
    """Return whether one response should be recorded in Sentry."""
    if not _setting_bool("sentry_http_status_capture_enabled", True):
        return False
    if method.strip().upper() == "OPTIONS":
        return False
    if status_code < _setting_int("sentry_http_status_min_code", 400):
        return False
    if status_code > _setting_int("sentry_http_status_max_code", 599):
        return False
    return not _is_excluded_path(path)


def _resolve_route_template(request: Request) -> str:
    route = request.scope.get("route")
    for attr in ("path_format", "path"):
        value = getattr(route, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return request.url.path


def _serialize_query_params(request: Request) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in request.query_params.keys():
        values = request.query_params.getlist(key)
        if len(values) <= 1:
            payload[key] = values[0] if values else ""
        else:
            payload[key] = values
    return payload


class SentryHTTPStatusMiddleware(BaseHTTPMiddleware):
    """Send one Sentry message when an HTTP response matches configured status range."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        started_at = perf_counter()
        response = await call_next(request)

        status_code = int(response.status_code)
        method = request.method.upper()
        path = request.url.path
        if not should_capture_http_status_event(
            status_code=status_code,
            path=path,
            method=method,
        ):
            return response

        route_template = _resolve_route_template(request)
        status_family = _status_family(status_code)
        duration_ms = round((perf_counter() - started_at) * 1000.0, 2)
        level = "error" if status_code >= 500 else "warning"

        capture_message_with_context(
            f"FastAPI returned HTTP {status_code}",
            level=level,
            tags={
                "source": "fastapi_http_status",
                "status_code": status_code,
                "status_family": status_family,
                "method": method,
                "route": route_template,
            },
            extras={
                "method": method,
                "path": path,
                "route_template": route_template,
                "query_params": _serialize_query_params(request),
                "status_code": status_code,
                "status_family": status_family,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "referer": request.headers.get("referer"),
                "request_id": request.headers.get("x-request-id"),
                "correlation_id": request.headers.get("x-correlation-id"),
                "trace_id": request.headers.get("x-minsy-debug-trace-id"),
                "response_request_id": response.headers.get("x-request-id"),
            },
            fingerprint=[
                "fastapi-http-status",
                method,
                route_template,
                str(status_code),
            ],
        )
        return response
