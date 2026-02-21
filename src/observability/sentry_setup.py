"""Backend Sentry bootstrap and event sanitization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from src.config import settings
from src.util.logger import logger

SentrySource = Literal["fastapi", "celery", "mcp", "backend"]
_REDACTED_VALUE = "[REDACTED]"
_SANITIZE_MAX_DEPTH = 8

_SENSITIVE_KEYWORDS = {
    "apikey",
    "authorization",
    "cookie",
    "mcpcontext",
    "password",
    "secret",
    "setcookie",
    "signature",
    "token",
}

_BUILTIN_SENSITIVE_FIELD_PATHS = (
    ("request", "headers"),
    ("request", "env"),
    ("request", "data"),
    ("contexts", "request"),
    ("extra",),
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _is_sensitive_key(raw_key: str) -> bool:
    key = _normalize_key(raw_key)
    if not key:
        return False
    if key in _SENSITIVE_KEYWORDS:
        return True
    if "authorization" in key or "mcpcontext" in key:
        return True
    if key.endswith("token") or key.endswith("secret") or key.endswith("password"):
        return True
    if key.endswith("apikey") or key.endswith("keyid"):
        return True
    return False


def _sanitize_value(value: Any, *, depth: int = 0, key_hint: str | None = None) -> Any:
    if depth >= _SANITIZE_MAX_DEPTH:
        return str(value)

    if key_hint and _is_sensitive_key(key_hint):
        return _REDACTED_VALUE

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _is_sensitive_key(key_str):
                sanitized[key_str] = _REDACTED_VALUE
            else:
                sanitized[key_str] = _sanitize_value(
                    item,
                    depth=depth + 1,
                    key_hint=key_str,
                )
        return sanitized

    if isinstance(value, list):
        return [
            _sanitize_value(item, depth=depth + 1, key_hint=key_hint)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _sanitize_value(item, depth=depth + 1, key_hint=key_hint)
            for item in value
        )

    return value


def _sanitize_path(event: dict[str, Any], path: tuple[str, ...]) -> None:
    target: Any = event
    for segment in path:
        if not isinstance(target, dict):
            return
        target = target.get(segment)
    if isinstance(target, dict):
        sanitized = _sanitize_value(target)
        parent: Any = event
        for segment in path[:-1]:
            if not isinstance(parent, dict):
                return
            parent = parent.get(segment)
        if isinstance(parent, dict):
            parent[path[-1]] = sanitized
    elif target is not None:
        parent = event
        for segment in path[:-1]:
            if not isinstance(parent, dict):
                return
            parent = parent.get(segment)
        if isinstance(parent, dict):
            parent[path[-1]] = _sanitize_value(target, key_hint=path[-1])


def sentry_before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets from events before shipping to Sentry."""
    del hint
    sanitized = deepcopy(event)
    for path in _BUILTIN_SENSITIVE_FIELD_PATHS:
        _sanitize_path(sanitized, path)

    breadcrumbs = sanitized.get("breadcrumbs")
    if isinstance(breadcrumbs, dict):
        values = breadcrumbs.get("values")
        if isinstance(values, list):
            breadcrumbs["values"] = [
                _sanitize_value(item) if isinstance(item, dict) else item for item in values
            ]
    return sanitized


def _build_integrations(source: SentrySource) -> list[Any]:
    integrations: list[Any] = []
    try:
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
    except Exception:  # noqa: BLE001
        return integrations

    if source == "fastapi":
        integrations.append(FastApiIntegration(transaction_style="endpoint"))
    if source == "celery":
        integrations.append(CeleryIntegration())
    return integrations


def _setting_str(name: str, default: str = "") -> str:
    value = getattr(settings, name, default)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else default


def _setting_float(name: str, default: float = 0.0) -> float:
    value = getattr(settings, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_sentry_environment() -> str:
    explicit = _setting_str("effective_sentry_env")
    if explicit:
        return explicit
    runtime = _setting_str("runtime_env", "development").lower()
    if runtime == "prod":
        return "production"
    if runtime == "dev":
        return "development"
    return runtime or "development"


def _resolve_sentry_release() -> str | None:
    release = _setting_str("effective_sentry_release")
    if release:
        return release
    return None


def init_backend_sentry(*, source: SentrySource) -> bool:
    """Initialize Sentry for one backend process role."""
    dsn = _setting_str("sentry_dsn")
    if not dsn:
        return False

    try:
        import sentry_sdk
    except Exception:  # noqa: BLE001
        logger.warning("Sentry SDK is not installed. source=%s", source)
        return False

    init_kwargs: dict[str, Any] = {
        "dsn": dsn,
        "environment": _resolve_sentry_environment(),
        "release": _resolve_sentry_release(),
        "before_send": sentry_before_send,
        "send_default_pii": False,
        "traces_sample_rate": _setting_float("sentry_traces_sample_rate", 0.0),
        "profiles_sample_rate": _setting_float("sentry_profiles_sample_rate", 0.0),
        "integrations": _build_integrations(source),
    }
    sentry_sdk.init(**init_kwargs)
    sentry_sdk.set_tag("source", source)
    sentry_sdk.set_tag("runtime_env", _setting_str("runtime_env", "unknown"))
    logger.info(
        "Sentry initialized for source=%s env=%s",
        source,
        _resolve_sentry_environment(),
    )
    return True


def capture_exception_with_context(
    exc: BaseException,
    *,
    tags: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> None:
    """Capture one exception with optional tags/extras if Sentry is available."""
    try:
        import sentry_sdk
    except Exception:  # noqa: BLE001
        return

    with sentry_sdk.push_scope() as scope:
        for key, value in (tags or {}).items():
            if value is None:
                continue
            scope.set_tag(str(key), str(value))
        for key, value in (extras or {}).items():
            scope.set_extra(str(key), _sanitize_value(value))
        sentry_sdk.capture_exception(exc)


def capture_message_with_context(
    message: str,
    *,
    level: Literal["debug", "info", "warning", "error", "fatal"] = "warning",
    tags: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    fingerprint: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Capture one message event with tags/extras if Sentry is available."""
    try:
        import sentry_sdk
    except Exception:  # noqa: BLE001
        return

    with sentry_sdk.push_scope() as scope:
        for key, value in (tags or {}).items():
            if value is None:
                continue
            scope.set_tag(str(key), str(value))
        for key, value in (extras or {}).items():
            scope.set_extra(str(key), _sanitize_value(value))
        if fingerprint:
            normalized = [str(item).strip() for item in fingerprint if str(item).strip()]
            if normalized:
                scope.fingerprint = normalized
        sentry_sdk.capture_message(message, level=level)
