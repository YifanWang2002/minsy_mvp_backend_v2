"""Strategy MCP tools: DSL validation, persistence, and indicator knowledge lookup."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import anyio
from mcp.server.fastmcp import Context, FastMCP

from packages.domain.strategy.feature.indicators import IndicatorCategory, IndicatorRegistry
from packages.domain.strategy import (
    StrategyDslValidationException,
    StrategyPatchApplyError,
    StrategyRevisionNotFoundError,
    StrategyStorageNotFoundError,
    StrategyVersionConflictError,
    create_strategy_draft,
    diff_strategy_versions,
    get_session_user_id,
    get_strategy_or_raise,
    get_strategy_version_payload,
    list_strategy_versions,
    patch_strategy_dsl,
    rollback_strategy_dsl,
    upsert_strategy_dsl,
    validate_strategy_payload,
)
from apps.mcp.common.utils import log_mcp_tool_result, to_json, utc_now_iso
from apps.mcp.auth.context_auth import (
    McpContextClaims,
    decode_mcp_context_token,
    extract_mcp_context_token,
)
from packages.infra.db import session as db_module
from packages.infra.redis.client import get_redis_client, init_redis
from packages.infra.observability.logger import logger

TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_upsert_dsl",
    "strategy_get_dsl",
    "strategy_list_tunable_params",
    "strategy_patch_dsl",
    "strategy_list_versions",
    "strategy_get_version_dsl",
    "strategy_diff_versions",
    "strategy_rollback_dsl",
    "get_indicator_detail",
    "get_indicator_catalog",
)

_SKILL_DIR = Path(__file__).resolve().parent / "skills" / "indicators"
_EXCLUDED_CATALOG_CATEGORIES: set[str] = {IndicatorCategory.CANDLE.value}
_OWNERSHIP_MISMATCH_ERROR = (
    "Strategy ownership mismatch for the provided session/user context."
)
_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    IndicatorCategory.OVERLAP.value: "Moving averages and overlays.",
    IndicatorCategory.MOMENTUM.value: "Momentum and trend-strength indicators.",
    IndicatorCategory.VOLATILITY.value: "Volatility and range indicators.",
    IndicatorCategory.VOLUME.value: "Volume and money-flow indicators.",
    IndicatorCategory.UTILS.value: "Utility/statistical indicators.",
}
_VALIDATION_RETRY_COUNTER_KEY_PREFIX = "strategy:validation_retry:"
_VALIDATION_RETRY_COUNTER_TTL_SECONDS = 60 * 10
_MAX_VALIDATION_FAILURES_PER_REQUEST = 2


def _float_env(name: str, *, default: float, minimum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if parsed < minimum:
        return minimum
    return parsed


_STRATEGY_VALIDATE_DRAFT_TIMEOUT_SECONDS = _float_env(
    "STRATEGY_VALIDATE_DRAFT_TIMEOUT_SECONDS",
    default=8.0,
    minimum=0.1,
)
_ALLOW_IN_MEMORY_VALIDATION_RETRY_FALLBACK = (
    os.getenv("ALLOW_IN_MEMORY_VALIDATION_RETRY_FALLBACK", "").strip().lower()
    in {"1", "true", "yes"}
)
_IN_MEMORY_VALIDATION_RETRY_COUNTS: dict[str, tuple[int, datetime, str]] = {}


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    error_context: dict[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "category": "strategy",
        "tool": tool,
        "ok": ok,
        "timestamp_utc": utc_now_iso(),
    }
    resolved_error_code: str | None = None
    resolved_error_message: str | None = None
    if data:
        body.update(data)
    if not ok:
        resolved_error_code = error_code or "UNKNOWN_ERROR"
        resolved_error_message = error_message or "Unknown error"
        body["error"] = {
            "code": resolved_error_code,
            "message": resolved_error_message,
        }
    log_mcp_tool_result(
        category="strategy",
        tool=tool,
        ok=ok,
        error_code=resolved_error_code,
        error_message=resolved_error_message,
        error_context=error_context,
    )
    return to_json(body)


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("DSL payload must be a JSON object")
    return parsed


def _parse_patch_ops(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON patch payload: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError("Patch payload must be a JSON array of operations")
    operations: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Patch operation at index {index} must be a JSON object")
        operations.append(dict(item))
    return operations


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


def _truncate_text(value: str, *, limit: int = 220) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _preview_value(value: Any, *, limit: int = 220) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _truncate_text(value, limit=limit)
    try:
        compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except TypeError:
        compact = str(value)
    compact = compact.strip()
    if not compact:
        return None
    return _truncate_text(compact, limit=limit)


def _sanitize_json_like_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value
    return str(value)


def _validation_errors_to_dict(errors: tuple[Any, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in errors:
        path = str(getattr(item, "path", "") or "")
        pointer = str(getattr(item, "pointer", "") or "")
        value = getattr(item, "value", None)
        expected = _sanitize_json_like_value(getattr(item, "expected", None))
        actual = _sanitize_json_like_value(getattr(item, "actual", None))
        suggestion = str(getattr(item, "suggestion", "") or "").strip()
        entry: dict[str, Any] = {
            "code": str(getattr(item, "code", "") or ""),
            "message": str(getattr(item, "message", "") or ""),
            "path": path,
            "path_pointer": pointer,
            "stage": str(getattr(item, "stage", "") or ""),
            "value": value,
            "value_preview": _preview_value(value),
            "expected": expected,
            "actual": actual,
            "suggestion": suggestion,
        }
        output.append(entry)
    return output


def _build_validation_signature(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return "none"
    compact = [
        {
            "code": item.get("code", ""),
            "path": item.get("path", ""),
            "msg": _truncate_text(str(item.get("message", "")), limit=80),
        }
        for item in errors
    ]
    serialized = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _top_validation_codes(errors: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    counter = Counter(
        code
        for code in (str(item.get("code", "")).strip() for item in errors)
        if code
    )
    top = counter.most_common(max(limit, 1))
    return [{"code": code, "count": count} for code, count in top]


def _build_validation_summary(
    *,
    errors: list[dict[str, Any]],
    request_id: str | None,
    retry_count: int | None,
    retry_limit_reached: bool,
    last_signature: str | None,
) -> dict[str, Any]:
    primary_error = errors[0] if errors else None
    summary: dict[str, Any] = {
        "error_count": len(errors),
        "top_codes": _top_validation_codes(errors),
        "primary_error": primary_error,
        "error_signature": _build_validation_signature(errors),
        "retry_limit": _MAX_VALIDATION_FAILURES_PER_REQUEST,
        "retry_limit_reached": retry_limit_reached,
    }
    if request_id:
        summary["request_id"] = request_id
    if retry_count is not None:
        summary["retry_count"] = retry_count
    if last_signature:
        summary["previous_error_signature"] = last_signature
        summary["same_error_repeated"] = last_signature == summary["error_signature"]
    return summary


def _validation_log_context(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "error_count": summary.get("error_count"),
        "top_codes": summary.get("top_codes"),
        "error_signature": summary.get("error_signature"),
        "retry_count": summary.get("retry_count"),
        "retry_limit": summary.get("retry_limit"),
        "retry_limit_reached": summary.get("retry_limit_reached"),
        "same_error_repeated": summary.get("same_error_repeated"),
        "request_id": summary.get("request_id"),
    }


def _use_in_memory_validation_retry_fallback() -> bool:
    if _ALLOW_IN_MEMORY_VALIDATION_RETRY_FALLBACK:
        return True
    return "PYTEST_CURRENT_TEST" in os.environ


async def _get_ready_redis_client():
    try:
        return get_redis_client()
    except RuntimeError:
        await init_redis()
        return get_redis_client()


def _validation_retry_counter_key(request_id: str) -> str:
    return f"{_VALIDATION_RETRY_COUNTER_KEY_PREFIX}{request_id}"


async def _track_validation_retry_state(
    *,
    request_id: str | None,
    signature: str,
) -> tuple[int | None, bool, str | None]:
    if not request_id:
        return None, False, None

    try:
        redis = await _get_ready_redis_client()
        key = _validation_retry_counter_key(request_id)
        previous_signature_raw = await redis.hget(key, "last_signature")
        previous_signature = (
            previous_signature_raw.strip()
            if isinstance(previous_signature_raw, str) and previous_signature_raw.strip()
            else None
        )
        count_raw = await redis.hincrby(key, "count", 1)
        await redis.hset(
            key,
            mapping={
                "last_signature": signature,
                "updated_at": utc_now_iso(),
            },
        )
        await redis.expire(key, _VALIDATION_RETRY_COUNTER_TTL_SECONDS)
        count = int(count_raw)
        return count, count >= _MAX_VALIDATION_FAILURES_PER_REQUEST, previous_signature
    except Exception:  # noqa: BLE001
        if not _use_in_memory_validation_retry_fallback():
            return None, False, None

    now = datetime.now(UTC)
    key = request_id
    count = 1
    previous_signature: str | None = None
    cached = _IN_MEMORY_VALIDATION_RETRY_COUNTS.get(key)
    if cached is not None:
        previous_count, expires_at, last_signature = cached
        if expires_at > now:
            count = previous_count + 1
            previous_signature = last_signature
    expires_at = now + timedelta(seconds=_VALIDATION_RETRY_COUNTER_TTL_SECONDS)
    _IN_MEMORY_VALIDATION_RETRY_COUNTS[key] = (count, expires_at, signature)
    return count, count >= _MAX_VALIDATION_FAILURES_PER_REQUEST, previous_signature


async def _build_validation_failure_envelope(
    *,
    errors: tuple[Any, ...],
    dsl_version: str = "",
    claims: McpContextClaims | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    serialized_errors = _validation_errors_to_dict(errors)
    signature = _build_validation_signature(serialized_errors)
    request_id = claims.request_id if claims is not None else None
    retry_count, retry_limit_reached, previous_signature = await _track_validation_retry_state(
        request_id=request_id,
        signature=signature,
    )
    summary = _build_validation_summary(
        errors=serialized_errors,
        request_id=request_id,
        retry_count=retry_count,
        retry_limit_reached=retry_limit_reached,
        last_signature=previous_signature,
    )
    data: dict[str, Any] = {
        "errors": serialized_errors,
        "validation": summary,
    }
    if dsl_version.strip():
        data["dsl_version"] = dsl_version.strip()
    error_message = "Strategy DSL validation failed."
    if retry_limit_reached:
        retry_count_text = retry_count if retry_count is not None else "many"
        error_message = (
            "Strategy DSL validation failed. "
            f"Retry limit reached in this request ({retry_count_text} failures)."
        )
    return data, summary, error_message


def _payload_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _strategy_metadata_snapshot(strategy: Any) -> dict[str, Any]:
    raw_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    updated_at = strategy.updated_at if strategy.updated_at else strategy.created_at
    return {
        "user_id": str(strategy.user_id),
        "session_id": str(strategy.session_id),
        "strategy_name": strategy.name,
        "dsl_version": strategy.dsl_version or "",
        "version": int(strategy.version),
        "status": strategy.status,
        "timeframe": strategy.timeframe,
        "symbol_count": len(strategy.symbols or []),
        "payload_hash": _payload_hash(raw_payload),
        "last_updated_at": _as_utc(updated_at).isoformat(),
    }


def _strategy_metadata_from_receipt(receipt: Any) -> dict[str, Any]:
    return {
        "user_id": str(receipt.user_id),
        "session_id": str(receipt.session_id),
        "strategy_name": receipt.strategy_name,
        "dsl_version": receipt.dsl_version,
        "version": receipt.version,
        "status": receipt.status,
        "timeframe": receipt.timeframe,
        "symbol_count": receipt.symbol_count,
        "payload_hash": receipt.payload_hash,
        "last_updated_at": receipt.last_updated_at.isoformat(),
    }


def _set_draft_warning(
    response_data: dict[str, Any],
    *,
    code: str,
    message: str,
) -> None:
    response_data["draft_warning"] = {
        "code": code,
        "message": message,
    }


def _revision_snapshot(revision: Any) -> dict[str, Any]:
    return {
        "strategy_id": str(revision.strategy_id),
        "session_id": str(revision.session_id) if revision.session_id else None,
        "version": int(revision.version),
        "dsl_version": revision.dsl_version,
        "payload_hash": revision.payload_hash,
        "change_type": revision.change_type,
        "source_version": revision.source_version,
        "patch_op_count": int(revision.patch_op_count),
        "created_at": revision.created_at.isoformat(),
    }


def _read_indicator_skill(indicator: str) -> dict[str, Any] | None:
    file_path = _SKILL_DIR / f"{indicator.strip().lower()}.md"
    if not file_path.is_file():
        return None

    content = file_path.read_text(encoding="utf-8")
    category = ""
    summary = ""
    for line in content.splitlines():
        normalized = line.strip()
        if normalized.startswith("category:"):
            category = normalized.split(":", 1)[1].strip()
        if normalized.startswith("summary:"):
            summary = normalized.split(":", 1)[1].strip()

    return {
        "indicator": indicator.strip().lower(),
        "category": category,
        "summary": summary,
        "content": content,
        "skill_path": str(file_path),
    }


def _normalize_indicator_inputs(
    *,
    indicator: str,
    indicator_list: list[str] | None,
) -> list[str]:
    raw = [indicator]
    if indicator_list:
        raw.extend(indicator_list)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = item.strip().lower() if isinstance(item, str) else ""
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _indicator_metadata_snapshot(indicator: str) -> dict[str, Any] | None:
    metadata = IndicatorRegistry.get(indicator)
    if metadata is None:
        return None

    return {
        "name": metadata.name,
        "full_name": metadata.full_name,
        "category": metadata.category.value,
        "description": metadata.description,
        "params": [
            {
                "name": param.name,
                "type": param.type,
                "default": param.default,
                "min": param.min_value,
                "max": param.max_value,
                "choices": param.choices,
                "description": param.description,
            }
            for param in metadata.params
        ],
        "outputs": [
            {
                "name": output.name,
                "description": output.description,
            }
            for output in metadata.outputs
        ],
        "required_columns": list(metadata.required_columns),
    }


def _visible_catalog_categories() -> list[IndicatorCategory]:
    categories: list[IndicatorCategory] = []
    for category in IndicatorCategory:
        if category.value in _EXCLUDED_CATALOG_CATEGORIES:
            continue
        if IndicatorRegistry.list_by_category(category):
            categories.append(category)
    return categories


async def _new_db_session():
    if db_module.AsyncSessionLocal is None:
        # MCP worker process only needs a ready pool; schema is managed by API startup/migrations.
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


async def _get_owned_strategy(
    *,
    db: Any,
    strategy_id: UUID,
    session_id: UUID | None = None,
    user_id: UUID | None = None,
) -> Any:
    resolved_user_id = user_id
    if session_id is not None:
        session_user_id = await get_session_user_id(db, session_id=session_id)
        if resolved_user_id is not None and session_user_id != resolved_user_id:
            raise StrategyStorageNotFoundError(_OWNERSHIP_MISMATCH_ERROR)
        resolved_user_id = session_user_id

    if resolved_user_id is None:
        raise StrategyStorageNotFoundError(
            "Missing strategy ownership context: provide session_id or MCP context."
        )

    strategy = await get_strategy_or_raise(db, strategy_id=strategy_id)
    if strategy.user_id != resolved_user_id:
        raise StrategyStorageNotFoundError(_OWNERSHIP_MISMATCH_ERROR)
    return strategy


def _resolve_context_claims(ctx: Context | None) -> McpContextClaims | None:
    if ctx is None:
        return None

    try:
        request = ctx.request_context.request
    except Exception:  # noqa: BLE001
        return None
    headers = getattr(request, "headers", None)
    token = extract_mcp_context_token(headers)
    if token is None:
        return None
    return decode_mcp_context_token(token)


def _resolve_session_uuid(
    *,
    session_id: str,
    claims: McpContextClaims | None,
    required: bool,
) -> UUID | None:
    explicit_session_uuid: UUID | None = None
    raw_session_id = session_id.strip()
    if _is_placeholder_session_id(raw_session_id):
        raw_session_id = ""
    if raw_session_id:
        explicit_session_uuid = _parse_uuid(raw_session_id, "session_id")

    context_session_uuid = claims.session_id if claims is not None else None
    if (
        explicit_session_uuid is not None
        and context_session_uuid is not None
        and explicit_session_uuid != context_session_uuid
    ):
        raise ValueError("session_id does not match MCP context session.")

    resolved = explicit_session_uuid or context_session_uuid
    if required and resolved is None:
        raise ValueError("session_id is required (argument or MCP context).")
    return resolved


def _is_placeholder_session_id(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in {
        "-",
        "--",
        "—",
        "–",
        "_",
        "__",
        "none",
        "null",
        "nil",
        "n/a",
        "na",
        "undefined",
        "(none)",
        "<none>",
    }:
        return True
    # Common UI placeholders that may be rendered with dash-like characters.
    if all(ch in {"-", "—", "–", "_", " "} for ch in normalized):
        return True
    return False


async def _assert_session_matches_context_user(
    *,
    db: Any,
    session_id: UUID | None,
    claims: McpContextClaims | None,
) -> None:
    if session_id is None or claims is None:
        return
    session_user_id = await get_session_user_id(db, session_id=session_id)
    if session_user_id != claims.user_id:
        raise StrategyStorageNotFoundError(_OWNERSHIP_MISMATCH_ERROR)


def _resolve_context_user_id(claims: McpContextClaims | None) -> UUID | None:
    if claims is None:
        return None
    return claims.user_id


async def _resolve_owned_strategy(
    *,
    db: Any,
    strategy_id: UUID,
    session_id: str,
    ctx: Context | None,
) -> Any:
    claims = _resolve_context_claims(ctx)
    session_uuid = _resolve_session_uuid(
        session_id=session_id,
        claims=claims,
        required=False,
    )
    user_uuid = _resolve_context_user_id(claims)
    return await _get_owned_strategy(
        db=db,
        strategy_id=strategy_id,
        session_id=session_uuid,
        user_id=user_uuid,
    )


def _parse_expected_version(expected_version: int) -> int | None:
    if not isinstance(expected_version, int):
        raise ValueError("expected_version must be an integer")
    if expected_version > 0:
        return expected_version
    if expected_version < 0:
        raise ValueError("expected_version must be >= 0")
    return None


def _to_dsl_param_name(*, factor_type: str, indicator_param_name: str) -> str:
    mapping: dict[str, str] = {
        "length": "period",
        "std": "std_dev",
        "fastk_period": "k_period",
        "slowk_period": "k_smooth",
        "slowd_period": "d_period",
    }
    if factor_type in {"ema", "sma", "wma", "dema", "tema", "kama", "rsi", "atr", "bbands"}:
        if indicator_param_name == "length":
            return "period"
    if factor_type == "bbands" and indicator_param_name == "std":
        return "std_dev"
    if factor_type == "stoch":
        return mapping.get(indicator_param_name, indicator_param_name)
    return mapping.get(indicator_param_name, indicator_param_name)


def _estimate_step(
    *,
    value_type: str,
    min_value: Any,
    max_value: Any,
    default: Any,
) -> float | int | None:
    kind = value_type.strip().lower()
    if kind == "int":
        return 1
    if kind != "float":
        return None

    min_float = min_value if isinstance(min_value, int | float) else None
    max_float = max_value if isinstance(max_value, int | float) else None
    default_float = default if isinstance(default, int | float) else None

    if min_float is not None and max_float is not None and max_float > min_float:
        return float(max((max_float - min_float) / 100.0, 1e-4))
    if default_float is not None:
        return float(max(abs(default_float) / 10.0, 1e-4))
    return 0.01


def get_indicator_detail(
    indicator: str = "",
    indicator_list: list[str] | None = None,
) -> str:
    """Return indicator skill detail for one or more indicators."""

    requested = _normalize_indicator_inputs(
        indicator=indicator,
        indicator_list=indicator_list,
    )
    if not requested:
        return _payload(
            tool="get_indicator_detail",
            ok=False,
            error_code="INVALID_INPUT",
            error_message="Provide `indicator` or `indicator_list`.",
        )

    found: list[dict[str, Any]] = []
    missing: list[str] = []

    for name in requested:
        skill = _read_indicator_skill(name)
        metadata = _indicator_metadata_snapshot(name)
        if skill is None and metadata is None:
            missing.append(name)
            continue

        category = (metadata or {}).get("category") or (skill or {}).get("category") or ""
        summary = (skill or {}).get("summary") or (metadata or {}).get("description") or ""
        found.append(
            {
                "indicator": name,
                "category": category,
                "summary": summary,
                "skill_path": (skill or {}).get("skill_path", ""),
                "content": (skill or {}).get("content", ""),
                "registry": metadata,
            }
        )

    if not found:
        return _payload(
            tool="get_indicator_detail",
            ok=False,
            data={
                "requested_indicators": requested,
                "missing": missing,
                "available_categories": [
                    category.value for category in _visible_catalog_categories()
                ],
            },
            error_code="INDICATOR_NOT_FOUND",
            error_message="No indicator detail found for requested indicators.",
        )

    return _payload(
        tool="get_indicator_detail",
        ok=True,
        data={
            "requested_indicators": requested,
            "count": len(found),
            "indicators": found,
            "missing": missing,
            "available_categories": [
                category.value for category in _visible_catalog_categories()
            ],
        },
    )


def get_indicator_catalog(category: str = "") -> str:
    """Return registry catalog for categories: overlap, momentum, volatility, volume, utils."""

    available_categories = [
        category_item.value for category_item in _visible_catalog_categories()
    ]
    requested = category.strip().lower()

    categories_to_render: list[IndicatorCategory] = []
    if requested:
        if requested in _EXCLUDED_CATALOG_CATEGORIES:
            return _payload(
                tool="get_indicator_catalog",
                ok=False,
                data={
                    "available_categories": available_categories,
                    "excluded_categories": sorted(_EXCLUDED_CATALOG_CATEGORIES),
                },
                error_code="CATEGORY_EXCLUDED",
                error_message=(
                    f"Category '{requested}' is intentionally excluded from catalog output."
                ),
            )
        try:
            requested_category = IndicatorCategory(requested)
        except ValueError:
            return _payload(
                tool="get_indicator_catalog",
                ok=False,
                data={
                    "available_categories": available_categories,
                    "excluded_categories": sorted(_EXCLUDED_CATALOG_CATEGORIES),
                },
                error_code="INVALID_CATEGORY",
                error_message=f"Unknown category '{requested}'.",
            )
        categories_to_render = [requested_category]
    else:
        categories_to_render = _visible_catalog_categories()

    grouped: list[dict[str, Any]] = []
    total = 0
    for category_item in categories_to_render:
        if category_item.value in _EXCLUDED_CATALOG_CATEGORIES:
            continue
        names = IndicatorRegistry.list_by_category(category_item)
        indicators: list[dict[str, Any]] = []
        for name in names:
            snapshot = _indicator_metadata_snapshot(name)
            if snapshot is None:
                continue
            skill = _read_indicator_skill(name)
            indicators.append(
                {
                    "indicator": snapshot["name"],
                    "full_name": snapshot["full_name"],
                    "description": snapshot["description"],
                    "params": snapshot["params"],
                    "outputs": snapshot["outputs"],
                    "required_columns": snapshot["required_columns"],
                    "skill_summary": (skill or {}).get("summary", ""),
                    "skill_path": (skill or {}).get("skill_path", ""),
                }
            )

        if not indicators and requested:
            grouped.append(
                {
                    "category": category_item.value,
                    "description": _CATEGORY_DESCRIPTIONS.get(category_item.value, ""),
                    "count": 0,
                    "indicators": [],
                }
            )
            continue
        if not indicators:
            continue

        total += len(indicators)
        grouped.append(
            {
                "category": category_item.value,
                "description": _CATEGORY_DESCRIPTIONS.get(category_item.value, ""),
                "count": len(indicators),
                "indicators": indicators,
            }
        )

    return _payload(
        tool="get_indicator_catalog",
        ok=True,
        data={
            "category_filter": requested or None,
            "available_categories": available_categories,
            "excluded_categories": sorted(_EXCLUDED_CATALOG_CATEGORIES),
            "count": total,
            "categories": grouped,
        },
    )


async def strategy_validate_dsl(
    dsl_json: str,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        payload = _parse_payload(dsl_json)
    except ValueError as exc:
        return _payload(
            tool="strategy_validate_dsl",
            ok=False,
            error_code="INVALID_JSON",
            error_message=str(exc),
        )

    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=False,
        )
    except ValueError as exc:
        return _payload(
            tool="strategy_validate_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    validation = validate_strategy_payload(payload)
    if not validation.is_valid:
        data, summary, error_message = await _build_validation_failure_envelope(
            errors=validation.errors,
            dsl_version=str(payload.get("dsl_version", "")),
            claims=claims,
        )
        return _payload(
            tool="strategy_validate_dsl",
            ok=False,
            data=data,
            error_code="STRATEGY_VALIDATION_FAILED",
            error_message=error_message,
            error_context=_validation_log_context(summary),
        )

    response_data: dict[str, Any] = {
        "errors": [],
        "dsl_version": payload.get("dsl_version", ""),
    }
    if session_uuid is not None:
        try:
            async with await _new_db_session() as db:
                await _assert_session_matches_context_user(
                    db=db,
                    session_id=session_uuid,
                    claims=claims,
                )
                session_user_id = await get_session_user_id(db, session_id=session_uuid)
        except StrategyStorageNotFoundError as exc:
            return _payload(
                tool="strategy_validate_dsl",
                ok=False,
                error_code="STRATEGY_STORAGE_NOT_FOUND",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return _payload(
                tool="strategy_validate_dsl",
                ok=False,
                error_code="STRATEGY_STORAGE_ERROR",
                error_message=f"{type(exc).__name__}: {exc}",
            )

        try:
            with anyio.fail_after(_STRATEGY_VALIDATE_DRAFT_TIMEOUT_SECONDS):
                draft = await create_strategy_draft(
                    user_id=session_user_id,
                    session_id=session_uuid,
                    dsl_json=payload,
                )
        except TimeoutError:
            timeout_message = (
                "Strategy draft persistence timed out; "
                "validation result returned without strategy_draft_id."
            )
            logger.warning(
                "strategy_validate_dsl draft timeout session_id=%s timeout_seconds=%.2f",
                session_uuid,
                _STRATEGY_VALIDATE_DRAFT_TIMEOUT_SECONDS,
            )
            _set_draft_warning(
                response_data,
                code="DRAFT_PERSIST_TIMEOUT",
                message=timeout_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "strategy_validate_dsl draft persistence skipped session_id=%s error=%s",
                session_uuid,
                type(exc).__name__,
            )
            _set_draft_warning(
                response_data,
                code="DRAFT_PERSIST_FAILED",
                message=(
                    "Strategy draft persistence failed; "
                    "validation result returned without strategy_draft_id."
                ),
            )
        else:
            response_data["strategy_draft_id"] = str(draft.strategy_draft_id)
            response_data["draft_expires_at"] = draft.expires_at.isoformat()
            response_data["draft_ttl_seconds"] = draft.ttl_seconds

    return _payload(
        tool="strategy_validate_dsl",
        ok=True,
        data=response_data,
    )


async def strategy_upsert_dsl(
    dsl_json: str,
    strategy_id: str = "",
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id") if strategy_id.strip() else None
        payload = _parse_payload(dsl_json)
    except ValueError as exc:
        return _payload(
            tool="strategy_upsert_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            result = await upsert_strategy_dsl(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                dsl_payload=payload,
                auto_commit=True,
            )
    except StrategyDslValidationException as exc:
        data, summary, error_message = await _build_validation_failure_envelope(
            errors=tuple(exc.errors),
            dsl_version=str(payload.get("dsl_version", "")),
            claims=claims,
        )
        return _payload(
            tool="strategy_upsert_dsl",
            ok=False,
            data=data,
            error_code="STRATEGY_VALIDATION_FAILED",
            error_message=error_message,
            error_context=_validation_log_context(summary),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_upsert_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_upsert_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    receipt = result.receipt
    return _payload(
        tool="strategy_upsert_dsl",
        ok=True,
        data={
            "strategy_id": str(receipt.strategy_id),
            "metadata": _strategy_metadata_from_receipt(receipt),
        },
    )


async def strategy_get_dsl(
    strategy_id: str,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
    except ValueError as exc:
        return _payload(
            tool="strategy_get_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            strategy = await _resolve_owned_strategy(
                db=db,
                strategy_id=strategy_uuid,
                session_id=session_id,
                ctx=ctx,
            )
    except ValueError as exc:
        return _payload(
            tool="strategy_get_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_get_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_get_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    dsl_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    return _payload(
        tool="strategy_get_dsl",
        ok=True,
        data={
            "strategy_id": str(strategy.id),
            "dsl_json": dsl_payload,
            "metadata": _strategy_metadata_snapshot(strategy),
        },
    )


async def strategy_list_tunable_params(
    strategy_id: str,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
    except ValueError as exc:
        return _payload(
            tool="strategy_list_tunable_params",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            strategy = await _resolve_owned_strategy(
                db=db,
                strategy_id=strategy_uuid,
                session_id=session_id,
                ctx=ctx,
            )
    except ValueError as exc:
        return _payload(
            tool="strategy_list_tunable_params",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_list_tunable_params",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_list_tunable_params",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    dsl_payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
    factors_raw = dsl_payload.get("factors")
    factors = factors_raw if isinstance(factors_raw, dict) else {}
    tunable_params: list[dict[str, Any]] = []
    for factor_id, factor_def in factors.items():
        if not isinstance(factor_id, str) or not isinstance(factor_def, dict):
            continue

        factor_type = str(factor_def.get("type", "")).strip().lower()
        params = factor_def.get("params")
        if not isinstance(params, dict):
            params = {}

        metadata = IndicatorRegistry.get(factor_type)
        if metadata is None:
            continue

        for param in metadata.params:
            dsl_param = _to_dsl_param_name(
                factor_type=factor_type,
                indicator_param_name=param.name,
            )
            current_value = params.get(dsl_param, param.default)
            tunable = param.type in {"int", "float"} or bool(param.choices)
            tunable_params.append(
                {
                    "factor_id": factor_id,
                    "factor_type": factor_type,
                    "param_name": dsl_param,
                    "json_path": f"/factors/{factor_id}/params/{dsl_param}",
                    "current_value": current_value,
                    "type": param.type,
                    "min": param.min_value,
                    "max": param.max_value,
                    "default": param.default,
                    "choices": list(param.choices) if param.choices else [],
                    "suggested_step": _estimate_step(
                        value_type=param.type,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        default=param.default,
                    ),
                    "tunable": tunable,
                }
            )

    tunable_params.sort(
        key=lambda item: (
            str(item.get("factor_id", "")),
            str(item.get("param_name", "")),
        )
    )
    return _payload(
        tool="strategy_list_tunable_params",
        ok=True,
        data={
            "strategy_id": str(strategy.id),
            "metadata": _strategy_metadata_snapshot(strategy),
            "count": len(tunable_params),
            "params": tunable_params,
        },
    )


async def strategy_patch_dsl(
    strategy_id: str,
    patch_json: str,
    expected_version: int = 0,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        patch_ops = _parse_patch_ops(patch_json)
        parsed_expected_version = _parse_expected_version(expected_version)
    except ValueError as exc:
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            result = await patch_strategy_dsl(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                patch_ops=patch_ops,
                expected_version=parsed_expected_version,
                auto_commit=True,
            )
    except StrategyPatchApplyError as exc:
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            error_code="STRATEGY_PATCH_APPLY_FAILED",
            error_message=str(exc),
        )
    except StrategyVersionConflictError as exc:
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            error_code="STRATEGY_VERSION_CONFLICT",
            error_message=str(exc),
        )
    except StrategyDslValidationException as exc:
        data, summary, error_message = await _build_validation_failure_envelope(
            errors=tuple(exc.errors),
            claims=claims,
        )
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            data=data,
            error_code="STRATEGY_VALIDATION_FAILED",
            error_message=error_message,
            error_context=_validation_log_context(summary),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_patch_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    receipt = result.receipt
    return _payload(
        tool="strategy_patch_dsl",
        ok=True,
        data={
            "strategy_id": str(receipt.strategy_id),
            "patch_op_count": len(patch_ops),
            "metadata": _strategy_metadata_from_receipt(receipt),
        },
    )


async def strategy_list_versions(
    strategy_id: str,
    limit: int = 20,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        if not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if limit <= 0 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
    except ValueError as exc:
        return _payload(
            tool="strategy_list_versions",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            versions = await list_strategy_versions(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                limit=limit,
            )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_list_versions",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_list_versions",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="strategy_list_versions",
        ok=True,
        data={
            "strategy_id": str(strategy_uuid),
            "count": len(versions),
            "versions": [_revision_snapshot(item) for item in versions],
        },
    )


async def strategy_get_version_dsl(
    strategy_id: str,
    version: int,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        if not isinstance(version, int):
            raise ValueError("version must be an integer")
        if version <= 0:
            raise ValueError("version must be >= 1")
    except ValueError as exc:
        return _payload(
            tool="strategy_get_version_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            resolved = await get_strategy_version_payload(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                version=version,
            )
    except StrategyRevisionNotFoundError as exc:
        return _payload(
            tool="strategy_get_version_dsl",
            ok=False,
            error_code="STRATEGY_REVISION_NOT_FOUND",
            error_message=str(exc),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_get_version_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_get_version_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="strategy_get_version_dsl",
        ok=True,
        data={
            "strategy_id": str(resolved.strategy_id),
            "version": resolved.version,
            "dsl_json": resolved.dsl_payload,
            "revision": _revision_snapshot(resolved.receipt),
        },
    )


async def strategy_diff_versions(
    strategy_id: str,
    from_version: int,
    to_version: int,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        if not isinstance(from_version, int) or not isinstance(to_version, int):
            raise ValueError("from_version and to_version must be integers")
        if from_version <= 0 or to_version <= 0:
            raise ValueError("from_version and to_version must be >= 1")
    except ValueError as exc:
        return _payload(
            tool="strategy_diff_versions",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            diff_result = await diff_strategy_versions(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                from_version=from_version,
                to_version=to_version,
            )
    except StrategyRevisionNotFoundError as exc:
        return _payload(
            tool="strategy_diff_versions",
            ok=False,
            error_code="STRATEGY_REVISION_NOT_FOUND",
            error_message=str(exc),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_diff_versions",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_diff_versions",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return _payload(
        tool="strategy_diff_versions",
        ok=True,
        data={
            "strategy_id": str(diff_result.strategy_id),
            "from_version": diff_result.from_version,
            "to_version": diff_result.to_version,
            "patch_op_count": diff_result.op_count,
            "patch_ops": diff_result.patch_ops,
            "from_payload_hash": diff_result.from_payload_hash,
            "to_payload_hash": diff_result.to_payload_hash,
        },
    )


async def strategy_rollback_dsl(
    strategy_id: str,
    target_version: int,
    expected_version: int = 0,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    try:
        claims = _resolve_context_claims(ctx)
        session_uuid = _resolve_session_uuid(
            session_id=session_id,
            claims=claims,
            required=True,
        )
        strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
        if not isinstance(target_version, int):
            raise ValueError("target_version must be an integer")
        if target_version <= 0:
            raise ValueError("target_version must be >= 1")
        parsed_expected_version = _parse_expected_version(expected_version)
    except ValueError as exc:
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="INVALID_INPUT",
            error_message=str(exc),
        )

    try:
        async with await _new_db_session() as db:
            await _assert_session_matches_context_user(
                db=db,
                session_id=session_uuid,
                claims=claims,
            )
            result = await rollback_strategy_dsl(
                db,
                session_id=session_uuid,
                strategy_id=strategy_uuid,
                target_version=target_version,
                expected_version=parsed_expected_version,
                auto_commit=True,
            )
    except StrategyRevisionNotFoundError as exc:
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="STRATEGY_REVISION_NOT_FOUND",
            error_message=str(exc),
        )
    except StrategyPatchApplyError as exc:
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="STRATEGY_PATCH_APPLY_FAILED",
            error_message=str(exc),
        )
    except StrategyVersionConflictError as exc:
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="STRATEGY_VERSION_CONFLICT",
            error_message=str(exc),
        )
    except StrategyDslValidationException as exc:
        data, summary, error_message = await _build_validation_failure_envelope(
            errors=tuple(exc.errors),
            claims=claims,
        )
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            data=data,
            error_code="STRATEGY_VALIDATION_FAILED",
            error_message=error_message,
            error_context=_validation_log_context(summary),
        )
    except StrategyStorageNotFoundError as exc:
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_NOT_FOUND",
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _payload(
            tool="strategy_rollback_dsl",
            ok=False,
            error_code="STRATEGY_STORAGE_ERROR",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    receipt = result.receipt
    return _payload(
        tool="strategy_rollback_dsl",
        ok=True,
        data={
            "strategy_id": str(receipt.strategy_id),
            "target_version": target_version,
            "metadata": _strategy_metadata_from_receipt(receipt),
        },
    )


def register_strategy_tools(mcp: FastMCP) -> None:
    """Register strategy-related tools."""
    mcp.tool()(get_indicator_detail)
    mcp.tool()(get_indicator_catalog)
    mcp.tool()(strategy_validate_dsl)
    mcp.tool()(strategy_upsert_dsl)
    mcp.tool()(strategy_get_dsl)
    mcp.tool()(strategy_list_tunable_params)
    mcp.tool()(strategy_patch_dsl)
    mcp.tool()(strategy_list_versions)
    mcp.tool()(strategy_get_version_dsl)
    mcp.tool()(strategy_diff_versions)
    mcp.tool()(strategy_rollback_dsl)
