"""Strategy MCP tools: DSL validation, persistence, and indicator knowledge lookup."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from src.engine.feature.indicators import IndicatorCategory, IndicatorRegistry
from src.engine.strategy import (
    StrategyDslValidationException,
    StrategyPatchApplyError,
    create_strategy_draft,
    StrategyRevisionNotFoundError,
    StrategyStorageNotFoundError,
    StrategyVersionConflictError,
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
from src.mcp._utils import log_mcp_tool_result, to_json, utc_now_iso
from src.models import database as db_module

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
_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    IndicatorCategory.OVERLAP.value: "Moving averages and overlays.",
    IndicatorCategory.MOMENTUM.value: "Momentum and trend-strength indicators.",
    IndicatorCategory.VOLATILITY.value: "Volatility and range indicators.",
    IndicatorCategory.VOLUME.value: "Volume and money-flow indicators.",
    IndicatorCategory.UTILS.value: "Utility/statistical indicators.",
}


def _payload(
    *,
    tool: str,
    ok: bool,
    data: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
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


def _validation_errors_to_dict(errors: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "code": item.code,
            "message": item.message,
            "path": item.path,
            "value": item.value,
        }
        for item in errors
    ]


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
        await db_module.init_postgres()
    assert db_module.AsyncSessionLocal is not None
    return db_module.AsyncSessionLocal()


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


def register_strategy_tools(mcp: FastMCP) -> None:
    """Register strategy-related tools."""

    @mcp.tool()
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

            category = (
                (metadata or {}).get("category")
                or (skill or {}).get("category")
                or ""
            )
            summary = (
                (skill or {}).get("summary")
                or (metadata or {}).get("description")
                or ""
            )

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

    @mcp.tool()
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
                    error_message=(
                        f"Unknown category '{requested}'."
                    ),
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

    @mcp.tool()
    async def strategy_validate_dsl(
        dsl_json: str,
        session_id: str = "",
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

        session_uuid: UUID | None = None
        if session_id.strip():
            try:
                session_uuid = _parse_uuid(session_id, "session_id")
            except ValueError as exc:
                return _payload(
                    tool="strategy_validate_dsl",
                    ok=False,
                    error_code="INVALID_INPUT",
                    error_message=str(exc),
                )

        validation = validate_strategy_payload(payload)
        if not validation.is_valid:
            return _payload(
                tool="strategy_validate_dsl",
                ok=False,
                data={
                    "errors": _validation_errors_to_dict(validation.errors),
                    "dsl_version": payload.get("dsl_version", ""),
                },
                error_code="STRATEGY_VALIDATION_FAILED",
                error_message="Strategy DSL validation failed.",
            )

        response_data: dict[str, Any] = {
            "errors": [],
            "dsl_version": payload.get("dsl_version", ""),
        }

        if session_uuid is not None:
            try:
                async with await _new_db_session() as db:
                    session_user_id = await get_session_user_id(db, session_id=session_uuid)
                draft = await create_strategy_draft(
                    user_id=session_user_id,
                    session_id=session_uuid,
                    dsl_json=payload,
                )
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
                    error_code="STRATEGY_DRAFT_STORE_ERROR",
                    error_message=f"{type(exc).__name__}: {exc}",
                )

            response_data["strategy_draft_id"] = str(draft.strategy_draft_id)
            response_data["draft_expires_at"] = draft.expires_at.isoformat()
            response_data["draft_ttl_seconds"] = draft.ttl_seconds

        return _payload(
            tool="strategy_validate_dsl",
            ok=True,
            data=response_data,
        )

    @mcp.tool()
    async def strategy_upsert_dsl(
        session_id: str,
        dsl_json: str,
        strategy_id: str = "",
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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
                result = await upsert_strategy_dsl(
                    db,
                    session_id=session_uuid,
                    strategy_id=strategy_uuid,
                    dsl_payload=payload,
                    auto_commit=True,
                )
        except StrategyDslValidationException as exc:
            return _payload(
                tool="strategy_upsert_dsl",
                ok=False,
                data={"errors": _validation_errors_to_dict(tuple(exc.errors))},
                error_code="STRATEGY_VALIDATION_FAILED",
                error_message="Strategy DSL validation failed.",
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

    @mcp.tool()
    async def strategy_get_dsl(
        session_id: str,
        strategy_id: str,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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
                session_user_id = await get_session_user_id(db, session_id=session_uuid)
                strategy = await get_strategy_or_raise(db, strategy_id=strategy_uuid)
                if strategy.user_id != session_user_id:
                    raise StrategyStorageNotFoundError(
                        "Strategy ownership mismatch for the provided session/user context.",
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

    @mcp.tool()
    async def strategy_list_tunable_params(
        session_id: str,
        strategy_id: str,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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
                session_user_id = await get_session_user_id(db, session_id=session_uuid)
                strategy = await get_strategy_or_raise(db, strategy_id=strategy_uuid)
                if strategy.user_id != session_user_id:
                    raise StrategyStorageNotFoundError(
                        "Strategy ownership mismatch for the provided session/user context.",
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
            if not isinstance(factor_id, str):
                continue
            if not isinstance(factor_def, dict):
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

    @mcp.tool()
    async def strategy_patch_dsl(
        session_id: str,
        strategy_id: str,
        patch_json: str,
        expected_version: int = 0,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
            strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
            patch_ops = _parse_patch_ops(patch_json)
            parsed_expected_version: int | None = None
            if not isinstance(expected_version, int):
                raise ValueError("expected_version must be an integer")
            if expected_version > 0:
                parsed_expected_version = expected_version
            elif expected_version < 0:
                raise ValueError("expected_version must be >= 0")
        except ValueError as exc:
            return _payload(
                tool="strategy_patch_dsl",
                ok=False,
                error_code="INVALID_INPUT",
                error_message=str(exc),
            )

        try:
            async with await _new_db_session() as db:
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
            return _payload(
                tool="strategy_patch_dsl",
                ok=False,
                data={"errors": _validation_errors_to_dict(tuple(exc.errors))},
                error_code="STRATEGY_VALIDATION_FAILED",
                error_message="Strategy DSL validation failed.",
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

    @mcp.tool()
    async def strategy_list_versions(
        session_id: str,
        strategy_id: str,
        limit: int = 20,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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

    @mcp.tool()
    async def strategy_get_version_dsl(
        session_id: str,
        strategy_id: str,
        version: int,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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

    @mcp.tool()
    async def strategy_diff_versions(
        session_id: str,
        strategy_id: str,
        from_version: int,
        to_version: int,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
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

    @mcp.tool()
    async def strategy_rollback_dsl(
        session_id: str,
        strategy_id: str,
        target_version: int,
        expected_version: int = 0,
    ) -> str:
        try:
            session_uuid = _parse_uuid(session_id, "session_id")
            strategy_uuid = _parse_uuid(strategy_id, "strategy_id")
            if not isinstance(target_version, int):
                raise ValueError("target_version must be an integer")
            if target_version <= 0:
                raise ValueError("target_version must be >= 1")
            parsed_expected_version: int | None = None
            if not isinstance(expected_version, int):
                raise ValueError("expected_version must be an integer")
            if expected_version > 0:
                parsed_expected_version = expected_version
            elif expected_version < 0:
                raise ValueError("expected_version must be >= 0")
        except ValueError as exc:
            return _payload(
                tool="strategy_rollback_dsl",
                ok=False,
                error_code="INVALID_INPUT",
                error_message=str(exc),
            )

        try:
            async with await _new_db_session() as db:
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
            return _payload(
                tool="strategy_rollback_dsl",
                ok=False,
                data={"errors": _validation_errors_to_dict(tuple(exc.errors))},
                error_code="STRATEGY_VALIDATION_FAILED",
                error_message="Strategy DSL validation failed.",
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
