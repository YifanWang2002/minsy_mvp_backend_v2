"""Strategy MCP tools: DSL validation, persistence, and indicator knowledge lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import FastMCP

from src.engine.feature.indicators import IndicatorCategory, IndicatorRegistry
from src.engine.strategy import (
    StrategyDslValidationException,
    StrategyStorageNotFoundError,
    upsert_strategy_dsl,
    validate_strategy_payload,
)
from src.mcp._utils import to_json, utc_now_iso
from src.models import database as db_module

TOOL_NAMES: tuple[str, ...] = (
    "strategy_validate_dsl",
    "strategy_upsert_dsl",
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
    if data:
        body.update(data)
    if not ok:
        body["error"] = {
            "code": error_code or "UNKNOWN_ERROR",
            "message": error_message or "Unknown error",
        }
    return to_json(body)


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("DSL payload must be a JSON object")
    return parsed


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
    def strategy_validate_dsl(dsl_json: str) -> str:
        try:
            payload = _parse_payload(dsl_json)
        except ValueError as exc:
            return _payload(
                tool="strategy_validate_dsl",
                ok=False,
                error_code="INVALID_JSON",
                error_message=str(exc),
            )

        validation = validate_strategy_payload(payload)
        return _payload(
            tool="strategy_validate_dsl",
            ok=validation.is_valid,
            data={
                "errors": _validation_errors_to_dict(validation.errors),
                "dsl_version": payload.get("dsl_version", ""),
            },
            error_code="STRATEGY_VALIDATION_FAILED" if not validation.is_valid else None,
            error_message="Strategy DSL validation failed." if not validation.is_valid else None,
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
                "metadata": {
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
                },
            },
        )
