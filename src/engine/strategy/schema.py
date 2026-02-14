"""JSON schema validation for strategy DSL payloads."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.engine.strategy.errors import StrategyDslError

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SCHEMA_PATH = _ASSETS_DIR / "strategy_dsl_schema.json"

_ADDITIONAL_PROPERTY_PATTERN = re.compile(r"'([^']+)' was unexpected")


@lru_cache(maxsize=1)
def load_dsl_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_dsl_schema())


def _format_path(error: ValidationError) -> str:
    parts: list[str] = ["$"]
    for item in error.absolute_path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}")
    return "".join(parts)


def _map_schema_error(error: ValidationError) -> StrategyDslError:
    path = _format_path(error)

    if error.validator == "required":
        return StrategyDslError(
            code="MISSING_REQUIRED_FIELD",
            message=error.message,
            path=path,
            value=getattr(error, "instance", None),
        )

    if error.validator == "type":
        return StrategyDslError(
            code="TYPE_MISMATCH",
            message=error.message,
            path=path,
            value=getattr(error, "instance", None),
        )

    if error.validator == "additionalProperties":
        unexpected_field = None
        matched = _ADDITIONAL_PROPERTY_PATTERN.search(error.message)
        if matched:
            unexpected_field = matched.group(1)
        message = error.message
        if unexpected_field:
            message = f"Unexpected field '{unexpected_field}'"
        return StrategyDslError(
            code="ADDITIONAL_PROPERTY",
            message=message,
            path=path,
            value=unexpected_field,
        )

    if error.validator == "anyOf" and tuple(error.absolute_path) == ("trade",):
        return StrategyDslError(
            code="NO_TRADE_SIDE",
            message="At least one of trade.long / trade.short must be defined.",
            path=path,
            value=getattr(error, "instance", None),
        )

    return StrategyDslError(
        code="SCHEMA_VALIDATION_ERROR",
        message=error.message,
        path=path,
        value=getattr(error, "instance", None),
    )


def validate_against_schema(payload: dict[str, Any]) -> list[StrategyDslError]:
    errors = sorted(_validator().iter_errors(payload), key=lambda item: (str(item.path), item.message))
    return [_map_schema_error(error) for error in errors]
