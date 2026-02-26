"""JSON schema validation for strategy DSL payloads."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from packages.domain.strategy.errors import (
    StrategyDslError,
    json_path_to_pointer,
    normalize_strategy_error,
)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SCHEMA_PATH = _ASSETS_DIR / "strategy_dsl_schema.json"

_ADDITIONAL_PROPERTY_PATTERN = re.compile(r"'([^']+)' was unexpected")
_REQUIRED_PROPERTY_PATTERN = re.compile(r"'([^']+)' is a required property")


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
    pointer = json_path_to_pointer(path)
    instance = getattr(error, "instance", None)

    if error.validator == "required":
        missing_field = None
        matched = _REQUIRED_PROPERTY_PATTERN.search(error.message)
        if matched:
            missing_field = matched.group(1)
        return StrategyDslError(
            code="MISSING_REQUIRED_FIELD",
            message=error.message,
            path=path,
            value=instance,
            stage="schema",
            pointer=pointer,
            expected=missing_field or error.validator_value,
            actual=sorted(instance.keys()) if isinstance(instance, dict) else instance,
            suggestion=(
                f"Add required field '{missing_field}' at '{path}'."
                if missing_field
                else "Add the missing required field."
            ),
        )

    if error.validator == "type":
        return StrategyDslError(
            code="TYPE_MISMATCH",
            message=error.message,
            path=path,
            value=instance,
            stage="schema",
            pointer=pointer,
            expected=error.validator_value,
            actual=type(instance).__name__,
            suggestion="Use the value type required by the schema.",
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
            stage="schema",
            pointer=pointer,
            expected="No additional properties are allowed.",
            actual=unexpected_field,
            suggestion=(
                f"Remove unexpected field '{unexpected_field}'."
                if unexpected_field
                else "Remove unexpected fields not defined in the schema."
            ),
        )

    if error.validator == "anyOf" and tuple(error.absolute_path) == ("trade",):
        return StrategyDslError(
            code="NO_TRADE_SIDE",
            message="At least one of trade.long / trade.short must be defined.",
            path=path,
            value=instance,
            stage="schema",
            pointer=pointer,
            expected="trade.long or trade.short",
            actual=instance,
            suggestion="Define at least one trade side: trade.long or trade.short.",
        )

    mapped = StrategyDslError(
        code="SCHEMA_VALIDATION_ERROR",
        message=error.message,
        path=path,
        value=instance,
    )
    return normalize_strategy_error(
        mapped,
        stage="schema",
        expected=error.validator_value,
        actual=instance,
        suggestion="Adjust the payload to satisfy the DSL JSON schema.",
    )


def validate_against_schema(payload: dict[str, Any]) -> list[StrategyDslError]:
    errors = sorted(_validator().iter_errors(payload), key=lambda item: (str(item.path), item.message))
    return [_map_schema_error(error) for error in errors]
