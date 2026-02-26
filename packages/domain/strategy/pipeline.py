"""End-to-end DSL validation/parsing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.domain.strategy.errors import (
    StrategyDslValidationException,
    StrategyDslValidationResult,
)
from packages.domain.strategy.parser import build_parsed_strategy
from packages.domain.strategy.schema import validate_against_schema
from packages.domain.strategy.semantic import validate_strategy_semantics


def load_strategy_payload(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load strategy payload from dict or JSON file path."""
    if isinstance(source, dict):
        return dict(source)

    path = Path(source).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def validate_strategy_payload(
    payload: dict[str, Any],
    *,
    allow_temporal: bool = False,
) -> StrategyDslValidationResult:
    """Run schema validation first, then semantic validation."""
    schema_errors = validate_against_schema(payload)
    if schema_errors:
        return StrategyDslValidationResult(
            is_valid=False,
            errors=tuple(schema_errors),
        )

    semantic_errors = validate_strategy_semantics(payload, allow_temporal=allow_temporal)
    return StrategyDslValidationResult(
        is_valid=not semantic_errors,
        errors=tuple(semantic_errors),
    )


def parse_strategy_payload(
    payload: dict[str, Any],
    *,
    allow_temporal: bool = False,
):
    """Validate and parse a strategy payload into typed objects."""
    validation = validate_strategy_payload(payload, allow_temporal=allow_temporal)
    if not validation.is_valid:
        raise StrategyDslValidationException(list(validation.errors))
    return build_parsed_strategy(payload)
