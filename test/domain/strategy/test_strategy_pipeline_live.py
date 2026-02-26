from __future__ import annotations

from copy import deepcopy

from packages.domain.strategy import EXAMPLE_PATH, load_strategy_payload, parse_strategy_payload, validate_strategy_payload


def test_000_accessibility_example_strategy_validates() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    validation = validate_strategy_payload(payload)
    assert validation.is_valid is True
    assert validation.errors == ()


def test_010_strategy_pipeline_rejects_missing_required_field() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    invalid_payload = deepcopy(payload)
    invalid_payload.pop("trade", None)

    validation = validate_strategy_payload(invalid_payload)
    assert validation.is_valid is False
    assert validation.errors


def test_020_strategy_pipeline_parse_returns_typed_dsl() -> None:
    payload = load_strategy_payload(EXAMPLE_PATH)
    parsed = parse_strategy_payload(payload)
    assert parsed.strategy.name
    assert parsed.universe.market
    assert parsed.universe.timeframe
