from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.engine.strategy import (
    EXAMPLE_PATH,
    parse_strategy_payload,
    validate_strategy_payload,
)
from src.engine.strategy.pipeline import load_strategy_payload
from src.engine.strategy.semantic import validate_strategy_semantics


@pytest.fixture
def example_payload() -> dict[str, Any]:
    return load_strategy_payload(EXAMPLE_PATH)


def _replace_value(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, dict):
        return {key: _replace_value(value, old, new) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_value(item, old, new) for item in obj]
    if obj == old:
        return new
    return obj


def _codes(result) -> set[str]:
    return {item.code for item in result.errors}


def test_example_strategy_passes_schema_and_semantic_validation(example_payload: dict[str, Any]) -> None:
    result = validate_strategy_payload(example_payload)
    assert result.is_valid is True
    assert result.errors == ()

    parsed = parse_strategy_payload(example_payload)
    assert parsed.strategy.name == "EMA Cross + RSI Filter + ATR Risk (Demo)"
    assert parsed.universe.market == example_payload["universe"]["market"]
    assert parsed.universe.timeframe == example_payload["timeframe"]
    assert "macd_12_26_9" in parsed.factors


def test_schema_error_for_missing_required_field(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload.pop("timeframe", None)

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "MISSING_REQUIRED_FIELD" in _codes(result)
    assert any("required property" in item.message for item in result.errors)


def test_schema_error_for_additional_property(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["extra_field"] = {"foo": 1}

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "ADDITIONAL_PROPERTY" in _codes(result)
    assert any("extra_field" in item.message for item in result.errors)


def test_schema_error_for_type_mismatch(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["timeframe"] = 4

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "TYPE_MISMATCH" in _codes(result)


def test_schema_error_for_unsupported_timeframe_value(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["timeframe"] = "1w"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "SCHEMA_VALIDATION_ERROR" in _codes(result)
    assert any(item.path == "$.timeframe" for item in result.errors)


def test_schema_error_for_missing_trade_side(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"] = {}

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "NO_TRADE_SIDE" in _codes(result)
    assert any(item.path == "$.trade" for item in result.errors)


def test_schema_error_for_invalid_not_structure(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"] = {
        "not": [
            {
                "cmp": {
                    "left": {"ref": "rsi_14"},
                    "op": "lt",
                    "right": 30,
                }
            }
        ]
    }

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "SCHEMA_VALIDATION_ERROR" in _codes(result)


def test_semantic_error_for_unknown_factor_ref(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"]["all"][0]["cross"]["a"]["ref"] = "ema_999"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "UNKNOWN_FACTOR_REF" in _codes(result)
    assert any("Unknown factor reference" in item.message for item in result.errors)


def test_semantic_error_for_unsupported_factor_type(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["factors"]["ema_9"]["type"] = "not_real_indicator"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "UNSUPPORTED_FACTOR_TYPE" in _codes(result)
    assert any(item.path == "$.factors.ema_9.type" for item in result.errors)


def test_semantic_error_for_invalid_factor_param_value(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["factors"]["ema_9"]["params"]["period"] = 0
    payload["factors"]["ema_9"]["params"]["source"] = "close"
    payload = _replace_value(payload, "ema_9", "ema_0")
    payload["factors"]["ema_0"] = payload["factors"].pop("ema_9")

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "INVALID_FACTOR_PARAM_VALUE" in _codes(result)
    assert any("must be >=" in item.message for item in result.errors)


def test_semantic_error_for_unsupported_factor_param(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["factors"]["ema_9"]["params"]["unexpected"] = 123

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "UNSUPPORTED_FACTOR_PARAM" in _codes(result)
    assert any(item.path == "$.factors.ema_9.params" for item in result.errors)


def test_semantic_error_for_invalid_output_name(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"]["all"][3]["cmp"]["left"]["ref"] = "macd_12_26_9.bad_output"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "INVALID_OUTPUT_NAME" in _codes(result)
    assert any("Invalid output" in item.message for item in result.errors)


def test_semantic_error_for_multi_output_ref_without_dot_notation(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"]["all"][3]["cmp"]["left"]["ref"] = "macd_12_26_9"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "INVALID_OUTPUT_NAME" in _codes(result)
    assert any("dot notation" in item.message for item in result.errors)


def test_semantic_error_for_known_multi_output_ref_without_dot_even_with_single_alias(
    example_payload: dict[str, Any],
) -> None:
    payload = deepcopy(example_payload)
    payload["factors"]["macd_12_26_9"]["outputs"] = ["line_only"]
    payload["trade"]["long"]["entry"]["condition"]["all"][3]["cmp"]["left"]["ref"] = "macd_12_26_9"
    payload["trade"]["long"]["entry"]["condition"]["all"][3]["cmp"]["right"]["ref"] = "macd_12_26_9"
    payload["trade"]["short"]["entry"]["condition"]["all"][2]["cmp"]["left"]["ref"] = "macd_12_26_9"
    payload["trade"]["short"]["entry"]["condition"]["all"][2]["cmp"]["right"]["ref"] = "macd_12_26_9"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "INVALID_OUTPUT_NAME" in _codes(result)
    assert any("multi-output" in item.message for item in result.errors)


def test_semantic_allows_single_output_factor_alias_ref(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["factors"]["ema_9"]["outputs"] = ["alias"]
    payload = _replace_value(payload, "ema_9", "ema_9.alias")

    result = validate_strategy_payload(payload)
    assert result.is_valid is True
    assert result.errors == ()


def test_semantic_error_for_factor_id_mismatch(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    ema_9 = payload["factors"].pop("ema_9")
    payload["factors"]["ema_fast"] = ema_9
    payload = _replace_value(payload, "ema_9", "ema_fast")

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "FACTOR_ID_MISMATCH" in _codes(result)
    assert any("should be" in item.message for item in result.errors)


def test_semantic_error_for_temporal_condition(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"] = {
        "temporal": {
            "type": "within_bars",
            "bars": 5,
            "condition": {
                "cmp": {
                    "left": {"ref": "rsi_14"},
                    "op": "lt",
                    "right": 30,
                }
            },
        }
    }

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "TEMPORAL_NOT_SUPPORTED" in _codes(result)
    assert any("not supported" in item.message for item in result.errors)


def test_semantic_error_for_invalid_atr_ref(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["exits"][1]["stop"]["atr_ref"] = "ema_21"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "MISSING_ATR_REF" in _codes(result)
    assert any("atr_ref" in item.message for item in result.errors)


def test_semantic_error_for_future_look_offset(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["trade"]["long"]["entry"]["condition"]["all"][1]["cmp"]["left"]["offset"] = 1

    errors = validate_strategy_semantics(payload)
    assert any(item.code == "FUTURE_LOOK" for item in errors)
    assert any("offset must be <= 0" in item.message for item in errors)


def test_semantic_error_for_unsupported_dsl_major_version(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    payload["dsl_version"] = "2.1.0"

    result = validate_strategy_payload(payload)
    assert result.is_valid is False
    assert "UNSUPPORTED_DSL_VERSION" in _codes(result)
    assert any(item.path == "$.dsl_version" for item in result.errors)


def test_semantic_direct_check_for_factor_id_format_error(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    bad_factor = payload["factors"].pop("ema_9")
    payload["factors"]["EMA_FAST"] = bad_factor
    payload = _replace_value(payload, "ema_9", "EMA_FAST")

    errors = validate_strategy_semantics(payload)
    assert any(item.code == "FACTOR_ID_FORMAT_ERROR" for item in errors)


def test_semantic_direct_check_for_bracket_rr_conflict(example_payload: dict[str, Any]) -> None:
    payload = deepcopy(example_payload)
    # Deliberately break semantic contract; schema would reject this earlier.
    payload["trade"]["long"]["exits"][2]["take"] = {"kind": "pct", "value": 0.03}

    errors = validate_strategy_semantics(payload)
    assert any(item.code == "BRACKET_RR_CONFLICT" for item in errors)


def test_load_strategy_payload_from_path_round_trip(tmp_path: Path, example_payload: dict[str, Any]) -> None:
    file_path = tmp_path / "strategy.json"
    file_path.write_text(json.dumps(example_payload), encoding="utf-8")

    loaded = load_strategy_payload(file_path)
    assert loaded["strategy"]["name"] == example_payload["strategy"]["name"]
