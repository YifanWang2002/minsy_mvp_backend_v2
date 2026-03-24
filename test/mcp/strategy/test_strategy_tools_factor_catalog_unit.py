from __future__ import annotations

import json
from typing import Any

from apps.mcp.domains.strategy import tools as strategy_tools


class _FakeMCP:
    def __init__(self) -> None:
        self.registered_names: list[str] = []

    def tool(self):  # noqa: ANN201 - mirrors FastMCP decorator API
        def _decorator(fn):  # noqa: ANN001, ANN202
            self.registered_names.append(str(getattr(fn, "__name__", "")))
            return fn

        return _decorator


def _catalog_payload(*, category: str = "") -> dict[str, Any]:
    raw = strategy_tools.get_indicator_catalog(category=category)
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


def test_strategy_tool_names_exclude_indicator_detail() -> None:
    assert "get_indicator_detail" not in strategy_tools.TOOL_NAMES
    assert "get_indicator_catalog" in strategy_tools.TOOL_NAMES


def test_register_strategy_tools_exposes_catalog_but_not_detail() -> None:
    fake = _FakeMCP()
    strategy_tools.register_strategy_tools(fake)  # type: ignore[arg-type]

    assert "get_indicator_catalog" in fake.registered_names
    assert "get_indicator_detail" not in fake.registered_names


def test_indicator_catalog_exposes_expected_categories_and_fields() -> None:
    payload = _catalog_payload()
    assert payload.get("ok") is True
    assert payload.get("tool") == "get_indicator_catalog"
    assert payload.get("skills_enabled") is False

    categories = payload.get("categories")
    assert isinstance(categories, list) and categories
    category_names = {item.get("category") for item in categories if isinstance(item, dict)}
    expected = {"overlap", "momentum", "volatility", "volume", "utils"}
    assert expected.issubset(category_names)
    assert "candle" not in category_names

    first_indicator: dict[str, Any] | None = None
    for category_payload in categories:
        if not isinstance(category_payload, dict):
            continue
        indicators = category_payload.get("indicators")
        if not isinstance(indicators, list) or not indicators:
            continue
        if isinstance(indicators[0], dict):
            first_indicator = indicators[0]
            break

    assert isinstance(first_indicator, dict)
    for required_key in (
        "indicator",
        "full_name",
        "description",
        "params",
        "outputs",
        "required_columns",
        "version",
        "status",
        "deprecated_since",
        "replacement",
        "remove_after",
    ):
        assert required_key in first_indicator
    assert "skill_path" not in first_indicator
    assert "skill_summary" not in first_indicator


def test_indicator_catalog_rejects_invalid_category() -> None:
    payload = _catalog_payload(category="not_a_real_category")
    assert payload.get("ok") is False
    assert payload.get("error", {}).get("code") == "INVALID_CATEGORY"
    assert isinstance(payload.get("available_categories"), list)

