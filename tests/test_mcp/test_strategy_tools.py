from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.mcp.strategy import tools as strategy_tools
from src.models.session import Session as AgentSession
from src.models.user import User


class _SessionContext:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _extract_payload(call_result: object) -> dict[str, Any]:
    if isinstance(call_result, tuple) and len(call_result) == 2:
        maybe_result = call_result[1]
        if isinstance(maybe_result, dict):
            raw = maybe_result.get("result")
            if isinstance(raw, str):
                return json.loads(raw)
    raise AssertionError(f"Unexpected call result: {call_result!r}")


async def _create_session(db_session: AsyncSession, email: str) -> AgentSession:
    user = User(email=email, password_hash="hash", name=email)
    db_session.add(user)
    await db_session.flush()

    session = AgentSession(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts={},
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()
    return session


@pytest.mark.asyncio
async def test_indicator_catalog_and_detail_tools() -> None:
    mcp = FastMCP("test-strategy-tools-skills")
    strategy_tools.register_strategy_tools(mcp)

    detail_call = await mcp.call_tool("get_indicator_detail", {"indicator": "ema"})
    detail_payload = _extract_payload(detail_call)

    assert detail_payload["ok"] is True
    assert detail_payload["count"] == 1
    assert detail_payload["indicators"][0]["indicator"] == "ema"
    assert detail_payload["indicators"][0]["registry"]["name"] == "ema"

    detail_multi_call = await mcp.call_tool(
        "get_indicator_detail",
        {"indicator_list": ["ema", "rsi", "not_real_indicator"]},
    )
    detail_multi_payload = _extract_payload(detail_multi_call)
    assert detail_multi_payload["ok"] is True
    assert detail_multi_payload["count"] == 2
    assert "not_real_indicator" in detail_multi_payload["missing"]

    detail_missing_call = await mcp.call_tool(
        "get_indicator_detail",
        {"indicator": "not_real_indicator"},
    )
    detail_missing_payload = _extract_payload(detail_missing_call)
    assert detail_missing_payload["ok"] is False
    assert detail_missing_payload["error"]["code"] == "INDICATOR_NOT_FOUND"

    detail_input_error_call = await mcp.call_tool("get_indicator_detail", {})
    detail_input_error_payload = _extract_payload(detail_input_error_call)
    assert detail_input_error_payload["ok"] is False
    assert detail_input_error_payload["error"]["code"] == "INVALID_INPUT"

    catalog_call = await mcp.call_tool("get_indicator_catalog", {"category": "momentum"})
    catalog_payload = _extract_payload(catalog_call)

    assert catalog_payload["ok"] is True
    assert catalog_payload["category_filter"] == "momentum"
    assert len(catalog_payload["categories"]) == 1
    assert catalog_payload["categories"][0]["category"] == "momentum"
    assert any(item["indicator"] == "rsi" for item in catalog_payload["categories"][0]["indicators"])

    full_catalog_call = await mcp.call_tool("get_indicator_catalog", {})
    full_catalog_payload = _extract_payload(full_catalog_call)
    assert full_catalog_payload["ok"] is True
    assert "candle" not in full_catalog_payload["available_categories"]

    excluded_call = await mcp.call_tool("get_indicator_catalog", {"category": "candle"})
    excluded_payload = _extract_payload(excluded_call)
    assert excluded_payload["ok"] is False
    assert excluded_payload["error"]["code"] == "CATEGORY_EXCLUDED"

    invalid_category_call = await mcp.call_tool("get_indicator_catalog", {"category": "does-not-exist"})
    invalid_category_payload = _extract_payload(invalid_category_call)
    assert invalid_category_payload["ok"] is False
    assert invalid_category_payload["error"]["code"] == "INVALID_CATEGORY"


@pytest.mark.asyncio
async def test_strategy_validate_dsl_tool_covers_pass_and_fail() -> None:
    mcp = FastMCP("test-strategy-tools-validate")
    strategy_tools.register_strategy_tools(mcp)

    payload = load_strategy_payload(EXAMPLE_PATH)
    valid_call = await mcp.call_tool("strategy_validate_dsl", {"dsl_json": json.dumps(payload)})
    valid_payload = _extract_payload(valid_call)

    assert valid_payload["ok"] is True
    assert valid_payload["errors"] == []

    invalid_payload = deepcopy(payload)
    invalid_payload.pop("timeframe", None)
    invalid_call = await mcp.call_tool(
        "strategy_validate_dsl",
        {"dsl_json": json.dumps(invalid_payload)},
    )
    invalid_result = _extract_payload(invalid_call)

    assert invalid_result["ok"] is False
    assert invalid_result["error"]["code"] == "STRATEGY_VALIDATION_FAILED"
    assert any(item["code"] == "MISSING_REQUIRED_FIELD" for item in invalid_result["errors"])

    invalid_json_call = await mcp.call_tool(
        "strategy_validate_dsl",
        {"dsl_json": "{bad json"},
    )
    invalid_json_result = _extract_payload(invalid_json_call)
    assert invalid_json_result["ok"] is False
    assert invalid_json_result["error"]["code"] == "INVALID_JSON"


@pytest.mark.asyncio
async def test_strategy_upsert_tool_supports_create_and_update(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = await _create_session(db_session, email="strategy_mcp_a@example.com")

    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(strategy_tools, "_new_db_session", _fake_new_db_session)

    mcp = FastMCP("test-strategy-tools-upsert")
    strategy_tools.register_strategy_tools(mcp)

    payload = load_strategy_payload(EXAMPLE_PATH)
    upsert_call = await mcp.call_tool(
        "strategy_upsert_dsl",
        {
            "session_id": str(session.id),
            "dsl_json": json.dumps(payload),
        },
    )
    upsert_payload = _extract_payload(upsert_call)

    assert upsert_payload["ok"] is True
    strategy_id = upsert_payload["strategy_id"]
    assert strategy_id
    assert upsert_payload["metadata"]["version"] == 1

    updated_payload = deepcopy(payload)
    updated_payload["strategy"]["name"] = "Updated Strategy Name"
    update_call = await mcp.call_tool(
        "strategy_upsert_dsl",
        {
            "session_id": str(session.id),
            "strategy_id": strategy_id,
            "dsl_json": json.dumps(updated_payload),
        },
    )
    update_payload = _extract_payload(update_call)
    assert update_payload["ok"] is True
    assert update_payload["strategy_id"] == strategy_id
    assert update_payload["metadata"]["version"] == 2
    assert update_payload["metadata"]["strategy_name"] == "Updated Strategy Name"


@pytest.mark.asyncio
async def test_strategy_upsert_tool_reports_input_errors(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_new_db_session() -> _SessionContext:
        return _SessionContext(db_session)

    monkeypatch.setattr(strategy_tools, "_new_db_session", _fake_new_db_session)

    mcp = FastMCP("test-strategy-tools-input-errors")
    strategy_tools.register_strategy_tools(mcp)

    call_result = await mcp.call_tool(
        "strategy_upsert_dsl",
        {
            "session_id": "not-a-uuid",
            "dsl_json": "{}",
        },
    )
    payload = _extract_payload(call_result)

    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
