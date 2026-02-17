"""Live E2E from pre-strategy -> strategy -> post-confirm backtest request."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.agents.handler_registry import init_all_artifacts
from src.main import app
from src.models import database as db_module
from src.models.session import Session


_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _register_and_get_token(client: TestClient) -> str:
    email = f"pre_strategy_e2e_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Pre Strategy E2E User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _create_pre_strategy_session(
    client: TestClient,
    *,
    user_id: str,
) -> str:
    async def _insert() -> str:
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as db:
            session = Session(
                user_id=UUID(user_id),
                current_phase="pre_strategy",
                status="active",
                artifacts=init_all_artifacts(),
                metadata_={},
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return str(session.id)

    return client.portal.call(_insert)


def _send_stream(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
    message: str,
    language: str = "zh",
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    response = client.post(
        f"/api/v1/chat/send-openai-stream?language={language}",
        headers=headers,
        json={"session_id": session_id, "message": message},
    )
    assert response.status_code == 200
    payloads = _parse_sse_payloads(response.text)
    done = next(item for item in payloads if item.get("type") == "done")
    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    return payloads, done, text


def _session_detail(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
) -> dict[str, Any]:
    response = client.get(f"/api/v1/sessions/{session_id}", headers=headers)
    assert response.status_code == 200
    return response.json()


def _latest_assistant_mcp_calls(detail: dict[str, Any]) -> list[dict[str, Any]]:
    messages = detail.get("messages", [])
    assistant_messages = [
        item for item in messages if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    for message in reversed(assistant_messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        mcp_calls = [
            item
            for item in tool_calls
            if isinstance(item, dict) and str(item.get("type", "")).strip().lower() == "mcp_call"
        ]
        if mcp_calls:
            return mcp_calls
    return []


def _coerce_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _extract_strategy_draft_id(detail: dict[str, Any]) -> str | None:
    messages = detail.get("messages", [])
    assistant_messages = [
        item for item in messages if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    for message in reversed(assistant_messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for item in reversed(tool_calls):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip().lower() != "mcp_call":
                continue
            if str(item.get("name", "")).strip() != "strategy_validate_dsl":
                continue
            payload = _coerce_payload(item.get("output")) or _coerce_payload(item.get("result"))
            if not isinstance(payload, dict):
                continue
            draft_id = payload.get("strategy_draft_id")
            if isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id):
                return draft_id
    return None


def test_full_pre_strategy_to_strategy_flow_no_manual_session_intervention_live() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        user_id = me.json()["user_id"]
        session_id = _create_pre_strategy_session(client, user_id=user_id)

        # Step 1: complete pre-strategy with exact enum values.
        for _ in range(4):
            detail = _session_detail(client, headers=headers, session_id=session_id)
            if detail.get("current_phase") == "strategy":
                break
            pre_profile = ((detail.get("artifacts") or {}).get("pre_strategy") or {}).get("profile") or {}
            missing_fields = (
                ((detail.get("artifacts") or {}).get("pre_strategy") or {}).get("missing_fields")
                or []
            )
            parts: list[str] = []
            if "target_market" in missing_fields and not pre_profile.get("target_market"):
                parts.append("target_market=us_stocks")
            if "target_instrument" in missing_fields and not pre_profile.get("target_instrument"):
                parts.append("target_instrument=SPY")
            if (
                "opportunity_frequency_bucket" in missing_fields
                and not pre_profile.get("opportunity_frequency_bucket")
            ):
                parts.append("opportunity_frequency_bucket=few_per_week")
            if "holding_period_bucket" in missing_fields and not pre_profile.get("holding_period_bucket"):
                parts.append("holding_period_bucket=swing_days")
            if not parts:
                parts = [
                    "target_market=us_stocks",
                    "target_instrument=SPY",
                    "opportunity_frequency_bucket=few_per_week",
                    "holding_period_bucket=swing_days",
                ]
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message="Please set these exactly: " + ", ".join(parts),
                language="en",
            )

        detail = _session_detail(client, headers=headers, session_id=session_id)
        assert detail.get("current_phase") == "strategy", detail.get("artifacts")

        # Step 2: ask for strategy draft + validation.
        _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message="Please draft and validate one complete strategy DSL now.",
            language="en",
        )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        draft_id = _extract_strategy_draft_id(detail)
        assert isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id), detail.get("messages")

        # Step 3: simulate frontend confirm-save.
        draft_response = client.get(f"/api/v1/strategies/drafts/{draft_id}", headers=headers)
        assert draft_response.status_code == 200
        draft_payload = draft_response.json()
        dsl_json = draft_payload["dsl_json"]
        confirm_response = client.post(
            "/api/v1/strategies/confirm",
            headers=headers,
            json={
                "session_id": session_id,
                "dsl_json": dsl_json,
                "auto_start_backtest": False,
                "language": "zh",
            },
        )
        assert confirm_response.status_code == 200
        strategy_id = confirm_response.json()["strategy_id"]
        assert isinstance(strategy_id, str) and _UUID_PATTERN.search(strategy_id)

        # Step 4: request 2024 backtest; user should not need manual session operation.
        _, _, assistant_text = _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message="请用2024年整年的数据进行回测",
            language="zh",
        )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        mcp_calls = _latest_assistant_mcp_calls(detail)
        assert mcp_calls, detail.get("messages")

        strategy_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_get_dsl"
        ]
        assert strategy_get_calls, mcp_calls
        assert any(str(item.get("status", "")).strip() == "success" for item in strategy_get_calls), mcp_calls

        failed_invalid_session = [
            item
            for item in strategy_get_calls
            if str(item.get("status", "")).strip() == "failure"
            and "invalid_session_id" in str(item.get("error", "")).lower()
        ]
        assert not failed_invalid_session, strategy_get_calls
        assert "session_id" not in assistant_text.lower()
