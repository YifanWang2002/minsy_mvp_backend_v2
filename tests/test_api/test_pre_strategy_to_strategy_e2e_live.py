"""Live E2E from pre-strategy -> strategy -> post-confirm backtest request."""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.agents.handler_registry import init_all_artifacts
from src.config import settings
from src.engine.backtest.service import execute_backtest_job
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
            if str(item.get("type", "")).strip().lower() != "strategy_ref":
                continue
            draft_id = item.get("strategy_draft_id")
            if isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id):
                return draft_id
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


def _is_invalid_session_error(call: dict[str, Any]) -> bool:
    haystacks: list[str] = []
    error_text = call.get("error")
    if isinstance(error_text, str):
        haystacks.append(error_text)
    output_payload = _coerce_payload(call.get("output")) or _coerce_payload(call.get("result"))
    if isinstance(output_payload, dict):
        haystacks.append(json.dumps(output_payload, ensure_ascii=False))
    merged = " ".join(haystacks).lower()
    return ("invalid_session_id" in merged) or ("invalid session_id" in merged)


def _is_transient_http_424_error(call: dict[str, Any]) -> bool:
    haystacks: list[str] = []
    error_text = call.get("error")
    if isinstance(error_text, str):
        haystacks.append(error_text)
    output_payload = _coerce_payload(call.get("output")) or _coerce_payload(call.get("result"))
    if isinstance(output_payload, dict):
        haystacks.append(json.dumps(output_payload, ensure_ascii=False))
    merged = " ".join(haystacks).lower()
    return ("http_error" in merged) and ("424" in merged)


def _contains_manual_refresh_prompt(text: str) -> bool:
    normalized = text.lower()
    negative_hints = (
        "不需要你刷新",
        "不需要刷新",
        "无需刷新",
        "不用刷新",
        "不需要重连",
        "无需重连",
        "no need to refresh",
        "no need to reconnect",
        "don't need to refresh",
        "do not need to refresh",
    )
    if any(item in normalized for item in negative_hints):
        return False

    positive_hints = (
        "please refresh",
        "please reconnect",
        "please resync",
        "refresh strategy",
        "resync strategy",
        "请刷新",
        "请重连",
        "请重新同步",
        "需要刷新",
        "需要重连",
        "需要重新同步",
        "重新同步后",
        "刷新后",
        "重连后",
        "provide session_id",
        "send session_id",
        "share session_id",
        "提供session_id",
        "发送session_id",
        "给我session_id",
    )
    return any(item in normalized for item in positive_hints)


def _tool_calls(calls: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [item for item in calls if str(item.get("name", "")).strip() == name]


def _has_tool_success(calls: list[dict[str, Any]], name: str) -> bool:
    return any(str(item.get("status", "")).strip() == "success" for item in _tool_calls(calls, name))


def _extract_job_id_from_calls(calls: list[dict[str, Any]]) -> str | None:
    for item in reversed(calls):
        if str(item.get("name", "")).strip() != "backtest_create_job":
            continue
        if str(item.get("status", "")).strip() != "success":
            continue
        payload = _coerce_payload(item.get("output")) or _coerce_payload(item.get("result"))
        if not isinstance(payload, dict):
            continue
        job_id = payload.get("job_id")
        if isinstance(job_id, str) and _UUID_PATTERN.search(job_id):
            return job_id
    for item in reversed(calls):
        if str(item.get("name", "")).strip() != "backtest_get_job":
            continue
        if str(item.get("status", "")).strip() != "success":
            continue
        payload = _coerce_payload(item.get("output")) or _coerce_payload(item.get("result"))
        if not isinstance(payload, dict):
            continue
        job_id = payload.get("job_id")
        if isinstance(job_id, str) and _UUID_PATTERN.search(job_id):
            return job_id
    return None


def _force_complete_backtest_job(client: TestClient, *, job_id: str) -> None:
    async def _run() -> None:
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as db:
            await execute_backtest_job(db, job_id=UUID(job_id), auto_commit=True)

    client.portal.call(_run)


def _run_until_required_tools_success(
    client: TestClient,
    *,
    headers: dict[str, str],
    session_id: str,
    message: str,
    required_tools: list[str],
    language: str = "zh",
    max_turns: int = 6,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    current_message = message
    last_calls: list[dict[str, Any]] = []
    last_text = ""
    last_detail: dict[str, Any] = {}
    for _ in range(max_turns):
        _, _, text = _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message=current_message,
            language=language,
        )
        assert not _contains_manual_refresh_prompt(text), text
        detail = _session_detail(client, headers=headers, session_id=session_id)
        calls = _latest_assistant_mcp_calls(detail)
        assert calls, detail.get("messages")
        last_calls = calls
        last_text = text
        last_detail = detail

        invalid_session_calls = [
            item
            for item in calls
            if str(item.get("name", "")).strip() in required_tools and _is_invalid_session_error(item)
        ]
        assert not invalid_session_calls, calls

        missing: list[str] = []
        transient_only = True
        for tool_name in required_tools:
            tool_calls = _tool_calls(calls, tool_name)
            if any(str(item.get("status", "")).strip() == "success" for item in tool_calls):
                continue
            missing.append(tool_name)
            if not tool_calls or not all(_is_transient_http_424_error(item) for item in tool_calls):
                transient_only = False

        if not missing:
            return calls, text, detail
        if transient_only:
            continue

        current_message = (
            "Retry now in this same session. "
            f"You must successfully call these MCP tools in this turn: {', '.join(missing)}. "
            "Do not ask me to refresh/resync/reconnect/provide session_id."
        )

    raise AssertionError(
        {
            "required_tools": required_tools,
            "last_calls": last_calls,
            "last_text": last_text,
            "last_detail": last_detail.get("messages"),
        }
    )


def test_full_pre_strategy_to_strategy_flow_no_manual_session_intervention_live() -> None:
    assert settings.strategy_mcp_server_url == "https://dev.minsyai.com/mcp"
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        user_id = me.json()["user_id"]
        session_id = _create_pre_strategy_session(client, user_id=user_id)

        # Step 1: complete pre-strategy with exact enum values.
        for _ in range(6):
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
        # Live endpoint may transiently return MCP 424; retry a few turns before failing.
        draft_id: str | None = None
        detail: dict[str, Any] = {}
        for _ in range(4):
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message="Please draft and validate one complete strategy DSL now.",
                language="en",
            )
            detail = _session_detail(client, headers=headers, session_id=session_id)
            draft_id = _extract_strategy_draft_id(detail)
            if isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id):
                break

            mcp_calls = _latest_assistant_mcp_calls(detail)
            validate_calls = [
                item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_validate_dsl"
            ]
            if validate_calls and all(_is_transient_http_424_error(item) for item in validate_calls):
                continue
            # Non-transient/no-call case: break early and surface assertion with detail.
            break

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

        # Step 4: read saved strategy DSL first; this must not fail on session compatibility.
        _, _, read_text = _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message="请先读取我刚保存的策略DSL，并告诉我 market、ticker、timeframe（不要让我刷新或提供session）",
            language="zh",
        )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        mcp_calls = _latest_assistant_mcp_calls(detail)
        assert mcp_calls, detail.get("messages")
        strategy_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_get_dsl"
        ]
        assert strategy_get_calls, mcp_calls
        invalid_session_calls = [item for item in strategy_get_calls if _is_invalid_session_error(item)]
        assert not invalid_session_calls, strategy_get_calls
        has_get_success = any(
            str(item.get("status", "")).strip() == "success" for item in strategy_get_calls
        )
        if not has_get_success:
            assert all(_is_transient_http_424_error(item) for item in strategy_get_calls), strategy_get_calls
        assert "session_id" not in read_text.lower()
        assert "刷新" not in read_text
        assert "reconnect" not in read_text.lower()

        # Step 5: request 2024 backtest; user should still not need manual session operation.
        _, _, backtest_text = _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message="请用2024年整年的数据进行回测，并立即查询回测任务状态",
            language="zh",
        )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        mcp_calls = _latest_assistant_mcp_calls(detail)
        assert mcp_calls, detail.get("messages")

        strategy_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_get_dsl"
        ]
        assert strategy_get_calls, mcp_calls
        invalid_session_calls = [item for item in strategy_get_calls if _is_invalid_session_error(item)]
        assert not invalid_session_calls, strategy_get_calls
        has_get_success = any(
            str(item.get("status", "")).strip() == "success" for item in strategy_get_calls
        )
        if not has_get_success:
            assert all(_is_transient_http_424_error(item) for item in strategy_get_calls), strategy_get_calls

        backtest_create_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "backtest_create_job"
        ]
        backtest_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "backtest_get_job"
        ]
        assert backtest_create_calls, mcp_calls
        assert any(str(item.get("status", "")).strip() == "success" for item in backtest_create_calls), mcp_calls
        assert backtest_get_calls, mcp_calls
        assert any(str(item.get("status", "")).strip() == "success" for item in backtest_get_calls), mcp_calls
        assert "session_id" not in backtest_text.lower()


def test_saved_strategy_modification_flow_no_manual_refresh_live() -> None:
    assert settings.strategy_mcp_server_url == "https://dev.minsyai.com/mcp"
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        user_id = me.json()["user_id"]
        session_id = _create_pre_strategy_session(client, user_id=user_id)

        # Complete pre-strategy.
        for _ in range(4):
            detail = _session_detail(client, headers=headers, session_id=session_id)
            if detail.get("current_phase") == "strategy":
                break
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message=(
                    "Please set these exactly: "
                    "target_market=us_stocks, target_instrument=SPY, "
                    "opportunity_frequency_bucket=few_per_week, "
                    "holding_period_bucket=swing_days"
                ),
                language="en",
            )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        assert detail.get("current_phase") == "strategy", detail.get("artifacts")

        # Draft+validate with retry tolerance for transient 424.
        draft_id: str | None = None
        for _ in range(6):
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message="Please draft and validate one complete strategy DSL now.",
                language="en",
            )
            detail = _session_detail(client, headers=headers, session_id=session_id)
            draft_id = _extract_strategy_draft_id(detail)
            if isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id):
                break
            mcp_calls = _latest_assistant_mcp_calls(detail)
            validate_calls = [
                item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_validate_dsl"
            ]
            if validate_calls and all(_is_transient_http_424_error(item) for item in validate_calls):
                continue
            break
        assert isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id), detail.get("messages")

        # Confirm save.
        draft_response = client.get(f"/api/v1/strategies/drafts/{draft_id}", headers=headers)
        assert draft_response.status_code == 200
        confirm_response = client.post(
            "/api/v1/strategies/confirm",
            headers=headers,
            json={
                "session_id": session_id,
                "dsl_json": draft_response.json()["dsl_json"],
                "auto_start_backtest": False,
                "language": "zh",
            },
        )
        assert confirm_response.status_code == 200
        strategy_id = confirm_response.json()["strategy_id"]
        assert isinstance(strategy_id, str) and _UUID_PATTERN.search(strategy_id)

        # Modification phase: no manual refresh/resync should be requested.
        modified = False
        for _ in range(4):
            _, _, modify_text = _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message=(
                    "请把已保存策略做最小改动：把 strategy name 改成 AutoPatch-Name-1，其他不变；"
                    "请直接修改保存，不要让我刷新/重连/重新同步。"
                ),
                language="zh",
            )
            assert not _contains_manual_refresh_prompt(modify_text), modify_text

            detail = _session_detail(client, headers=headers, session_id=session_id)
            mcp_calls = _latest_assistant_mcp_calls(detail)
            assert mcp_calls, detail.get("messages")

            strategy_get_calls = [
                item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_get_dsl"
            ]
            if strategy_get_calls:
                invalid_session_calls = [
                    item for item in strategy_get_calls if _is_invalid_session_error(item)
                ]
                assert not invalid_session_calls, strategy_get_calls

            patch_success = any(
                str(item.get("name", "")).strip() == "strategy_patch_dsl"
                and str(item.get("status", "")).strip() == "success"
                for item in mcp_calls
            )
            upsert_success = any(
                str(item.get("name", "")).strip() == "strategy_upsert_dsl"
                and str(item.get("status", "")).strip() == "success"
                for item in mcp_calls
            )
            if patch_success or upsert_success:
                modified = True
                break

            # If not modified yet and all strategy_get failures are transient 424, retry.
            if strategy_get_calls and all(_is_transient_http_424_error(item) for item in strategy_get_calls):
                continue
            break

        assert modified, "Strategy modification was not applied successfully."

        # Backtest create/get should still work after modification.
        _, _, backtest_text = _send_stream(
            client,
            headers=headers,
            session_id=session_id,
            message="请用2024年整年的数据进行回测，并立即查询回测任务状态",
            language="zh",
        )
        assert not _contains_manual_refresh_prompt(backtest_text), backtest_text
        detail = _session_detail(client, headers=headers, session_id=session_id)
        mcp_calls = _latest_assistant_mcp_calls(detail)
        assert mcp_calls, detail.get("messages")

        strategy_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "strategy_get_dsl"
        ]
        if strategy_get_calls:
            invalid_session_calls = [item for item in strategy_get_calls if _is_invalid_session_error(item)]
            assert not invalid_session_calls, strategy_get_calls

        backtest_create_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "backtest_create_job"
        ]
        backtest_get_calls = [
            item for item in mcp_calls if str(item.get("name", "")).strip() == "backtest_get_job"
        ]
        assert backtest_create_calls, mcp_calls
        assert any(str(item.get("status", "")).strip() == "success" for item in backtest_create_calls), mcp_calls
        assert backtest_get_calls, mcp_calls
        assert any(str(item.get("status", "")).strip() == "success" for item in backtest_get_calls), mcp_calls


def test_full_mcp_tool_coverage_no_manual_actions_live() -> None:
    assert settings.strategy_mcp_server_url == "https://dev.minsyai.com/mcp"
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        user_id = me.json()["user_id"]
        session_id = _create_pre_strategy_session(client, user_id=user_id)

        # 1) Pre-strategy completion.
        for _ in range(4):
            detail = _session_detail(client, headers=headers, session_id=session_id)
            if detail.get("current_phase") == "strategy":
                break
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message=(
                    "Please set these exactly: "
                    "target_market=us_stocks, target_instrument=SPY, "
                    "opportunity_frequency_bucket=few_per_week, "
                    "holding_period_bucket=swing_days"
                ),
                language="en",
            )
        detail = _session_detail(client, headers=headers, session_id=session_id)
        assert detail.get("current_phase") == "strategy", detail.get("artifacts")

        # 2) Validate (strategy_validate_dsl) -> draft_id.
        draft_id: str | None = None
        for _ in range(6):
            _send_stream(
                client,
                headers=headers,
                session_id=session_id,
                message="Please draft and validate one complete strategy DSL now.",
                language="en",
            )
            detail = _session_detail(client, headers=headers, session_id=session_id)
            draft_id = _extract_strategy_draft_id(detail)
            if isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id):
                break
            calls = _latest_assistant_mcp_calls(detail)
            validate_calls = _tool_calls(calls, "strategy_validate_dsl")
            if validate_calls and all(_is_transient_http_424_error(item) for item in validate_calls):
                continue
            break
        assert isinstance(draft_id, str) and _UUID_PATTERN.search(draft_id), detail.get("messages")

        # 3) Confirm save via API.
        draft_response = client.get(f"/api/v1/strategies/drafts/{draft_id}", headers=headers)
        assert draft_response.status_code == 200
        confirm_response = client.post(
            "/api/v1/strategies/confirm",
            headers=headers,
            json={
                "session_id": session_id,
                "dsl_json": draft_response.json()["dsl_json"],
                "auto_start_backtest": False,
                "language": "zh",
            },
        )
        assert confirm_response.status_code == 200
        strategy_id = confirm_response.json()["strategy_id"]
        assert isinstance(strategy_id, str) and _UUID_PATTERN.search(strategy_id)

        # 4) strategy_get_dsl + strategy_list_tunable_params
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请针对 strategy_id={strategy_id} 调用 strategy_get_dsl 和 "
                "strategy_list_tunable_params，并简要汇总。"
            ),
            required_tools=["strategy_get_dsl", "strategy_list_tunable_params"],
            language="zh",
        )

        # 5) strategy_patch_dsl
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请对 strategy_id={strategy_id} 做最小 patch：把 strategy.name 改为 "
                "'Toolchain-Patch-A'，并保存。"
            ),
            required_tools=["strategy_get_dsl", "strategy_patch_dsl"],
            language="zh",
        )

        # 6) strategy_list_versions + strategy_get_version_dsl + strategy_diff_versions
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请对 strategy_id={strategy_id} 执行版本检查："
                "先 strategy_list_versions(limit=5)，再选择相邻两个版本执行 "
                "strategy_get_version_dsl 和 strategy_diff_versions。"
            ),
            required_tools=[
                "strategy_list_versions",
                "strategy_get_version_dsl",
                "strategy_diff_versions",
            ],
            language="zh",
        )

        # 7) strategy_rollback_dsl
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请对 strategy_id={strategy_id} 回滚到上一个版本（使用 strategy_rollback_dsl），"
                "并再调用 strategy_get_dsl 验证。"
            ),
            required_tools=["strategy_rollback_dsl", "strategy_get_dsl"],
            language="zh",
        )

        # 8) strategy_upsert_dsl
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请读取 strategy_id={strategy_id} 的最新 dsl_json，"
                "然后用 strategy_upsert_dsl 仅在 strategy.description 末尾追加 "
                "' [upsert-check]' 并保存。"
            ),
            required_tools=["strategy_get_dsl", "strategy_upsert_dsl"],
            language="zh",
        )

        # 9) Indicator tools.
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message="请调用 get_indicator_catalog(category='momentum') 和 get_indicator_detail(['ema','rsi'])。",
            required_tools=["get_indicator_catalog", "get_indicator_detail"],
            language="zh",
        )

        # 10) Coverage + backtest create/get.
        calls, _, _ = _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请对 strategy_id={strategy_id} 执行回测准备和提交："
                "strategy_get_dsl -> get_symbol_data_coverage -> backtest_create_job(2024全年，必要时按覆盖裁剪) "
                "-> backtest_get_job。"
            ),
            required_tools=[
                "strategy_get_dsl",
                "get_symbol_data_coverage",
                "backtest_create_job",
                "backtest_get_job",
            ],
            language="zh",
        )
        job_id = _extract_job_id_from_calls(calls)
        assert isinstance(job_id, str) and _UUID_PATTERN.search(job_id)

        # Force completion so analytics tools can run without JOB_NOT_READY.
        _force_complete_backtest_job(client, job_id=job_id)

        # 11) Analytics tools (group 1).
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请针对 job_id={job_id} 依次调用："
                "backtest_get_job, backtest_entry_hour_pnl_heatmap, "
                "backtest_entry_weekday_pnl, backtest_monthly_return_table。"
            ),
            required_tools=[
                "backtest_get_job",
                "backtest_entry_hour_pnl_heatmap",
                "backtest_entry_weekday_pnl",
                "backtest_monthly_return_table",
            ],
            language="zh",
        )

        # 12) Analytics tools (group 2).
        _run_until_required_tools_success(
            client,
            headers=headers,
            session_id=session_id,
            message=(
                f"请针对 job_id={job_id} 依次调用："
                "backtest_holding_period_pnl_bins, backtest_long_short_breakdown, "
                "backtest_exit_reason_breakdown, backtest_underwater_curve, "
                "backtest_rolling_metrics。"
            ),
            required_tools=[
                "backtest_holding_period_pnl_bins",
                "backtest_long_short_breakdown",
                "backtest_exit_reason_breakdown",
                "backtest_underwater_curve",
                "backtest_rolling_metrics",
            ],
            language="zh",
        )
