"""Live strategy-phase stream test (real OpenAI endpoint + real strategy skills)."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from src.agents.handler_registry import init_all_artifacts
from src.main import app
from src.models import database as db_module
from src.models.session import Session


def _register_and_get_token(client: TestClient) -> str:
    email = f"strategy_live_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Strategy Live User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _parse_sse_payloads(raw_text: str) -> list[dict]:
    payloads: list[dict] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _create_strategy_phase_session(
    client: TestClient,
    *,
    user_id: str,
) -> str:
    async def _insert() -> str:
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as db:
            session = Session(
                user_id=UUID(user_id),
                current_phase="strategy",
                status="active",
                artifacts=init_all_artifacts(),
                metadata_={},
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return str(session.id)

    return client.portal.call(_insert)


def test_openai_stream_strategy_phase_uses_real_prompt_and_skills() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        user_id = me.json()["user_id"]
        session_id = _create_strategy_phase_session(client, user_id=user_id)

        response = client.post(
            "/api/v1/chat/send-openai-stream?language=en",
            headers=headers,
            json={
                "session_id": session_id,
                "message": "Design a simple BTC trend-following strategy draft.",
            },
        )

    assert response.status_code == 200
    payloads = _parse_sse_payloads(response.text)
    assert payloads

    openai_events = [item for item in payloads if item.get("type") == "openai_event"]
    assert openai_events
    openai_types = [item.get("openai_type") for item in openai_events]
    assert "response.created" in openai_types
    assert "response.completed" in openai_types

    done = next(item for item in payloads if item.get("type") == "done")
    assert done.get("phase") == "strategy"

    text = "".join(item.get("delta", "") for item in payloads if item.get("type") == "text_delta")
    assert text.strip()
