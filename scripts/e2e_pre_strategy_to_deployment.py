#!/usr/bin/env python3
"""Deterministic backend E2E: pre-strategy -> strategy -> deployment -> paper trading runtime."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.engine.execution.adapters.base import OhlcvBar  # noqa: E402
from src.engine.market_data.runtime import market_data_runtime  # noqa: E402
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload  # noqa: E402
from src.main import app  # noqa: E402
from src.models import database as db_module  # noqa: E402
from src.models.deployment import Deployment  # noqa: E402
from src.models.deployment_run import DeploymentRun  # noqa: E402
from src.models.fill import Fill  # noqa: E402
from src.models.order import Order  # noqa: E402
from src.models.session import Session  # noqa: E402
from src.models.signal_event import SignalEvent  # noqa: E402

_MOCK_RESPONSES: list[str] = [
    (
        "已记录你的经验。"
        '<AGENT_STATE_PATCH>{"trading_years_bucket":"years_5_plus"}</AGENT_STATE_PATCH>'
    ),
    (
        "风险偏好已记录。"
        '<AGENT_STATE_PATCH>{"risk_tolerance":"aggressive"}</AGENT_STATE_PATCH>'
    ),
    (
        "KYC 完成。"
        '<AGENT_STATE_PATCH>{"return_expectation":"high_growth"}</AGENT_STATE_PATCH>'
    ),
    (
        "市场范围已记录。"
        '<AGENT_STATE_PATCH>{"target_market":"crypto"}</AGENT_STATE_PATCH>'
    ),
    (
        "标的已记录。"
        '<AGENT_STATE_PATCH>{"target_instrument":"BTCUSD"}</AGENT_STATE_PATCH>'
    ),
    (
        "机会频率已记录。"
        '<AGENT_STATE_PATCH>{"opportunity_frequency_bucket":"daily"}</AGENT_STATE_PATCH>'
    ),
    (
        "持有周期已记录。"
        '<AGENT_STATE_PATCH>{"holding_period_bucket":"swing_days"}</AGENT_STATE_PATCH>'
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-email",
        default="2@test.com",
        help="Target backend user email for the whole E2E flow.",
    )
    parser.add_argument(
        "--user-password",
        default="pass1234",
        help="Target backend user password.",
    )
    parser.add_argument(
        "--allow-register",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow auto-register when login fails (default: disabled).",
    )
    parser.add_argument("--symbol", default="BTCUSD", help="Strategy symbol.")
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Deployment capital allocation.",
    )
    parser.add_argument(
        "--order-qty",
        type=float,
        default=0.02,
        help="Risk limit order qty for runtime signal cycle.",
    )
    parser.add_argument(
        "--artifact-file",
        default="artifacts/e2e_pre_strategy_to_deployment/latest.json",
        help="Output artifact JSON path.",
    )
    parser.add_argument(
        "--disable-auto-enqueue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable celery auto enqueue on deployment start and run one runtime cycle via API.",
    )
    parser.add_argument(
        "--real-openai",
        action="store_true",
        help="Use real OpenAI responses stream instead of mock patching.",
    )
    parser.add_argument(
        "--validate-broker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate broker credentials on create when a reusable account is not found.",
    )
    return parser.parse_args()


def _assert_status(response: Any, expected_status: int, step: str) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise RuntimeError(f"{step} failed: {response.status_code} {response.text}")
    return response.json() if response.text else {}


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    for block in blocks:
        for line in block.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def _send_turn(
    client: TestClient,
    *,
    headers: dict[str, str],
    message: str,
    session_id: str | None,
) -> tuple[dict[str, Any], bool]:
    payload: dict[str, Any] = {"message": message}
    if session_id is not None:
        payload["session_id"] = session_id

    response = client.post(
        "/api/v1/chat/send-openai-stream?language=zh",
        headers=headers,
        json=payload,
    )
    if response.status_code != 200:
        raise RuntimeError(f"chat turn failed: {response.status_code} {response.text}")

    sse_payloads = _parse_sse_payloads(response.text)
    openai_event_seen = any(
        isinstance(item, dict) and item.get("type") == "openai_event"
        for item in sse_payloads
    )
    done_payload = next((item for item in sse_payloads if item.get("type") == "done"), None)
    if not isinstance(done_payload, dict):
        raise RuntimeError("chat turn missing done event")
    return done_payload, openai_event_seen


def _auth_token(
    client: TestClient,
    *,
    email: str,
    password: str,
    allow_register: bool,
) -> tuple[str, str]:
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if login.status_code == 200:
        return str(login.json()["access_token"]), "login"
    if not allow_register:
        raise RuntimeError(
            "login failed and auto-register is disabled. "
            f"email={email} status={login.status_code} body={login.text}"
        )

    register = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "E2E Deploy User"},
    )
    if register.status_code == 201:
        return str(register.json()["access_token"]), "register"
    raise RuntimeError(
        "auth failed: "
        f"login={login.status_code} {login.text} "
        f"register={register.status_code} {register.text}"
    )


def _resolve_or_create_broker_account(
    client: TestClient,
    *,
    headers: dict[str, str],
    validate: bool,
) -> tuple[str, str]:
    existing = client.get("/api/v1/broker-accounts", headers=headers)
    existing_rows = _assert_status(existing, 200, "broker_account_list")
    if isinstance(existing_rows, list):
        for row in existing_rows:
            if not isinstance(row, dict):
                continue
            if row.get("provider") != "alpaca":
                continue
            if row.get("mode") != "paper":
                continue
            if row.get("status") != "active":
                continue
            if row.get("last_validated_status") != "paper_probe_ok":
                continue
            broker_account_id = row.get("broker_account_id")
            if broker_account_id:
                return str(broker_account_id), "reused"

    if not settings.alpaca_api_key.strip() or not settings.alpaca_api_secret.strip():
        raise RuntimeError(
            "No reusable validated paper broker account found, and ALPACA_API_KEY/ALPACA_API_SECRET is missing."
        )

    credentials: dict[str, Any] = {
        "api_key": settings.alpaca_api_key,
        "api_secret": settings.alpaca_api_secret,
    }
    if settings.alpaca_trading_base_url.strip():
        credentials["trading_base_url"] = settings.alpaca_trading_base_url.strip()
    broker = client.post(
        "/api/v1/broker-accounts",
        headers=headers,
        params={"validate": validate},
        json={
            "provider": "alpaca",
            "mode": "paper",
            "credentials": credentials,
            "metadata": {"label": "e2e-pre-strategy-deploy"},
        },
    )
    broker_body = _assert_status(broker, 201, "broker_account_create")
    return str(broker_body["broker_account_id"]), "created"


def _build_strategy_payload(symbol: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["strategy"]["name"] = f"{symbol} E2E Deployment Strategy"
    payload["strategy"]["description"] = "Deterministic E2E strategy for deployment runtime probe."
    payload["timeframe"] = "1m"
    payload["universe"] = {"market": "crypto", "tickers": [symbol]}

    payload["trade"]["long"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}
    }
    payload["trade"]["short"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": -1}
    }
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "never_exit_long",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": -1}},
        }
    ]
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "never_exit_short",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
        }
    ]
    return payload


def _seed_market_data(symbol: str) -> None:
    market_data_runtime.reset()
    start = datetime(2026, 1, 6, 9, 30, tzinfo=UTC)
    base_close = Decimal("50000")
    for i in range(3):
        ts = start + timedelta(minutes=i)
        close = base_close + Decimal(str(i * 10))
        bar = OhlcvBar(
            timestamp=ts,
            open=close - Decimal("5"),
            high=close + Decimal("5"),
            low=close - Decimal("10"),
            close=close,
            volume=Decimal("10"),
        )
        market_data_runtime.ingest_1m_bar(
            market="crypto",
            symbol=symbol,
            bar=bar,
        )


async def _fetch_db_snapshot(deployment_id: UUID, session_id: UUID) -> dict[str, Any]:
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        deployment = await db.scalar(select(Deployment).where(Deployment.id == deployment_id))
        if deployment is None:
            raise RuntimeError(f"Deployment not found in DB: {deployment_id}")

        run_status = await db.scalar(
            select(DeploymentRun.status)
            .where(DeploymentRun.deployment_id == deployment_id)
            .order_by(DeploymentRun.created_at.desc())
            .limit(1)
        )
        order_count = int(
            await db.scalar(select(func.count(Order.id)).where(Order.deployment_id == deployment_id))
            or 0
        )
        fill_count = int(
            await db.scalar(
                select(func.count(Fill.id))
                .join(Order, Fill.order_id == Order.id)
                .where(Order.deployment_id == deployment_id)
            )
            or 0
        )
        signal_count = int(
            await db.scalar(
                select(func.count(SignalEvent.id)).where(SignalEvent.deployment_id == deployment_id)
            )
            or 0
        )

        session = await db.scalar(select(Session).where(Session.id == session_id))
        if session is None:
            raise RuntimeError(f"Session not found in DB: {session_id}")

        deployment_profile = (
            ((session.artifacts or {}).get("deployment") or {}).get("profile")
            if isinstance(session.artifacts, dict)
            else {}
        )
        return {
            "deployment_status": deployment.status,
            "deployment_mode": deployment.mode,
            "run_status": run_status,
            "order_count": order_count,
            "fill_count": fill_count,
            "signal_count": signal_count,
            "session_phase": session.current_phase,
            "session_deployment_profile": deployment_profile if isinstance(deployment_profile, dict) else {},
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.capital <= 0:
        raise RuntimeError("--capital must be > 0")
    if args.order_qty <= 0:
        raise RuntimeError("--order-qty must be > 0")

    result: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "symbol": args.symbol,
        "capital": args.capital,
        "order_qty": args.order_qty,
        "openai_mode": "real" if args.real_openai else "mock",
        "steps": [],
    }

    original_enqueue = settings.paper_trading_enqueue_on_start
    settings.paper_trading_enqueue_on_start = (
        False if args.disable_auto_enqueue else original_enqueue
    )

    call_index = {"value": 0}

    async def _mock_stream_events(
        *,
        model: str,
        input_text: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        reasoning: dict | None = None,
    ):
        del model, input_text, instructions, previous_response_id, tools, tool_choice, reasoning
        idx = call_index["value"]
        call_index["value"] += 1
        text = _MOCK_RESPONSES[idx] if idx < len(_MOCK_RESPONSES) else "收到。"
        yield {
            "type": "response.output_text.delta",
            "delta": text,
            "sequence_number": 1,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_e2e_{idx}",
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            },
        }

    openai_event_seen_any = False
    patch_context = nullcontext()
    if not args.real_openai:
        patch_context = patch(
            "src.services.openai_stream_service.OpenAIResponsesEventStreamer.stream_events",
            side_effect=_mock_stream_events,
        )

    try:
        with patch_context:
            with TestClient(app) as client:
                access_token, auth_mode = _auth_token(
                    client,
                    email=args.user_email,
                    password=args.user_password,
                    allow_register=args.allow_register,
                )
                headers = {"Authorization": f"Bearer {access_token}"}
                result["steps"].append(
                    {
                        "step": "auth",
                        "mode": auth_mode,
                        "email": args.user_email,
                        "status": "ok",
                    }
                )

                session_id: str | None = None
                for turn_no, message in enumerate(
                    [
                        "我有5年以上交易经验。",
                        "我的风险偏好是激进。",
                        "我的收益预期是高增长。",
                        "我想做 crypto 市场。",
                        "标的是 BTCUSD。",
                        "机会频率 daily。",
                        "持有周期 swing_days。",
                    ],
                    start=1,
                ):
                    done_payload, openai_event_seen = _send_turn(
                        client,
                        headers=headers,
                        message=message,
                        session_id=session_id,
                    )
                    openai_event_seen_any = openai_event_seen_any or openai_event_seen
                    session_id = str(done_payload.get("session_id") or session_id)
                    if session_id is None:
                        raise RuntimeError("missing session_id in done payload")
                    result["steps"].append(
                        {
                            "step": f"chat_turn_{turn_no}",
                            "phase": done_payload.get("phase"),
                            "missing_fields": done_payload.get("missing_fields"),
                            "status": "ok",
                        }
                    )

                assert session_id is not None
                detail_before = _assert_status(
                    client.get(f"/api/v1/sessions/{session_id}", headers=headers),
                    200,
                    "session_detail_before_confirm",
                )
                final_phase = detail_before.get("current_phase")
                if final_phase != "strategy":
                    raise RuntimeError(f"expected strategy phase, got {final_phase}")

                strategy_payload = _build_strategy_payload(args.symbol)
                confirm = client.post(
                    "/api/v1/strategies/confirm",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "dsl_json": strategy_payload,
                        "auto_start_backtest": False,
                    },
                )
                confirm_body = _assert_status(confirm, 200, "strategy_confirm")
                strategy_id = str(confirm_body["strategy_id"])
                if str(confirm_body.get("phase")) != "deployment":
                    raise RuntimeError(f"strategy confirm did not move to deployment: {confirm_body}")
                result["steps"].append(
                    {
                        "step": "strategy_confirm",
                        "strategy_id": strategy_id,
                        "phase": confirm_body.get("phase"),
                        "status": "ok",
                    }
                )

                broker_account_id, broker_mode = _resolve_or_create_broker_account(
                    client,
                    headers=headers,
                    validate=args.validate_broker,
                )
                result["steps"].append(
                    {
                        "step": "broker_account_resolve",
                        "mode": broker_mode,
                        "broker_account_id": broker_account_id,
                        "validate_requested": args.validate_broker,
                        "status": "ok",
                    }
                )

                create_deployment = client.post(
                    "/api/v1/deployments",
                    headers=headers,
                    json={
                        "strategy_id": strategy_id,
                        "broker_account_id": broker_account_id,
                        "mode": "paper",
                        "capital_allocated": str(args.capital),
                        "risk_limits": {
                            "order_qty": args.order_qty,
                            "max_position_notional": max(args.capital, args.order_qty * 100000),
                            "max_symbol_exposure_pct": 0.9,
                        },
                        "runtime_state": {"source": "e2e_pre_strategy_to_deployment"},
                    },
                )
                deployment_body = _assert_status(create_deployment, 201, "deployment_create")
                deployment_id = str(deployment_body["deployment_id"])
                result["steps"].append(
                    {
                        "step": "deployment_create",
                        "deployment_id": deployment_id,
                        "status": "ok",
                    }
                )

                start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
                start_body = _assert_status(start, 200, "deployment_start")
                result["steps"].append(
                    {
                        "step": "deployment_start",
                        "deployment_status": start_body.get("deployment", {}).get("status"),
                        "run_status": start_body.get("deployment", {}).get("run", {}).get("status"),
                        "queued_task_id": start_body.get("queued_task_id"),
                        "status": "ok",
                    }
                )

                _seed_market_data(args.symbol)

                process_events: list[dict[str, Any]] = []
                orders_body: list[dict[str, Any]] = []
                for _ in range(3):
                    process_resp = client.post(
                        f"/api/v1/deployments/{deployment_id}/process-now",
                        headers=headers,
                    )
                    process_body = _assert_status(process_resp, 200, "deployment_process_now")
                    process_events.append(process_body)
                    orders_resp = client.get(
                        f"/api/v1/deployments/{deployment_id}/orders",
                        headers=headers,
                    )
                    orders_json = _assert_status(orders_resp, 200, "deployment_orders")
                    orders_body = orders_json if isinstance(orders_json, list) else []
                    if orders_body:
                        break

                result["steps"].append(
                    {
                        "step": "paper_trading_runtime_cycle",
                        "attempts": len(process_events),
                        "last_signal": process_events[-1].get("signal") if process_events else None,
                        "execution_event_id": (
                            process_events[-1].get("execution_event_id") if process_events else None
                        ),
                        "orders_count": len(orders_body),
                        "status": "ok",
                    }
                )

                if not process_events or process_events[-1].get("execution_event_id") is None:
                    raise RuntimeError("paper trading cycle did not emit execution_event_id")
                if not orders_body:
                    raise RuntimeError("paper trading cycle did not create orders")

                db_snapshot = client.portal.call(
                    _fetch_db_snapshot,
                    UUID(deployment_id),
                    UUID(session_id),
                )
                result["db_snapshot"] = db_snapshot

                checks = {
                    "reached_strategy_phase": final_phase == "strategy",
                    "strategy_confirm_to_deployment": str(confirm_body.get("phase")) == "deployment",
                    "deployment_active_in_db": db_snapshot.get("deployment_status") == "active",
                    "runtime_run_status_started": db_snapshot.get("run_status") in {"starting", "running"},
                    "signals_persisted": int(db_snapshot.get("signal_count", 0)) > 0,
                    "orders_persisted": int(db_snapshot.get("order_count", 0)) > 0,
                    "fills_persisted": int(db_snapshot.get("fill_count", 0)) > 0,
                }
                if args.real_openai:
                    checks["openai_stream_seen"] = openai_event_seen_any
                result["checks"] = checks
                if not all(checks.values()):
                    raise RuntimeError(f"one or more checks failed: {checks}")
    finally:
        settings.paper_trading_enqueue_on_start = original_enqueue

    result["finished_at"] = datetime.now(UTC).isoformat()
    result["result"] = "ok"
    return result


def main() -> None:
    args = _parse_args()
    result = run(args)

    artifact_path = Path(args.artifact_file).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
