#!/usr/bin/env python3
"""Backend-only E2E click-flow simulator for trading APIs (with optional real order)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload  # noqa: E402
from src.main import app  # noqa: E402


def _load_env() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-email", default="2@test.com", help="Target user email.")
    parser.add_argument("--user-password", default="pass1234", help="Target user password.")
    parser.add_argument(
        "--allow-register",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow auto-register when login fails (default: disabled).",
    )
    parser.add_argument("--symbol", default="BTCUSD", help="Trading symbol.")
    parser.add_argument("--qty", type=float, default=0.0002, help="Manual open qty.")
    parser.add_argument("--capital", type=float, default=15000, help="Deployment capital.")
    parser.add_argument("--mark-price", type=float, default=50000, help="Manual action mark price.")
    parser.add_argument("--mode", default="paper", choices=["paper"], help="Deployment mode.")
    parser.add_argument(
        "--trading-base-url",
        default="",
        help="Optional override trading base URL (default from .env).",
    )
    parser.add_argument("--wait-seconds", type=int, default=30, help="Order status polling timeout.")
    parser.add_argument(
        "--validate-broker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to validate broker credentials on create (/broker-accounts?validate=...).",
    )
    parser.add_argument(
        "--artifact-file",
        default="artifacts/e2e_trading_click_flow/latest.json",
        help="Output artifact JSON path.",
    )
    return parser.parse_args()


def _market_from_symbol(symbol: str) -> str:
    upper = symbol.strip().upper()
    if upper.endswith(("USD", "USDT", "USDC")) and len(upper) >= 6:
        return "crypto"
    return "stocks"


def _auth_token(client: TestClient, *, email: str, password: str) -> str:
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if login.status_code == 200:
        return str(login.json()["access_token"])
    raise RuntimeError(
        "login failed. "
        "Use an existing account or enable auto-register with --allow-register. "
        f"status={login.status_code} body={login.text}"
    )


def _auth_token_with_optional_register(
    client: TestClient,
    *,
    email: str,
    password: str,
    allow_register: bool,
) -> str:
    try:
        return _auth_token(client, email=email, password=password)
    except RuntimeError as exc:
        if not allow_register:
            raise
        register = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "name": "Trading E2E User"},
        )
        if register.status_code == 201:
            return str(register.json()["access_token"])
        raise RuntimeError(
            f"{exc} register={register.status_code} {register.text}"
        )


def _resolve_or_create_broker_account(
    client: TestClient,
    *,
    headers: dict[str, str],
    mode: str,
    credentials: dict[str, Any],
    validate: bool,
) -> tuple[str, str, str | None]:
    listing = client.get("/api/v1/broker-accounts", headers=headers)
    listed_rows = _assert_status(listing, 200, "broker_account_list")
    if isinstance(listed_rows, list):
        for row in listed_rows:
            if not isinstance(row, dict):
                continue
            if row.get("provider") != "alpaca":
                continue
            if row.get("mode") != mode:
                continue
            if row.get("status") != "active":
                continue
            if row.get("last_validated_status") != "paper_probe_ok":
                continue
            broker_account_id = row.get("broker_account_id")
            if broker_account_id:
                return str(broker_account_id), "reused", str(row.get("last_validated_status") or "")

    broker = client.post(
        "/api/v1/broker-accounts",
        headers=headers,
        params={"validate": validate},
        json={
            "provider": "alpaca",
            "mode": mode,
            "credentials": credentials,
            "metadata": {"label": "e2e-click-flow"},
        },
    )
    broker_body = _assert_status(broker, 201, "broker_account_create")
    return (
        str(broker_body["broker_account_id"]),
        "created",
        str(broker_body.get("last_validated_status") or ""),
    )


def _strategy_payload(symbol: str, market: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["timeframe"] = "1m"
    payload["universe"] = {"market": market, "tickers": [symbol]}
    # Keep runtime process deterministic: disable auto signal trading in this script.
    payload["trade"]["long"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 1_000_000_000}
    }
    payload["trade"]["short"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": -1}
    }
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "noop_exit_long",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": -1}},
        }
    ]
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "noop_exit_short",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 1_000_000_000}},
        }
    ]
    return payload


def _assert_status(response: Any, expected: int, step: str) -> Any:
    if response.status_code != expected:
        raise RuntimeError(f"{step} failed: {response.status_code} {response.text}")
    return response.json() if response.text else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not settings.alpaca_api_key.strip() or not settings.alpaca_api_secret.strip():
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing in backend/.env")
    if args.qty <= 0:
        raise RuntimeError("--qty must be > 0")
    if args.capital <= 0:
        raise RuntimeError("--capital must be > 0")

    settings.paper_trading_enabled = True
    settings.paper_trading_execute_orders = True
    settings.paper_trading_enqueue_on_start = False

    started_at = datetime.now(UTC).isoformat()
    market = _market_from_symbol(args.symbol)
    deployment_id: str | None = None
    summary: dict[str, Any] = {
        "started_at": started_at,
        "user_email": args.user_email,
        "symbol": args.symbol,
        "market": market,
        "steps": [],
    }

    with TestClient(app) as client:
        token = _auth_token_with_optional_register(
            client,
            email=args.user_email,
            password=args.user_password,
            allow_register=args.allow_register,
        )
        headers = {"Authorization": f"Bearer {token}"}
        summary["steps"].append({"step": "auth", "status": "ok"})

        credentials: dict[str, Any] = {
            "api_key": settings.alpaca_api_key,
            "api_secret": settings.alpaca_api_secret,
        }
        if args.trading_base_url.strip():
            credentials["trading_base_url"] = args.trading_base_url.strip()

        broker_account_id, broker_source, validation_status = _resolve_or_create_broker_account(
            client,
            headers=headers,
            mode=args.mode,
            credentials=credentials,
            validate=args.validate_broker,
        )
        summary["steps"].append(
            {
                "step": "broker_account_resolve",
                "status": "ok",
                "broker_account_id": broker_account_id,
                "source": broker_source,
                "validation_status": validation_status,
                "validation_requested": args.validate_broker,
            }
        )

        new_thread = client.post("/api/v1/chat/new-thread", headers=headers, json={"metadata": {}})
        thread_body = _assert_status(new_thread, 201, "chat_new_thread")
        session_id = thread_body["session_id"]
        summary["steps"].append({"step": "chat_new_thread", "status": "ok", "session_id": session_id})

        confirm = client.post(
            "/api/v1/strategies/confirm",
            headers=headers,
            json={
                "session_id": session_id,
                "dsl_json": _strategy_payload(args.symbol, market),
                "auto_start_backtest": False,
            },
        )
        confirm_body = _assert_status(confirm, 200, "strategy_confirm")
        strategy_id = confirm_body["strategy_id"]
        summary["steps"].append({"step": "strategy_confirm", "status": "ok", "strategy_id": strategy_id})

        create_deployment = client.post(
            "/api/v1/deployments",
            headers=headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": args.mode,
                "capital_allocated": str(args.capital),
                "risk_limits": {
                    "order_qty": args.qty,
                    "max_position_notional": max(args.capital, args.mark_price * args.qty * 4),
                    "max_symbol_exposure_pct": 0.95,
                },
                "runtime_state": {"source": "e2e_click_flow"},
            },
        )
        deployment_body = _assert_status(create_deployment, 201, "deployment_create")
        deployment_id = str(deployment_body["deployment_id"])
        summary["steps"].append({"step": "deployment_create", "status": "ok", "deployment_id": deployment_id})

        start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
        start_body = _assert_status(start, 200, "deployment_start")
        summary["steps"].append(
            {
                "step": "deployment_start",
                "status": "ok",
                "deployment_status": start_body["deployment"]["status"],
            }
        )

        manual_open = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers=headers,
            json={
                "action": "open",
                "payload": {
                    "symbol": args.symbol,
                    "side": "long",
                    "qty": args.qty,
                    "mark_price": args.mark_price,
                },
            },
        )
        manual_open_body = _assert_status(manual_open, 200, "manual_action_open")
        if manual_open_body.get("status") != "completed":
            raise RuntimeError(f"manual open not completed: {manual_open_body}")
        summary["steps"].append(
            {
                "step": "manual_action_open",
                "status": "ok",
                "action_id": manual_open_body.get("manual_trade_action_id"),
                "execution": manual_open_body.get("payload", {}).get("_execution", {}),
            }
        )

        order_rows: list[dict[str, Any]] = []
        deadline = time.monotonic() + max(args.wait_seconds, 1)
        while time.monotonic() < deadline:
            orders = client.get(f"/api/v1/deployments/{deployment_id}/orders", headers=headers)
            rows = _assert_status(orders, 200, "orders_list")
            order_rows = rows if isinstance(rows, list) else []
            if order_rows:
                break
            time.sleep(1.0)
        if not order_rows:
            raise RuntimeError("no order observed after manual open within timeout")
        latest_order = order_rows[0]
        summary["steps"].append(
            {
                "step": "orders_observed",
                "status": "ok",
                "order_id": latest_order.get("order_id"),
                "order_status": latest_order.get("status"),
                "provider_status": latest_order.get("provider_status"),
                "last_sync_at": latest_order.get("last_sync_at"),
            }
        )

        stream = client.get(
            f"/api/v1/stream/deployments/{deployment_id}",
            headers=headers,
            params={"max_events": 10, "poll_seconds": 0.2},
        )
        stream_text = stream.text
        required_events = [
            "event: deployment_status",
            "event: order_update",
            "event: fill_update",
            "event: position_update",
            "event: pnl_update",
            "event: heartbeat",
        ]
        missing_events = [event for event in required_events if event not in stream_text]
        if stream.status_code != 200 or missing_events:
            raise RuntimeError(
                f"stream probe failed: status={stream.status_code} missing={missing_events} body={stream_text[:500]}"
            )
        summary["steps"].append({"step": "stream_probe", "status": "ok", "events_checked": required_events})

        fills = client.get(f"/api/v1/deployments/{deployment_id}/fills", headers=headers)
        fills_body = _assert_status(fills, 200, "fills_list")
        summary["steps"].append(
            {
                "step": "fills_snapshot",
                "status": "ok",
                "fills_count": len(fills_body) if isinstance(fills_body, list) else 0,
            }
        )

        positions = client.get(f"/api/v1/deployments/{deployment_id}/positions", headers=headers)
        positions_body = _assert_status(positions, 200, "positions_list")
        summary["steps"].append(
            {
                "step": "positions_snapshot",
                "status": "ok",
                "positions_count": len(positions_body) if isinstance(positions_body, list) else 0,
            }
        )

        portfolio = client.get(f"/api/v1/deployments/{deployment_id}/portfolio", headers=headers)
        portfolio_body = _assert_status(portfolio, 200, "portfolio_snapshot")
        summary["steps"].append(
            {
                "step": "portfolio_snapshot",
                "status": "ok",
                "equity": portfolio_body.get("equity"),
                "cash": portfolio_body.get("cash"),
            }
        )

        stop = client.post(f"/api/v1/deployments/{deployment_id}/stop", headers=headers)
        _assert_status(stop, 200, "deployment_stop")
        summary["steps"].append({"step": "deployment_stop", "status": "ok"})

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["result"] = "ok"
    return summary


def main() -> None:
    _load_env()
    args = _parse_args()
    result = run(args)
    artifact = Path(args.artifact_file).resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"artifact={artifact}")


if __name__ == "__main__":
    main()
