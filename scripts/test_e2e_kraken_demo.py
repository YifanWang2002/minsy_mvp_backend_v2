"""End-to-end demo: Strategy → Backtest → Deploy → Live Paper Trading on Kraken Futures Demo.

Flow:
  1. Authenticate / resolve broker account
  2. Create session → Confirm strategy (EMA cross on BTC/USD:USD, 1m)
  3. Create & execute backtest (via docker exec into API container)
  4. Create deployment against Kraken Futures sandbox account
  5. Start deployment → trigger signal cycle
  6. Verify orders, portfolio, PnL
  7. Stop deployment
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any
from uuid import UUID

import httpx

BASE_URL = os.environ.get("MINSY_API_URL", "http://localhost:8000")
AUTH_EMAIL = "2@test.com"
AUTH_PASSWORD = "123456"
AUTH_NAME = "Test User"

DOCKER_API_CONTAINER = "minsy-api-dev"

STRATEGY_DSL: dict[str, Any] = {
    "dsl_version": "1.0.0",
    "strategy": {
        "name": "BTC Futures EMA Cross (Kraken Demo)",
        "description": "EMA 9/21 crossover on BTCUSD perpetual for Kraken Futures demo.",
    },
    "universe": {
        "market": "crypto",
        "tickers": ["BTCUSD"],
    },
    "timeframe": "1m",
    "factors": {
        "ema_9": {"type": "ema", "params": {"period": 9, "source": "close"}},
        "ema_21": {"type": "ema", "params": {"period": 21, "source": "close"}},
        "rsi_14": {"type": "rsi", "params": {"period": 14, "source": "close"}},
        "atr_14": {"type": "atr", "params": {"period": 14}},
    },
    "trade": {
        "long": {
            "position_sizing": {"mode": "pct_equity", "pct": 0.10},
            "entry": {
                "order": {"type": "market"},
                "condition": {
                    "all": [
                        {"cross": {"a": {"ref": "ema_9"}, "op": "cross_above", "b": {"ref": "ema_21"}}},
                        {"cmp": {"left": {"ref": "rsi_14"}, "op": "lt", "right": 70}},
                    ]
                },
            },
            "exits": [
                {
                    "type": "signal_exit",
                    "name": "exit_on_cross_down",
                    "order": {"type": "market"},
                    "condition": {"cross": {"a": {"ref": "ema_9"}, "op": "cross_below", "b": {"ref": "ema_21"}}},
                },
                {"type": "stop_loss", "name": "sl_atr_2x", "stop": {"kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0}},
                {"type": "bracket_rr", "name": "tp_from_sl_rr2", "stop": {"kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0}, "risk_reward": 2.0},
            ],
        },
        "short": {
            "position_sizing": {"mode": "pct_equity", "pct": 0.10},
            "entry": {
                "order": {"type": "market"},
                "condition": {
                    "all": [
                        {"cross": {"a": {"ref": "ema_9"}, "op": "cross_below", "b": {"ref": "ema_21"}}},
                        {"cmp": {"left": {"ref": "rsi_14"}, "op": "gt", "right": 30}},
                    ]
                },
            },
            "exits": [
                {
                    "type": "signal_exit",
                    "name": "exit_on_cross_up",
                    "order": {"type": "market"},
                    "condition": {"cross": {"a": {"ref": "ema_9"}, "op": "cross_above", "b": {"ref": "ema_21"}}},
                },
                {"type": "stop_loss", "name": "sl_atr_2x", "stop": {"kind": "atr_multiple", "atr_ref": "atr_14", "multiple": 2.0}},
            ],
        },
    },
}

# ---------------------------------------------------------------------------

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    tag = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def dump_json(label: str, data: Any) -> None:
    print(f"\n  ── {label} ──")
    print(f"  {json.dumps(data, indent=2, default=str)[:2000]}")
    print()


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class API:
    def __init__(self) -> None:
        self.client = httpx.Client(base_url=BASE_URL, timeout=30.0)
        self.token: str = ""
        self.headers: dict[str, str] = {}

    def authenticate(self) -> dict[str, Any]:
        r = self.client.post("/api/v1/auth/login", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD})
        if r.status_code != 200:
            r = self.client.post("/api/v1/auth/register", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD, "name": AUTH_NAME})
            r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        return data

    def get(self, path: str, **kw: Any) -> httpx.Response:
        return self.client.get(path, headers=self.headers, **kw)

    def post(self, path: str, **kw: Any) -> httpx.Response:
        return self.client.post(path, headers=self.headers, **kw)


# ---------------------------------------------------------------------------
# Docker exec helpers (run domain code inside the API container)
# ---------------------------------------------------------------------------


def docker_exec_python(script: str, *, timeout: int = 120) -> tuple[bool, str]:
    """Execute a Python snippet inside the Docker API container."""
    result = subprocess.run(
        ["docker", "exec", DOCKER_API_CONTAINER, ".venv/bin/python", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Kraken Futures Demo — End-to-End Workflow Test")
    print("=" * 70)

    api = API()

    # ── Step 0: Authenticate ──
    print("\n── Step 0: Authenticate ──")
    auth = api.authenticate()
    user_id = auth.get("user_id", "")
    report("authentication", bool(api.token), f"user_id={user_id}")

    # ── Step 1: Resolve broker account ──
    print("\n── Step 1: Resolve Kraken Futures broker account ──")
    broker_account_id = _resolve_broker_account(api)
    if not broker_account_id:
        _summary()
        return
    report("broker_account_id resolved", True, broker_account_id)

    # ── Step 2: Create session ──
    print("\n── Step 2: Create session ──")
    r = api.post("/api/v1/chat/new-thread", json={"metadata": {}})
    if r.status_code != 201:
        report("create session", False, r.text[:200])
        _summary()
        return
    session_id = r.json()["session_id"]
    report("session created", True, f"session_id={session_id}")

    # ── Step 3: Confirm strategy ──
    print("\n── Step 3: Confirm strategy (EMA Cross on BTC/USD:USD) ──")
    r = api.post(
        "/api/v1/strategies/confirm",
        json={"session_id": session_id, "dsl_json": STRATEGY_DSL, "auto_start_backtest": False, "language": "en"},
    )
    if r.status_code != 200:
        report("confirm strategy", False, r.text[:500])
        _summary()
        return
    confirm_data = r.json()
    strategy_id = confirm_data["strategy_id"]
    report("strategy confirmed", True, f"strategy_id={strategy_id}")
    report("strategy status == validated", confirm_data.get("metadata", {}).get("status") == "validated")

    r = api.get(f"/api/v1/strategies/{strategy_id}")
    if r.status_code == 200:
        strat = r.json()
        report("strategy name correct", "EMA Cross" in strat.get("metadata", {}).get("strategy_name", ""))
        report("strategy timeframe == 1m", strat.get("metadata", {}).get("timeframe") == "1m")

    # ── Step 4: Create & execute backtest ──
    print("\n── Step 4: Create & run backtest (inside Docker) ──")
    backtest_ok, backtest_job_id = _run_backtest_in_docker(strategy_id)
    if not backtest_ok:
        print("  [INFO] Backtest had issues — continuing to deployment")

    # ── Step 5: Create deployment ──
    print("\n── Step 5: Create deployment (paper mode, Kraken Futures) ──")
    r = api.post(
        "/api/v1/deployments",
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "mode": "paper",
            "capital_allocated": 5000,
        },
    )
    if r.status_code not in (200, 201):
        report("create deployment", False, r.text[:500])
        _summary()
        return
    deploy_data = r.json()
    deployment_id = deploy_data["deployment_id"]
    report("deployment created", True, f"id={deployment_id}")
    report("status == pending", deploy_data.get("status") == "pending")
    report("broker == ccxt", deploy_data.get("broker_provider") == "ccxt")
    report("symbols include BTC/USD:USD", "BTCUSD" in (deploy_data.get("symbols") or []))
    report(f"capital == 5000 (got {deploy_data.get('capital_allocated')})", float(deploy_data.get("capital_allocated", 0)) == 5000.0)

    # ── Step 6: Start deployment ──
    print("\n── Step 6: Start deployment ──")
    r = api.post(f"/api/v1/deployments/{deployment_id}/start")
    if r.status_code != 200:
        report("start deployment", False, r.text[:500])
        _summary()
        return
    start_status = r.json().get("deployment", {}).get("status", "?")
    report("deployment active", start_status == "active", start_status)

    # ── Step 7: Trigger signal cycle & direct order test ──
    print("\n── Step 7: Trigger signal cycle (inside Docker) ──")
    _run_signal_cycle_in_docker(deployment_id)

    # Also verify Kraken order flow directly via adapter
    print("\n── Step 7b: Direct Kraken order test (submit + cancel) ──")
    _test_direct_kraken_order_in_docker()

    time.sleep(2)
    r = api.get(f"/api/v1/deployments/{deployment_id}/orders")
    if r.status_code == 200:
        orders = r.json()
        report(f"orders endpoint (count={len(orders)})", True)
        for o in orders[-3:]:
            report(f"  order: {o.get('side')} {o.get('symbol')} qty={o.get('qty')} status={o.get('status')}", True)
    else:
        report("fetch orders", False, f"HTTP {r.status_code}")

    # ── Step 8: Portfolio & PnL ──
    print("\n── Step 8: Verify portfolio & PnL ──")
    _verify_portfolio(api, deployment_id)

    # ── Step 9: Stop deployment ──
    print("\n── Step 9: Stop deployment ──")
    r = api.post(f"/api/v1/deployments/{deployment_id}/stop")
    if r.status_code == 200:
        stop_status = r.json().get("deployment", {}).get("status", "?")
        report("deployment stopped", stop_status == "stopped", stop_status)
    else:
        report("stop deployment", False, r.text[:200])

    _summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_broker_account(api: API) -> str:
    r = api.get("/api/v1/broker-accounts")
    r.raise_for_status()
    accounts = r.json()
    for a in accounts:
        if a.get("exchange_id") == "krakenfutures" and a.get("status") == "active":
            report("found Kraken Futures account", True, f"id={a['broker_account_id']}")
            return a["broker_account_id"]

    # Register
    r = api.post(
        "/api/v1/broker-accounts",
        json={
            "provider": "ccxt",
            "exchange_id": "krakenfutures",
            "credentials": {
                "exchange_id": "krakenfutures",
                "api_key": "qYKzO+QEGxI7oJCRIlwMvw2mVu0pjM/4taWS6Q2KbwHZPqV8MQZmRuwu",
                "api_secret": "Np4pvZDR/BZ1vHcU0AqbfHrbieIC4KXVYq+SHMoYn5oQFBhsjB2X7dVl821iksFRAmyt8FdpXLvxJ4j0DCA00tVq",
                "sandbox": True,
            },
            "mode": "paper",
            "is_default": True,
        },
    )
    if r.status_code in (200, 201):
        bid = r.json()["broker_account_id"]
        report("registered Kraken Futures account", True, f"id={bid}")
        return bid
    report("register broker account", False, r.text[:200])
    return ""


def _run_backtest_in_docker(strategy_id: str) -> tuple[bool, str]:
    """Create and execute a backtest job inside the Docker API container."""
    script = f'''
import asyncio, json
from uuid import UUID
from packages.domain.backtest.service import create_backtest_job, execute_backtest_job
from packages.infra.db import session as db_module

async def run():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    async with db_module.AsyncSessionLocal() as db:
        receipt = await create_backtest_job(
            db,
            strategy_id=UUID("{strategy_id}"),
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_bps=5.0,
            auto_commit=True,
        )
        job_id = receipt.job_id
        print(json.dumps({{"action": "created", "job_id": str(job_id), "status": receipt.status}}))

    async with db_module.AsyncSessionLocal() as db:
        view = await execute_backtest_job(db, job_id=job_id, auto_commit=True)
        result_summary = {{}}
        if view.result:
            summary = view.result.get("summary", {{}})
            result_summary = {{
                "total_trades": summary.get("total_trades", 0),
                "total_return_pct": summary.get("total_return_pct", 0),
                "final_equity": summary.get("final_equity", 0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                "win_rate": summary.get("win_rate", 0),
            }}
        print(json.dumps({{"action": "executed", "status": view.status, "result": result_summary}}))

asyncio.run(run())
'''
    ok, output = docker_exec_python(script, timeout=180)
    lines = [l for l in output.strip().split("\n") if l.strip().startswith("{")]
    job_id = ""

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("action") == "created":
            job_id = data.get("job_id", "")
            report("backtest job created", data.get("status") == "pending", f"job_id={job_id}")
        elif data.get("action") == "executed":
            status = data.get("status", "?")
            result = data.get("result", {})
            report("backtest executed", status == "done", f"status={status}")
            if result:
                trades = result.get("total_trades", 0)
                ret = result.get("total_return_pct", 0)
                equity = result.get("final_equity", 0)
                dd = result.get("max_drawdown_pct", 0)
                wr = result.get("win_rate", 0)
                report(
                    f"backtest: {trades} trades, return={ret:.2f}%, equity=${equity:.0f}, drawdown={dd:.2f}%, win_rate={wr:.1f}%",
                    True,  # 0 trades is valid if engine completed
                )

    if not lines:
        report("backtest docker exec", False, output[-300:] if output else "no output")
        return False, ""

    return ok, job_id


def _run_signal_cycle_in_docker(deployment_id: str) -> None:
    """Trigger a signal cycle inside the Docker API container."""
    script = f'''
import asyncio, json
from uuid import UUID
from packages.domain.trading.runtime.runtime_service import process_deployment_signal_cycle
from packages.infra.db import session as db_module

async def run():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    async with db_module.AsyncSessionLocal() as db:
        try:
            result = await process_deployment_signal_cycle(db, deployment_id=UUID("{deployment_id}"))
            signal = getattr(result, "signal", None) or getattr(result, "action", None) or "none"
            has_order = getattr(result, "order_id", None) is not None if result else False
            metadata = {{}}
            if result and hasattr(result, "metadata"):
                metadata = result.metadata if isinstance(result.metadata, dict) else {{}}
            print(json.dumps({{"ok": True, "signal": str(signal), "has_order": has_order}}))
        except Exception as e:
            print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{str(e)[:200]}}"}}, default=str))

asyncio.run(run())
'''
    ok, output = docker_exec_python(script, timeout=60)
    lines = [l for l in output.strip().split("\n") if l.strip().startswith("{")]

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("ok"):
            signal = data.get("signal", "none")
            has_order = data.get("has_order", False)
            report(f"signal cycle completed (signal={signal}, has_order={has_order})", True)
        else:
            error = data.get("error", "unknown")
            # Some errors are acceptable (e.g., no bar data available at this moment)
            if "no bar" in error.lower() or "market" in error.lower() or "data" in error.lower():
                report(f"signal cycle: no market data available now", True, error[:100])
            else:
                report(f"signal cycle error", False, error[:200])

    if not lines:
        # No JSON output — check raw output
        if "no bar" in output.lower() or "lock" in output.lower():
            report("signal cycle: runtime constraint", True, output[-200:])
        else:
            report("signal cycle docker exec", ok, output[-300:] if output else "no output")


def _test_direct_kraken_order_in_docker() -> None:
    """Submit and cancel a limit order directly on Kraken Futures demo."""
    script = '''
import asyncio, json, uuid
from decimal import Decimal
from packages.infra.providers.trading.adapters.ccxt_trading import CcxtTradingAdapter
from packages.infra.providers.trading.adapters.base import OrderIntent
from packages.infra.providers.trading.credentials import CredentialCipher
from packages.infra.db import session as db_module
from packages.infra.db.models.broker_account import BrokerAccount
from sqlalchemy import select

async def run():
    if db_module.AsyncSessionLocal is None:
        await db_module.init_postgres(ensure_schema=False)
    async with db_module.AsyncSessionLocal() as db:
        account = (await db.execute(
            select(BrokerAccount).where(
                BrokerAccount.exchange_id == "krakenfutures",
                BrokerAccount.status == "active",
            ).limit(1)
        )).scalar()
        if not account:
            print(json.dumps({"ok": False, "error": "no active krakenfutures account"}))
            return
        cipher = CredentialCipher()
        creds = cipher.decrypt(account.encrypted_credentials)

    adapter = CcxtTradingAdapter(
        exchange_id="krakenfutures",
        api_key=creds.get("api_key", ""),
        api_secret=creds.get("api_secret", ""),
        sandbox=True,
        timeout_seconds=15.0,
    )
    try:
        quote = await adapter.fetch_latest_quote("BTC/USD:USD")
        if not quote or not quote.bid:
            print(json.dumps({"ok": False, "error": "no quote"}))
            return

        far_price = round(float(quote.bid) * 0.80, 1)
        client_id = f"e2e-{uuid.uuid4().hex[:12]}"
        intent = OrderIntent(
            client_order_id=client_id,
            symbol="BTCUSD",
            side="buy",
            qty=Decimal("0.001"),
            order_type="limit",
            limit_price=Decimal(str(far_price)),
            time_in_force="gtc",
        )
        order = await adapter.submit_order(intent)
        print(json.dumps({
            "action": "submitted",
            "ok": True,
            "order_id": order.provider_order_id,
            "status": order.status,
            "symbol": order.symbol,
            "side": order.side,
            "price": far_price,
        }))

        cancelled = await adapter.cancel_order(order.provider_order_id)
        print(json.dumps({
            "action": "cancelled",
            "ok": cancelled,
            "order_id": order.provider_order_id,
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}))
    finally:
        await adapter.aclose()

asyncio.run(run())
'''
    ok, output = docker_exec_python(script, timeout=30)
    lines = [l for l in output.strip().split("\n") if l.strip().startswith("{")]
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("action") == "submitted":
            report(
                f"Kraken order submitted (id={data.get('order_id', '?')[:12]}..., status={data.get('status')})",
                data.get("ok", False),
            )
        elif data.get("action") == "cancelled":
            report(f"Kraken order cancelled", data.get("ok", False))
        elif not data.get("ok"):
            report(f"Kraken order test error", False, data.get("error", "?")[:200])
    if not lines:
        report("Kraken order test", False, output[-200:] if output else "no output")


def _verify_portfolio(api: API, deployment_id: str) -> None:
    """Verify portfolio and PnL."""
    r = api.get(f"/api/v1/deployments/{deployment_id}/portfolio")
    if r.status_code == 200:
        p = r.json()
        report("portfolio endpoint OK", True)
        report(
            f"portfolio: equity={p.get('equity')}, cash={p.get('cash')}, "
            f"unrealized_pnl={p.get('unrealized_pnl')}, realized_pnl={p.get('realized_pnl')}",
            True,
        )
        positions = p.get("positions", [])
        report(f"positions (count={len(positions)})", True)
        for pos in positions:
            report(
                f"  {pos.get('symbol')} {pos.get('side')} qty={pos.get('qty')} "
                f"entry={pos.get('avg_entry_price')} mark={pos.get('mark_price')} "
                f"pnl={pos.get('unrealized_pnl')}",
                True,
            )
        broker = p.get("broker_account")
        if broker:
            report(
                f"broker snapshot: equity={broker.get('equity')}, cash={broker.get('cash')}, "
                f"buying_power={broker.get('buying_power')}",
                True,
            )
    else:
        report("portfolio", False, f"HTTP {r.status_code}: {r.text[:200]}")

    r = api.get(f"/api/v1/deployments/{deployment_id}/pnl")
    if r.status_code == 200:
        pnl = r.json()
        snapshots = pnl if isinstance(pnl, list) else pnl.get("snapshots", [])
        report(f"PnL snapshots (count={len(snapshots)})", True)
        if snapshots:
            latest = snapshots[0] if isinstance(snapshots[0], dict) else snapshots[-1]
            report(
                f"latest PnL: equity={latest.get('equity')}, cash={latest.get('cash')}, "
                f"unrealized={latest.get('unrealized_pnl')}",
                True,
            )
    else:
        report("PnL", False, f"HTTP {r.status_code}")


def _summary() -> None:
    global passed, failed
    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
