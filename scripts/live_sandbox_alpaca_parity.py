"""Run a live parity check between Alpaca broker execution and internal sandbox.

This script:
1) Logs in (or registers) a user via API
2) Ensures one usable Alpaca broker account and one sandbox broker account
3) Creates one strategy and two deployments (same strategy, different broker)
4) Runs process-now on minute boundaries for N cycles
5) Collects signals/orders/positions/pnl and writes a JSON report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from packages.shared_settings.schema.settings import settings


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _now_tag() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _json_or_none(response: httpx.Response) -> Any:
    text = response.text.strip()
    if not text:
        return None
    return response.json()


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    headers = dict(kwargs.pop("headers", {}) or {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = client.request(method, path, headers=headers, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"{method.upper()} {path} failed [{response.status_code}] {response.text[:600]}",
        )
    return _json_or_none(response)


def _login_or_register(
    client: httpx.Client,
    *,
    email: str,
    password: str,
    name: str,
) -> str:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    if login.status_code == 200:
        return str(login.json()["access_token"])

    register = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": name},
    )
    if register.status_code == 201:
        return str(register.json()["access_token"])

    retry = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    if retry.status_code != 200:
        raise RuntimeError(
            f"Auth failed: login={login.status_code}, register={register.status_code}, retry={retry.status_code}",
        )
    return str(retry.json()["access_token"])


def _ensure_auto_execute(client: httpx.Client, token: str) -> dict[str, Any]:
    return _request(
        client,
        "PUT",
        "/api/v1/trading/preferences",
        token=token,
        json={"execution_mode": "auto_execute"},
    )


def _list_deployments(client: httpx.Client, token: str) -> list[dict[str, Any]]:
    payload = _request(client, "GET", "/api/v1/deployments", token=token)
    return payload if isinstance(payload, list) else []


def _stop_noisy_deployments(client: httpx.Client, token: str) -> list[dict[str, Any]]:
    rows = _list_deployments(client, token)
    stopped: list[dict[str, Any]] = []
    for row in rows:
        deployment_id = row.get("deployment_id")
        status = str(row.get("status", "")).strip().lower()
        if not deployment_id or status in {"stopped", "error"}:
            continue
        response = client.post(
            f"/api/v1/deployments/{deployment_id}/manual-actions",
            headers={"Authorization": f"Bearer {token}"},
            json={"action": "stop", "payload": {"reason": "parity_pre_cleanup"}},
        )
        if response.status_code not in {200, 409}:
            raise RuntimeError(
                f"failed to stop deployment {deployment_id}: {response.status_code} {response.text[:300]}",
            )
        stopped.append(
            {
                "deployment_id": str(deployment_id),
                "previous_status": status,
                "stop_status_code": response.status_code,
            }
        )
    return stopped


def _list_broker_accounts(client: httpx.Client, token: str) -> list[dict[str, Any]]:
    payload = _request(client, "GET", "/api/v1/broker-accounts", token=token)
    return payload if isinstance(payload, list) else []


def _validate_account(client: httpx.Client, token: str, broker_account_id: str) -> bool:
    response = client.post(
        f"/api/v1/broker-accounts/{broker_account_id}/validate",
        headers={"Authorization": f"Bearer {token}"},
    )
    return response.status_code == 200


def _ensure_alpaca_account(client: httpx.Client, token: str, tag: str) -> tuple[str, str]:
    rows = _list_broker_accounts(client, token)
    active_candidates = [
        row
        for row in rows
        if str(row.get("provider", "")).strip().lower() == "alpaca"
        and str(row.get("status", "")).strip().lower() == "active"
    ]
    for row in active_candidates:
        broker_account_id = str(row["broker_account_id"])
        if _validate_account(client, token, broker_account_id):
            return broker_account_id, "reused_active_validated"

    create_payload = {
        "provider": "alpaca",
        "mode": "paper",
        "credentials": {
            "api_key": settings.alpaca_api_key,
            "api_secret": settings.alpaca_api_secret,
        },
        "metadata": {
            "source": "sandbox-alpaca-parity",
            "tag": tag,
        },
    }
    response = client.post(
        "/api/v1/broker-accounts",
        headers={"Authorization": f"Bearer {token}"},
        json=create_payload,
    )
    if response.status_code == 201:
        return str(response.json()["broker_account_id"]), "created_validated"

    if response.status_code == 409:
        rows = _list_broker_accounts(client, token)
        fallback_candidates = [
            row
            for row in rows
            if str(row.get("provider", "")).strip().lower() == "alpaca"
            and str(row.get("status", "")).strip().lower() == "active"
        ]
        for row in fallback_candidates:
            broker_account_id = str(row["broker_account_id"])
            if _validate_account(client, token, broker_account_id):
                return broker_account_id, "reused_after_conflict"

    raise RuntimeError(
        f"Unable to resolve Alpaca broker account: {response.status_code} {response.text[:400]}",
    )


def _ensure_sandbox_account(client: httpx.Client, token: str, tag: str) -> tuple[str, str]:
    rows = _list_broker_accounts(client, token)
    active_candidates = [
        row
        for row in rows
        if str(row.get("provider", "")).strip().lower() == "sandbox"
        and str(row.get("status", "")).strip().lower() == "active"
    ]
    if active_candidates:
        return str(active_candidates[0]["broker_account_id"]), "reused_active"

    response = client.post(
        "/api/v1/broker-accounts",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "provider": "sandbox",
            "mode": "paper",
            "credentials": {},
            "metadata": {
                "source": "sandbox-alpaca-parity",
                "tag": tag,
                "starting_cash": "100000",
                "slippage_bps": "0",
            },
        },
    )
    if response.status_code != 201:
        raise RuntimeError(
            f"Unable to create sandbox broker account: {response.status_code} {response.text[:400]}",
        )
    return str(response.json()["broker_account_id"]), "created"


def _build_toggle_strategy(symbol: str, tag: str) -> dict[str, Any]:
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": f"Parity-Toggle-{tag}",
            "description": "Open long on every bar, close on next bar via short-entry condition.",
        },
        "universe": {"market": "crypto", "tickers": [symbol]},
        "timeframe": "1m",
        "factors": {
            "ema_2": {
                "type": "ema",
                "params": {"period": 2, "source": "close"},
            }
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "pct_equity", "pct": 0.01},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "never_hit_long_exit",
                        "order": {"type": "market"},
                        "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
                    }
                ],
            },
            "short": {
                "position_sizing": {"mode": "pct_equity", "pct": 0.01},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": 0}},
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "never_hit_short_exit",
                        "order": {"type": "market"},
                        "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
                    }
                ],
            },
        },
    }


def _create_strategy(client: httpx.Client, token: str, symbol: str, tag: str) -> str:
    thread = _request(
        client,
        "POST",
        "/api/v1/chat/new-thread",
        token=token,
        json={"metadata": {"source": "sandbox-alpaca-parity", "tag": tag}},
    )
    session_id = str(thread["session_id"])
    strategy = _request(
        client,
        "POST",
        "/api/v1/strategies/confirm",
        token=token,
        json={
            "session_id": session_id,
            "dsl_json": _build_toggle_strategy(symbol, tag),
            "auto_start_backtest": False,
            "language": "en",
        },
    )
    return str(strategy["strategy_id"])


def _create_deployment(
    client: httpx.Client,
    token: str,
    *,
    strategy_id: str,
    broker_account_id: str,
    order_qty: float,
    tag: str,
) -> str:
    payload = _request(
        client,
        "POST",
        "/api/v1/deployments",
        token=token,
        json={
            "strategy_id": strategy_id,
            "broker_account_id": broker_account_id,
            "mode": "paper",
            "capital_allocated": 1000,
            "risk_limits": {"order_qty": order_qty},
            "runtime_state": {"source": "sandbox-alpaca-parity", "tag": tag},
        },
    )
    return str(payload["deployment_id"])


def _start_deployment(client: httpx.Client, token: str, deployment_id: str) -> dict[str, Any]:
    return _request(
        client,
        "POST",
        f"/api/v1/deployments/{deployment_id}/start",
        token=token,
    )


def _stop_deployment(client: httpx.Client, token: str, deployment_id: str, reason: str) -> None:
    response = client.post(
        f"/api/v1/deployments/{deployment_id}/manual-actions",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": "stop", "payload": {"reason": reason}},
    )
    if response.status_code not in {200, 409}:
        raise RuntimeError(
            f"failed to stop deployment {deployment_id}: {response.status_code} {response.text[:300]}",
        )


def _sleep_until_next_minute(*, cushion_seconds: float = 2.0) -> float:
    now = time.time()
    next_minute_epoch = (math.floor(now / 60) + 1) * 60
    target = next_minute_epoch + max(0.0, cushion_seconds)
    delay = max(0.0, target - now)
    time.sleep(delay)
    return delay


def _process_now(
    client: httpx.Client,
    token: str,
    deployment_id: str,
    *,
    max_retries: int = 40,
    retry_sleep_seconds: float = 0.25,
) -> dict[str, Any]:
    attempts = 0
    payload: dict[str, Any] | None = None
    for attempts in range(1, max_retries + 1):
        response = _request(
            client,
            "POST",
            f"/api/v1/deployments/{deployment_id}/process-now",
            token=token,
        )
        assert isinstance(response, dict)
        payload = response
        reason = str(response.get("reason", "")).strip().lower()
        if reason != "deployment_locked":
            break
        if attempts < max_retries:
            time.sleep(max(0.0, retry_sleep_seconds))

    assert payload is not None
    payload["lock_retry_attempts"] = attempts
    return payload


def _fetch_runtime_views(client: httpx.Client, token: str, deployment_id: str) -> dict[str, Any]:
    detail = _request(client, "GET", f"/api/v1/deployments/{deployment_id}", token=token)
    orders = _request(client, "GET", f"/api/v1/deployments/{deployment_id}/orders", token=token)
    positions = _request(client, "GET", f"/api/v1/deployments/{deployment_id}/positions", token=token)
    pnl = _request(client, "GET", f"/api/v1/deployments/{deployment_id}/pnl", token=token)
    signals = _request(
        client,
        "GET",
        f"/api/v1/deployments/{deployment_id}/signals",
        token=token,
        params={"limit": 500},
    )
    return {
        "deployment": detail if isinstance(detail, dict) else {},
        "orders": orders if isinstance(orders, list) else [],
        "positions": positions if isinstance(positions, list) else [],
        "pnl": pnl if isinstance(pnl, list) else [],
        "signals": signals if isinstance(signals, list) else [],
    }


def _sort_by_time(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    def _safe(value: dict[str, Any]) -> str:
        raw = value.get(key)
        return str(raw or "")

    return sorted(rows, key=_safe)


def _compact_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = []
    for row in _sort_by_time(rows, "bar_time"):
        signal = str(row.get("signal", "")).strip().upper()
        if not signal:
            continue
        signals.append(
            {
                "signal_event_id": row.get("signal_event_id"),
                "signal": signal,
                "bar_time": row.get("bar_time"),
                "reason": row.get("reason"),
            }
        )
    return signals


def _compact_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in _sort_by_time(rows, "submitted_at"):
        compact.append(
            {
                "order_id": row.get("order_id"),
                "provider_order_id": row.get("provider_order_id"),
                "side": row.get("side"),
                "qty": row.get("qty"),
                "price": row.get("price"),
                "status": row.get("status"),
                "provider_status": row.get("provider_status"),
                "submitted_at": row.get("submitted_at"),
                "reject_reason": row.get("reject_reason"),
            }
        )
    return compact


def _latest_pnl(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    ordered = _sort_by_time(rows, "snapshot_time")
    return ordered[-1]


def _compare_cycles(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    matched_signal = 0
    matched_bar_time = 0
    pairs: list[dict[str, Any]] = []
    for row in cycles:
        alpaca = row["alpaca"]
        sandbox = row["sandbox"]
        signal_match = str(alpaca.get("signal")) == str(sandbox.get("signal"))
        bar_match = str(alpaca.get("bar_time")) == str(sandbox.get("bar_time"))
        if signal_match:
            matched_signal += 1
        if bar_match:
            matched_bar_time += 1
        pairs.append(
            {
                "cycle_index": row["cycle_index"],
                "alpaca_signal": alpaca.get("signal"),
                "sandbox_signal": sandbox.get("signal"),
                "alpaca_bar_time": alpaca.get("bar_time"),
                "sandbox_bar_time": sandbox.get("bar_time"),
                "signal_match": signal_match,
                "bar_time_match": bar_match,
            }
        )
    total = len(cycles)
    return {
        "cycles_total": total,
        "matched_signal_count": matched_signal,
        "matched_bar_time_count": matched_bar_time,
        "signal_match_ratio": (matched_signal / total) if total else 0.0,
        "bar_time_match_ratio": (matched_bar_time / total) if total else 0.0,
        "pairs": pairs,
    }


def _compare_pnl(alpaca_latest: dict[str, Any] | None, sandbox_latest: dict[str, Any] | None) -> dict[str, Any]:
    if alpaca_latest is None or sandbox_latest is None:
        return {
            "available": False,
            "alpaca_latest_snapshot_time": alpaca_latest.get("snapshot_time") if alpaca_latest else None,
            "sandbox_latest_snapshot_time": sandbox_latest.get("snapshot_time") if sandbox_latest else None,
        }

    def _as_float(row: dict[str, Any], key: str) -> float:
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    alpaca_equity = _as_float(alpaca_latest, "equity")
    sandbox_equity = _as_float(sandbox_latest, "equity")
    alpaca_unrealized = _as_float(alpaca_latest, "unrealized_pnl")
    sandbox_unrealized = _as_float(sandbox_latest, "unrealized_pnl")
    alpaca_realized = _as_float(alpaca_latest, "realized_pnl")
    sandbox_realized = _as_float(sandbox_latest, "realized_pnl")

    return {
        "available": True,
        "alpaca_latest_snapshot_time": alpaca_latest.get("snapshot_time"),
        "sandbox_latest_snapshot_time": sandbox_latest.get("snapshot_time"),
        "alpaca_equity": alpaca_equity,
        "sandbox_equity": sandbox_equity,
        "equity_delta_abs": abs(alpaca_equity - sandbox_equity),
        "alpaca_unrealized_pnl": alpaca_unrealized,
        "sandbox_unrealized_pnl": sandbox_unrealized,
        "unrealized_delta_abs": abs(alpaca_unrealized - sandbox_unrealized),
        "alpaca_realized_pnl": alpaca_realized,
        "sandbox_realized_pnl": sandbox_realized,
        "realized_delta_abs": abs(alpaca_realized - sandbox_realized),
    }


@dataclass(frozen=True)
class RunConfig:
    api_base_url: str
    email: str
    password: str
    name: str
    symbol: str
    cycles: int
    order_qty: float
    minute_cushion_seconds: float
    output_path: Path


def run_parity(config: RunConfig) -> dict[str, Any]:
    client = httpx.Client(base_url=config.api_base_url, timeout=60.0)
    started_at = _utc_now()
    tag = _now_tag()
    token: str | None = None
    alpaca_deployment_id: str | None = None
    sandbox_deployment_id: str | None = None

    try:
        token = _login_or_register(
            client,
            email=config.email,
            password=config.password,
            name=config.name,
        )
        me = _request(client, "GET", "/api/v1/auth/me", token=token)
        stopped_noise = _stop_noisy_deployments(client, token)
        preferences = _ensure_auto_execute(client, token)
        alpaca_account_id, alpaca_account_source = _ensure_alpaca_account(client, token, tag)
        sandbox_account_id, sandbox_account_source = _ensure_sandbox_account(client, token, tag)
        strategy_id = _create_strategy(client, token, config.symbol, tag)

        alpaca_deployment_id = _create_deployment(
            client,
            token,
            strategy_id=strategy_id,
            broker_account_id=alpaca_account_id,
            order_qty=config.order_qty,
            tag=tag,
        )
        sandbox_deployment_id = _create_deployment(
            client,
            token,
            strategy_id=strategy_id,
            broker_account_id=sandbox_account_id,
            order_qty=config.order_qty,
            tag=tag,
        )
        _start_deployment(client, token, alpaca_deployment_id)
        _start_deployment(client, token, sandbox_deployment_id)

        cycle_rows: list[dict[str, Any]] = []
        for index in range(1, config.cycles + 1):
            slept_seconds = _sleep_until_next_minute(cushion_seconds=config.minute_cushion_seconds)
            alpaca_result = _process_now(client, token, alpaca_deployment_id)
            sandbox_result = _process_now(client, token, sandbox_deployment_id)
            cycle_rows.append(
                {
                    "cycle_index": index,
                    "requested_at_utc": _utc_now().isoformat(),
                    "slept_seconds": slept_seconds,
                    "alpaca": alpaca_result,
                    "sandbox": sandbox_result,
                }
            )

        alpaca_views = _fetch_runtime_views(client, token, alpaca_deployment_id)
        sandbox_views = _fetch_runtime_views(client, token, sandbox_deployment_id)

        alpaca_signals = _compact_signals(alpaca_views["signals"])
        sandbox_signals = _compact_signals(sandbox_views["signals"])
        alpaca_orders = _compact_orders(alpaca_views["orders"])
        sandbox_orders = _compact_orders(sandbox_views["orders"])
        alpaca_latest_pnl = _latest_pnl(alpaca_views["pnl"])
        sandbox_latest_pnl = _latest_pnl(sandbox_views["pnl"])

        ended_at = _utc_now()
        report = {
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "config": {
                "api_base_url": config.api_base_url,
                "email": config.email,
                "symbol": config.symbol,
                "cycles": config.cycles,
                "order_qty": config.order_qty,
                "minute_cushion_seconds": config.minute_cushion_seconds,
            },
            "user": me,
            "pre_cleanup": {"stopped_deployments": stopped_noise},
            "trading_preferences": preferences,
            "broker_accounts": {
                "alpaca": {"broker_account_id": alpaca_account_id, "source": alpaca_account_source},
                "sandbox": {"broker_account_id": sandbox_account_id, "source": sandbox_account_source},
            },
            "strategy_id": strategy_id,
            "deployments": {
                "alpaca": alpaca_deployment_id,
                "sandbox": sandbox_deployment_id,
            },
            "cycle_results": cycle_rows,
            "signals": {
                "alpaca": alpaca_signals,
                "sandbox": sandbox_signals,
            },
            "orders": {
                "alpaca": alpaca_orders,
                "sandbox": sandbox_orders,
            },
            "positions": {
                "alpaca": _sort_by_time(alpaca_views["positions"], "updated_at"),
                "sandbox": _sort_by_time(sandbox_views["positions"], "updated_at"),
            },
            "pnl": {
                "alpaca": _sort_by_time(alpaca_views["pnl"], "snapshot_time"),
                "sandbox": _sort_by_time(sandbox_views["pnl"], "snapshot_time"),
            },
            "comparison": {
                "cycles": _compare_cycles(cycle_rows),
                "order_count_delta": abs(len(alpaca_orders) - len(sandbox_orders)),
                "signal_count_delta": abs(len(alpaca_signals) - len(sandbox_signals)),
                "pnl_latest": _compare_pnl(alpaca_latest_pnl, sandbox_latest_pnl),
            },
        }
        return report
    finally:
        try:
            if alpaca_deployment_id is not None and token is not None:
                _stop_deployment(client, token, alpaca_deployment_id, "parity_script_cleanup")
        except Exception:  # noqa: BLE001
            pass
        try:
            if sandbox_deployment_id is not None and token is not None:
                _stop_deployment(client, token, sandbox_deployment_id, "parity_script_cleanup")
        except Exception:  # noqa: BLE001
            pass
        client.close()


def _parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run sandbox vs alpaca live parity check")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--email", default="2@test.com")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--name", default="Parity Runner")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--order-qty", type=float, default=0.0002)
    parser.add_argument("--minute-cushion-seconds", type=float, default=2.0)
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    output = Path(args.output_path).expanduser().resolve() if args.output_path else None
    if output is None:
        reports_dir = Path("runtime/reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        output = reports_dir / f"sandbox_alpaca_parity_{_now_tag()}.json"

    return RunConfig(
        api_base_url=str(args.api_base_url).rstrip("/"),
        email=str(args.email).strip(),
        password=str(args.password),
        name=str(args.name).strip() or "Parity Runner",
        symbol=str(args.symbol).strip().upper(),
        cycles=max(1, int(args.cycles)),
        order_qty=max(float(args.order_qty), 0.00000001),
        minute_cushion_seconds=max(0.0, float(args.minute_cushion_seconds)),
        output_path=output,
    )


def main() -> None:
    config = _parse_args()
    report = run_parity(config)
    config.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_path={config.output_path}")
    print(f"cycle_signal_match_ratio={report['comparison']['cycles']['signal_match_ratio']}")
    print(f"order_count_delta={report['comparison']['order_count_delta']}")
    pnl_compare = report["comparison"]["pnl_latest"]
    print(f"pnl_available={pnl_compare.get('available')}")
    if pnl_compare.get("available"):
        print(f"equity_delta_abs={pnl_compare.get('equity_delta_abs')}")
        print(f"realized_delta_abs={pnl_compare.get('realized_delta_abs')}")


if __name__ == "__main__":
    main()
