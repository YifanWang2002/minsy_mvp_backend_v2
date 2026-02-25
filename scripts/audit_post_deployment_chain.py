#!/usr/bin/env python3
"""Audit deployment->paper-trading->notifications chain on existing deployments.

This script intentionally avoids project test suites and drives real APIs.
It focuses on post-deployment runtime evidence:
1) order_qty patch (risk_limits.order_qty)
2) deployment restart -> open order/fill/position evidence
3) manual close -> close notification evidence
4) backtest completion -> backtest notification evidence
5) Telegram delivery attempts/success
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import requests
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.engine import DataLoader
from src.engine.backtest.service import (
    BacktestJobNotFoundError,
    create_backtest_job,
    get_backtest_job_view,
    schedule_backtest_job,
)
from src.engine.execution.credentials import CredentialCipher
from src.models import database as db_module


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class DeploymentAuditResult:
    deployment_id: str
    strategy_id: str | None
    user_id: str | None
    market: str | None
    symbol: str | None
    timeframe: str | None
    order_qty_before: float | None
    order_qty_after: float | None
    deployment_status_after_start: str | None
    runtime_evidence: dict[str, Any]
    order_evidence: dict[str, Any]
    close_evidence: dict[str, Any]
    backtest: dict[str, Any]
    notification_evidence: dict[str, Any]
    telegram_delivery: dict[str, Any]
    findings: list[str] = field(default_factory=list)
    ok: bool = False


class ApiClient:
    def __init__(self, *, base_url: str, email: str, password: str, language: str = "zh") -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.language = language
        self.session = requests.Session()
        self.access_token: str | None = None

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: int = 30,
        retries: int = 3,
    ) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                resp = self.session.request(
                    method=method.upper(),
                    url=self._url(path),
                    headers=self._headers(),
                    json=json_body,
                    timeout=timeout,
                )
                if resp.status_code >= 500 and attempt < retries - 1:
                    time.sleep(1.5)
                    continue
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt >= retries - 1:
                    raise
                time.sleep(1.5)
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed without response")

    def login(self) -> dict[str, Any]:
        resp = self.session.post(
            self._url("/auth/login"),
            json={"email": self.email, "password": self.password},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self.access_token = str(payload["access_token"])
        return payload

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/deployments/{deployment_id}", retries=4).json()

    def get_orders(self, deployment_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/deployments/{deployment_id}/orders", retries=4).json()
        return data if isinstance(data, list) else []

    def get_positions(self, deployment_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/deployments/{deployment_id}/positions", retries=4).json()
        return data if isinstance(data, list) else []

    def get_pnl(self, deployment_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/deployments/{deployment_id}/pnl", retries=4).json()
        return data if isinstance(data, list) else []

    def get_signals(self, deployment_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/deployments/{deployment_id}/signals", retries=4).json()
        return data if isinstance(data, list) else []

    def stop_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("POST", f"/deployments/{deployment_id}/stop", json_body={}, retries=4).json()

    def start_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("POST", f"/deployments/{deployment_id}/start", json_body={}, retries=4).json()

    def close_position(self, deployment_id: str, *, symbol: str) -> dict[str, Any]:
        payload = {"action": "close", "payload": {"symbol": symbol}}
        return self._request("POST", f"/deployments/{deployment_id}/manual-action", json_body=payload, retries=4).json()


async def _load_broker_credentials(broker_account_id: UUID) -> dict[str, str] | None:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        row = (
            await db.execute(
                text(
                    """
                    select encrypted_credentials
                    from broker_accounts
                    where id = :id
                    limit 1
                    """
                ),
                {"id": broker_account_id},
            )
        ).mappings().first()
    await db_module.close_postgres()
    if row is None:
        return None
    encrypted = row.get("encrypted_credentials")
    if not isinstance(encrypted, str) or not encrypted.strip():
        return None
    try:
        payload = CredentialCipher().decrypt(encrypted)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    return {str(k): str(v) for k, v in payload.items() if isinstance(v, str)}


async def _set_order_qty(*, deployment_id: UUID, order_qty: float) -> tuple[float | None, float | None]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        before_row = (
            await db.execute(
                text("select risk_limits from deployments where id = :id limit 1"),
                {"id": deployment_id},
            )
        ).mappings().first()
        before_limits = before_row.get("risk_limits") if before_row is not None else {}
        before_qty = None
        if isinstance(before_limits, dict):
            before_qty = _safe_float(before_limits.get("order_qty"))

        await db.execute(
            text(
                """
                update deployments
                set risk_limits = coalesce(risk_limits, '{}'::jsonb) || jsonb_build_object('order_qty', cast(:qty as numeric)),
                    updated_at = now()
                where id = :id
                """
            ),
            {"id": deployment_id, "qty": order_qty},
        )
        await db.commit()

        after_row = (
            await db.execute(
                text("select risk_limits from deployments where id = :id limit 1"),
                {"id": deployment_id},
            )
        ).mappings().first()
        after_limits = after_row.get("risk_limits") if after_row is not None else {}
        after_qty = None
        if isinstance(after_limits, dict):
            after_qty = _safe_float(after_limits.get("order_qty"))
    await db_module.close_postgres()
    return before_qty, after_qty


async def _trigger_backtest_completion(
    *,
    strategy_id: UUID,
    user_id: UUID,
    market: str,
    symbol: str,
    timeframe: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    loader = DataLoader()
    metadata = loader.get_symbol_metadata(market, symbol)
    available = metadata.get("available_timerange") if isinstance(metadata.get("available_timerange"), dict) else {}
    start_raw = str(available.get("start") or "").strip()
    end_raw = str(available.get("end") or "").strip()
    if not start_raw or not end_raw:
        return {
            "ok": False,
            "reason": "missing_data_coverage",
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
        }

    end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(UTC)
    start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(UTC)
    target_start = max(start_dt, end_dt - timedelta(days=2))

    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        receipt = await create_backtest_job(
            db,
            strategy_id=strategy_id,
            start_date=target_start.isoformat(),
            end_date=end_dt.isoformat(),
            user_id=user_id,
            auto_commit=True,
        )
        await schedule_backtest_job(receipt.job_id)

    deadline = time.time() + timeout_seconds
    last_view: dict[str, Any] | None = None
    while time.time() < deadline:
        await asyncio.sleep(2.0)
        await db_module.init_postgres(ensure_schema=False)
        assert db_module.AsyncSessionLocal is not None
        async with db_module.AsyncSessionLocal() as db:
            try:
                view = await get_backtest_job_view(db, job_id=receipt.job_id, user_id=user_id)
            except BacktestJobNotFoundError:
                continue
            last_view = {
                "job_id": str(view.job_id),
                "status": view.status,
                "progress": view.progress,
                "current_step": view.current_step,
                "error": view.error,
            }
            if view.status in {"done", "failed"}:
                await db_module.close_postgres()
                return {
                    "ok": view.status == "done",
                    "job_id": str(view.job_id),
                    "status": view.status,
                    "progress": view.progress,
                    "current_step": view.current_step,
                    "error": view.error,
                    "timeframe": timeframe,
                    "market": market,
                    "symbol": symbol,
                    "start_date": target_start.isoformat(),
                    "end_date": end_dt.isoformat(),
                }

    await db_module.close_postgres()
    return {
        "ok": False,
        "reason": "timeout",
        "job": last_view,
        "timeframe": timeframe,
        "market": market,
        "symbol": symbol,
        "start_date": target_start.isoformat(),
        "end_date": end_dt.isoformat(),
    }


async def _query_notification_state(*, user_id: UUID, since_at: datetime) -> dict[str, Any]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        outbox_rows = (
            await db.execute(
                text(
                    """
                    select id, channel, event_type, event_key, payload, status,
                           retry_count, max_retries, sent_at, next_retry_at, last_error, created_at
                    from notification_outbox
                    where user_id = :user_id
                      and created_at >= :since_at
                    order by created_at asc
                    """
                ),
                {"user_id": user_id, "since_at": since_at},
            )
        ).mappings().all()
        attempts = (
            await db.execute(
                text(
                    """
                    select a.outbox_id, a.provider, a.success, a.error_code, a.error_message, a.attempted_at
                    from notification_delivery_attempts a
                    join notification_outbox o on o.id = a.outbox_id
                    where o.user_id = :user_id
                      and o.created_at >= :since_at
                    order by a.attempted_at asc
                    """
                ),
                {"user_id": user_id, "since_at": since_at},
            )
        ).mappings().all()
        binding = (
            await db.execute(
                text(
                    """
                    select provider, status, external_chat_id, external_username
                    from social_connector_bindings
                    where user_id = :user_id
                      and provider = 'telegram'
                    limit 1
                    """
                ),
                {"user_id": user_id},
            )
        ).mappings().first()
    await db_module.close_postgres()

    events = [dict(row) for row in outbox_rows]
    attempt_rows = [dict(row) for row in attempts]
    by_event: dict[str, int] = {}
    for row in events:
        key = str(row.get("event_type", "")).upper()
        by_event[key] = by_event.get(key, 0) + 1

    return {
        "events": events,
        "attempts": attempt_rows,
        "event_type_counts": by_event,
        "telegram_binding": dict(binding) if binding is not None else None,
    }


def _resolve_deployment_scope(deployment: dict[str, Any]) -> tuple[str, str, str]:
    market = str(deployment.get("market") or "crypto")
    symbols = deployment.get("symbols") if isinstance(deployment.get("symbols"), list) else []
    symbol = str(symbols[0]) if symbols else "BTCUSD"
    timeframe = str(deployment.get("timeframe") or "1m")
    return market, symbol, timeframe


def _verify_provider_order(provider_order_id: str, creds: dict[str, str] | None) -> tuple[bool, dict[str, Any]]:
    if not provider_order_id or provider_order_id.startswith("paper-"):
        return False, {"provider_lookup_status": None, "provider_order_verified": False}
    if not creds:
        return False, {"provider_lookup_status": None, "provider_order_verified": False, "provider_lookup_error": "missing_creds"}

    api_key = creds.get("APCA-API-KEY-ID") or creds.get("api_key")
    api_secret = creds.get("APCA-API-SECRET-KEY") or creds.get("api_secret")
    base_url = creds.get("trading_base_url") or settings.alpaca_paper_trading_base_url
    if not api_key or not api_secret:
        return False, {"provider_lookup_status": None, "provider_order_verified": False, "provider_lookup_error": "missing_api_keys"}

    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/v2/orders/{provider_order_id}",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
            timeout=20,
        )
        evidence: dict[str, Any] = {"provider_lookup_status": resp.status_code, "provider_order_verified": False}
        if resp.status_code == 200:
            payload = resp.json()
            evidence["provider_lookup_payload"] = {
                "id": payload.get("id"),
                "status": payload.get("status"),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "filled_qty": payload.get("filled_qty"),
            }
            evidence["provider_order_verified"] = str(payload.get("id")) == provider_order_id
        return bool(evidence.get("provider_order_verified")), evidence
    except Exception as exc:  # noqa: BLE001
        return False, {"provider_lookup_status": None, "provider_order_verified": False, "provider_lookup_error": f"{type(exc).__name__}: {exc}"}


def _pick_new_order(*, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any] | None:
    before_ids = {str(item.get("order_id")) for item in before if isinstance(item, dict) and item.get("order_id")}
    for item in after:
        if not isinstance(item, dict):
            continue
        order_id = item.get("order_id")
        if order_id and str(order_id) not in before_ids:
            return item
    return None


def audit_one_deployment(
    client: ApiClient,
    *,
    deployment_id: str,
    order_qty: float,
    runtime_wait_seconds: int,
    close_wait_seconds: int,
    backtest_timeout_seconds: int,
) -> DeploymentAuditResult:
    now_at = datetime.now(UTC)
    dep_before = client.get_deployment(deployment_id)
    strategy_id = str(dep_before.get("strategy_id")) if dep_before.get("strategy_id") else None
    user_id = str(dep_before.get("user_id")) if dep_before.get("user_id") else None
    market, symbol, timeframe = _resolve_deployment_scope(dep_before)

    before_qty, after_qty = asyncio.run(_set_order_qty(deployment_id=UUID(deployment_id), order_qty=order_qty))

    orders_before = client.get_orders(deployment_id)
    positions_before = client.get_positions(deployment_id)
    pnl_before = client.get_pnl(deployment_id)
    signals_before = client.get_signals(deployment_id)

    client.stop_deployment(deployment_id)
    start_payload = client.start_deployment(deployment_id)
    dep_after_start = (
        start_payload.get("deployment")
        if isinstance(start_payload, dict) and isinstance(start_payload.get("deployment"), dict)
        else client.get_deployment(deployment_id)
    )
    status_after_start = str(dep_after_start.get("status") or "")
    run_after_start = dep_after_start.get("run") if isinstance(dep_after_start.get("run"), dict) else {}

    observed_dep = dep_after_start
    observed_orders = orders_before
    observed_positions = positions_before
    observed_pnl = pnl_before
    observed_signals = signals_before

    deadline = time.time() + runtime_wait_seconds
    while time.time() < deadline:
        time.sleep(5)
        observed_dep = client.get_deployment(deployment_id)
        observed_orders = client.get_orders(deployment_id)
        observed_positions = client.get_positions(deployment_id)
        observed_pnl = client.get_pnl(deployment_id)
        observed_signals = client.get_signals(deployment_id)
        if len(observed_orders) > len(orders_before):
            break

    new_order = _pick_new_order(before=orders_before, after=observed_orders)
    run_payload = observed_dep.get("run") if isinstance(observed_dep.get("run"), dict) else {}
    broker_account_raw = run_payload.get("broker_account_id")
    creds = (
        asyncio.run(_load_broker_credentials(UUID(str(broker_account_raw))))
        if broker_account_raw
        else None
    )
    provider_order_id = str(new_order.get("provider_order_id")) if isinstance(new_order, dict) and new_order.get("provider_order_id") else ""
    provider_verified, provider_evidence = _verify_provider_order(provider_order_id, creds)

    close_evidence: dict[str, Any] = {
        "attempted": False,
        "close_action_status": None,
        "orders_before_close": len(observed_orders),
        "orders_after_close": len(observed_orders),
        "order_delta_after_close": 0,
    }
    if observed_positions:
        close_evidence["attempted"] = True
        symbol_for_close = str(observed_positions[0].get("symbol") or symbol)
        close_resp = client.close_position(deployment_id, symbol=symbol_for_close)
        close_evidence["close_action_status"] = close_resp.get("status")
        close_deadline = time.time() + close_wait_seconds
        close_orders = observed_orders
        close_positions = observed_positions
        while time.time() < close_deadline:
            time.sleep(3)
            close_orders = client.get_orders(deployment_id)
            close_positions = client.get_positions(deployment_id)
            if len(close_orders) > len(observed_orders):
                break
        close_evidence["orders_after_close"] = len(close_orders)
        close_evidence["order_delta_after_close"] = len(close_orders) - len(observed_orders)
        close_evidence["positions_after_close"] = len(close_positions)
        observed_orders = close_orders
        observed_positions = close_positions

    if strategy_id and user_id:
        backtest = asyncio.run(
            _trigger_backtest_completion(
                strategy_id=UUID(strategy_id),
                user_id=UUID(user_id),
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                timeout_seconds=backtest_timeout_seconds,
            )
        )
    else:
        backtest = {"ok": False, "reason": "missing_strategy_or_user"}

    time.sleep(12)
    notif_state = (
        asyncio.run(_query_notification_state(user_id=UUID(user_id), since_at=now_at))
        if user_id
        else {"events": [], "attempts": [], "event_type_counts": {}, "telegram_binding": None}
    )

    events = notif_state.get("events") if isinstance(notif_state.get("events"), list) else []
    attempts = notif_state.get("attempts") if isinstance(notif_state.get("attempts"), list) else []
    dep_events = [
        row
        for row in events
        if isinstance(row, dict)
        and isinstance(row.get("payload"), dict)
        and str(row["payload"].get("deployment_id", "")) == deployment_id
    ]
    event_types = {str(row.get("event_type", "")).upper() for row in dep_events}
    if isinstance(backtest.get("job_id"), str):
        for row in events:
            payload = row.get("payload") if isinstance(row, dict) else None
            if isinstance(payload, dict) and str(payload.get("job_id", "")) == str(backtest["job_id"]):
                event_types.add(str(row.get("event_type", "")).upper())

    successful_attempts = [row for row in attempts if isinstance(row, dict) and row.get("success") is True]
    failed_attempts = [row for row in attempts if isinstance(row, dict) and row.get("success") is False]

    runtime_state = run_payload.get("runtime_state") if isinstance(run_payload.get("runtime_state"), dict) else {}
    runtime_evidence = {
        "scheduler_last_enqueued_at": runtime_state.get("scheduler", {}).get("last_enqueued_at")
        if isinstance(runtime_state.get("scheduler"), dict)
        else None,
        "run_status": run_payload.get("status"),
        "signals_before": len(signals_before),
        "signals_after": len(observed_signals),
        "signal_delta": len(observed_signals) - len(signals_before),
        "runtime_last_signal": runtime_state.get("last_signal"),
        "runtime_last_signal_reason": runtime_state.get("last_signal_reason"),
    }

    order_evidence = {
        "orders_before": len(orders_before),
        "orders_after": len(observed_orders),
        "order_delta": len(observed_orders) - len(orders_before),
        "positions_before": len(positions_before),
        "positions_after": len(observed_positions),
        "position_delta": len(observed_positions) - len(positions_before),
        "pnl_snapshots_before": len(pnl_before),
        "pnl_snapshots_after": len(observed_pnl),
        "pnl_snapshot_delta": len(observed_pnl) - len(pnl_before),
        "new_order": new_order,
        "provider_order_id": provider_order_id or None,
        "provider_order_verified": provider_verified,
        **provider_evidence,
    }

    notification_evidence = {
        "event_type_counts": notif_state.get("event_type_counts"),
        "target_deployment_event_types": sorted(event_types),
        "target_deployment_event_count": len(dep_events),
        "outbox_rows_total": len(events),
    }
    telegram_delivery = {
        "binding": notif_state.get("telegram_binding"),
        "attempt_total": len(attempts),
        "attempt_success": len(successful_attempts),
        "attempt_failed": len(failed_attempts),
        "failed_error_codes": sorted(
            {
                str(row.get("error_code"))
                for row in failed_attempts
                if isinstance(row.get("error_code"), str)
            }
        ),
    }

    findings: list[str] = []
    if status_after_start != "active":
        findings.append(f"deployment status after start is not active: {status_after_start}")
    if order_evidence["order_delta"] <= 0:
        findings.append("no new orders after start window")
    if not provider_verified:
        findings.append("could not verify provider order on Alpaca")
    if close_evidence.get("attempted") is False:
        findings.append("close action skipped because no position found")
    elif close_evidence.get("order_delta_after_close", 0) <= 0:
        findings.append("manual close did not create additional order")
    if not backtest.get("ok"):
        findings.append("backtest did not complete successfully")

    required_events = {"DEPLOYMENT_STARTED", "POSITION_OPENED", "POSITION_CLOSED", "BACKTEST_COMPLETED"}
    missing_events = sorted(required_events - event_types)
    if missing_events:
        findings.append(f"missing notification events: {missing_events}")
    if telegram_delivery["attempt_success"] <= 0:
        findings.append("no successful Telegram delivery attempts")

    return DeploymentAuditResult(
        deployment_id=deployment_id,
        strategy_id=strategy_id,
        user_id=user_id,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        order_qty_before=before_qty,
        order_qty_after=after_qty,
        deployment_status_after_start=status_after_start,
        runtime_evidence=runtime_evidence,
        order_evidence=order_evidence,
        close_evidence=close_evidence,
        backtest=backtest,
        notification_evidence=notification_evidence,
        telegram_delivery=telegram_delivery,
        findings=findings,
        ok=not findings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit post-deployment live chain for existing deployments.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--email", default="2@test.com")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--deployment-ids",
        required=True,
        help="Comma-separated deployment IDs to audit.",
    )
    parser.add_argument("--order-qty", type=float, default=0.009)
    parser.add_argument("--runtime-wait-seconds", type=int, default=180)
    parser.add_argument("--close-wait-seconds", type=int, default=90)
    parser.add_argument("--backtest-timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--json-report",
        default="logs/audit_post_deployment_chain_report.json",
        help="Path to write JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = _now_iso()
    deployment_ids = [item.strip() for item in str(args.deployment_ids).split(",") if item.strip()]
    if not deployment_ids:
        raise SystemExit("No deployment IDs provided.")
    if args.order_qty <= 0:
        raise SystemExit("--order-qty must be > 0")

    client = ApiClient(
        base_url=args.base_url,
        email=args.email,
        password=args.password,
        language=args.language,
    )
    login_payload = client.login()

    results: list[DeploymentAuditResult] = []
    for deployment_id in deployment_ids:
        result = audit_one_deployment(
            client,
            deployment_id=deployment_id,
            order_qty=float(args.order_qty),
            runtime_wait_seconds=max(30, int(args.runtime_wait_seconds)),
            close_wait_seconds=max(30, int(args.close_wait_seconds)),
            backtest_timeout_seconds=max(60, int(args.backtest_timeout_seconds)),
        )
        results.append(result)

    report = {
        "started_at": started_at,
        "finished_at": _now_iso(),
        "runtime": {
            "base_url": args.base_url,
            "openai_model": settings.openai_response_model,
            "paper_trading_enabled": settings.paper_trading_enabled,
            "paper_trading_execute_orders": settings.paper_trading_execute_orders,
            "notifications_enabled": settings.notifications_enabled,
            "telegram_enabled": settings.telegram_enabled,
        },
        "user": {
            "user_id": login_payload.get("user_id"),
            "email": args.email,
            "kyc_status": login_payload.get("user", {}).get("kyc_status"),
        },
        "order_qty_target": float(args.order_qty),
        "deployments": [asdict(result) for result in results],
        "summary": {
            "deployment_count": len(results),
            "ok_count": sum(1 for item in results if item.ok),
            "all_ok": all(item.ok for item in results),
        },
    }

    out_path = Path(args.json_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item.ok for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
