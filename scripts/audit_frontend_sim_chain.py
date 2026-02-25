#!/usr/bin/env python3
"""End-to-end backend audit via frontend-like HTTP flow.

Scope:
1) strategy -> deployment automatic chain through chat/orchestrator/MCP.
2) paper-trading runtime chain (Celery-driven execution, market data, orders, portfolio).
3) notification chain rooted in business events (deployment/order/backtest events).

This script intentionally does not reuse the repository test suite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import requests
from requests import Response
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from src.models import database as db_module
from src.models.backtest import BacktestJob
from src.models.broker_account import BrokerAccount
from src.models.deployment import Deployment
from src.models.deployment_run import DeploymentRun
from src.models.fill import Fill
from src.models.notification_delivery_attempt import NotificationDeliveryAttempt
from src.models.notification_outbox import NotificationOutbox
from src.models.order import Order
from src.models.position import Position
from src.models.pnl_snapshot import PnlSnapshot
from src.models.session import Session
from src.models.signal_event import SignalEvent
from src.models.social_connector import SocialConnectorBinding
from src.models.strategy import Strategy
from src.models.trade_approval_request import TradeApprovalRequest
from src.models.user import User


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)


@dataclass
class TurnResult:
    session_id: str
    done_payload: dict[str, Any] | None
    stream_text: str
    stream_timed_out: bool
    assistant_message: dict[str, Any]
    session_snapshot: dict[str, Any]


class FrontendLikeClient:
    def __init__(self, *, base_url: str, email: str, password: str, language: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.language = language
        self.http = requests.Session()
        self.token: str | None = None

    @property
    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("Client is not authenticated yet.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _raise_on_unexpected(self, response: Response, expected_status: int, step: str) -> None:
        if response.status_code == expected_status:
            return
        body = response.text
        raise RuntimeError(
            f"{step} failed: status={response.status_code} expected={expected_status} body={body[:1000]}"
        )

    def login(self) -> dict[str, Any]:
        response = self.http.post(
            f"{self.base_url}/auth/login",
            json={"email": self.email, "password": self.password},
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "auth_login")
        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("auth_login returned empty access_token.")
        self.token = token.strip()
        return payload

    def create_thread(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.http.post(
            f"{self.base_url}/chat/new-thread",
            json={"metadata": metadata or {}},
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 201, "chat_new_thread")
        payload = response.json()
        if not isinstance(payload.get("session_id"), str):
            raise RuntimeError(f"chat_new_thread missing session_id: {payload}")
        return payload

    def get_session(self, session_id: str) -> dict[str, Any]:
        response = self.http.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "sessions_get")
        return response.json()

    def list_deployments(self) -> list[dict[str, Any]]:
        response = self.http.get(
            f"{self.base_url}/deployments",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "deployments_list")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"deployments_list payload is not list: {payload}")
        return payload

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        response = self.http.get(
            f"{self.base_url}/deployments/{deployment_id}",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "deployments_get")
        return response.json()

    def get_orders(self, deployment_id: str) -> list[dict[str, Any]]:
        response = self.http.get(
            f"{self.base_url}/deployments/{deployment_id}/orders",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "deployments_orders")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"deployments_orders payload is not list: {payload}")
        return payload

    def get_positions(self, deployment_id: str) -> list[dict[str, Any]]:
        response = self.http.get(
            f"{self.base_url}/deployments/{deployment_id}/positions",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "deployments_positions")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"deployments_positions payload is not list: {payload}")
        return payload

    def get_portfolio(self, deployment_id: str) -> dict[str, Any]:
        response = self.http.get(
            f"{self.base_url}/deployments/{deployment_id}/portfolio",
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "deployments_portfolio")
        return response.json()

    def get_market_bars(
        self,
        *,
        symbol: str,
        market: str = "stocks",
        timeframe: str = "1m",
        limit: int = 20,
    ) -> dict[str, Any]:
        response = self.http.get(
            f"{self.base_url}/market-data/bars",
            params={
                "symbol": symbol,
                "market": market,
                "timeframe": timeframe,
                "limit": limit,
                "refresh_if_empty": True,
            },
            headers=self._auth_headers,
            timeout=30,
        )
        self._raise_on_unexpected(response, 200, "market_data_bars")
        return response.json()

    def send_turn(
        self,
        *,
        session_id: str,
        message: str,
        max_stream_seconds: float = 240.0,
    ) -> TurnResult:
        before = self.get_session(session_id)
        before_count = len(before.get("messages", []))

        stream_url = f"{self.base_url}/chat/send-openai-stream"
        params = {"language": self.language}
        payload = {"session_id": session_id, "message": message}

        done_payload: dict[str, Any] | None = None
        text_chunks: list[str] = []
        current_event: str | None = None
        started_at = time.monotonic()
        stream_timed_out = False

        with self.http.post(
            stream_url,
            params=params,
            json=payload,
            headers=self._auth_headers,
            stream=True,
            timeout=(30, max_stream_seconds + 30),
        ) as response:
            self._raise_on_unexpected(response, 200, "chat_send_stream")

            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip("\r")
                if not line:
                    if time.monotonic() - started_at > max_stream_seconds:
                        stream_timed_out = True
                        break
                    continue
                if line.startswith(":"):
                    if time.monotonic() - started_at > max_stream_seconds:
                        stream_timed_out = True
                        break
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_raw = line.split(":", 1)[1].lstrip()
                try:
                    data_payload = json.loads(data_raw)
                except json.JSONDecodeError:
                    continue
                if (
                    current_event == "stream"
                    and isinstance(data_payload, dict)
                    and data_payload.get("type") == "text_delta"
                    and isinstance(data_payload.get("delta"), str)
                ):
                    text_chunks.append(str(data_payload["delta"]))
                if (
                    isinstance(data_payload, dict)
                    and data_payload.get("type") == "done"
                ):
                    done_payload = data_payload
                    break
                if time.monotonic() - started_at > max_stream_seconds:
                    stream_timed_out = True
                    break

        session_snapshot = self._wait_for_assistant_message(
            session_id=session_id,
            previous_count=before_count,
            timeout_seconds=120,
        )
        messages = session_snapshot.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            raise RuntimeError("No assistant message persisted after chat turn.")

        return TurnResult(
            session_id=session_id,
            done_payload=done_payload,
            stream_text="".join(text_chunks),
            stream_timed_out=stream_timed_out,
            assistant_message=messages[-1],
            session_snapshot=session_snapshot,
        )

    def _wait_for_assistant_message(
        self,
        *,
        session_id: str,
        previous_count: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest_snapshot: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            snapshot = self.get_session(session_id)
            latest_snapshot = snapshot
            messages = snapshot.get("messages", [])
            if (
                isinstance(messages, list)
                and len(messages) >= previous_count + 2
                and messages[-1].get("role") == "assistant"
            ):
                return snapshot
            time.sleep(2.0)

        if latest_snapshot is None:
            raise RuntimeError("Timed out waiting for assistant message; no session snapshot.")
        return latest_snapshot


def _extract_tool_names_from_message(message_payload: dict[str, Any]) -> list[str]:
    tool_calls = message_payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    names: list[str] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_call":
            continue
        raw_name = item.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            names.append(raw_name.strip())
    return names


def _coerce_json(value: Any) -> Any:
    if isinstance(value, dict | list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _find_first_uuid(value: Any) -> str | None:
    if isinstance(value, str):
        match = UUID_PATTERN.search(value)
        if match:
            return match.group(0)
        return None
    if isinstance(value, list):
        for item in value:
            found = _find_first_uuid(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        preferred_keys = (
            "deployment_id",
            "strategy_id",
            "order_id",
            "job_id",
        )
        for key in preferred_keys:
            if key in value:
                found = _find_first_uuid(value[key])
                if found:
                    return found
        for nested in value.values():
            found = _find_first_uuid(nested)
            if found:
                return found
    return None


def _extract_deployment_id_from_tool_calls(message_payload: dict[str, Any]) -> str | None:
    tool_calls = message_payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_call":
            continue
        output_payload = _coerce_json(item.get("output"))
        found = _find_first_uuid(output_payload)
        if found:
            return found
    return None


def _read_md_tool_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    output: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        match = re.search(r"`([^`]+)`", line)
        if not match:
            continue
        output.append(match.group(1).strip())
    return output


def collect_static_sync_checks() -> dict[str, Any]:
    from src.agents.orchestrator.constants import (
        _BACKTEST_FEEDBACK_TOOL_NAMES,
        _MARKET_DATA_MINIMAL_TOOL_NAMES,
        _STRATEGY_ARTIFACT_OPS_TOOL_NAMES,
        _TRADING_DEPLOYMENT_TOOL_NAMES,
    )
    from src.mcp.backtest.tools import TOOL_NAMES as BACKTEST_TOOL_NAMES
    from src.mcp.market_data.tools import TOOL_NAMES as MARKET_DATA_TOOL_NAMES
    from src.mcp.strategy.tools import TOOL_NAMES as STRATEGY_TOOL_NAMES
    from src.mcp.trading.tools import TOOL_NAMES as TRADING_TOOL_NAMES

    import inspect
    from src.mcp.trading.tools import trading_create_paper_deployment

    base_dir = Path(__file__).resolve().parents[1]
    deployment_skill_md = base_dir / "src" / "agents" / "skills" / "deployment" / "skills.md"
    strategy_skill_md = base_dir / "src" / "agents" / "skills" / "strategy" / "skills.md"

    deployment_skill_tools = sorted(set(_read_md_tool_list(deployment_skill_md)))
    strategy_skill_tools = sorted(set(_read_md_tool_list(strategy_skill_md)))

    trading_expected = sorted(set(_TRADING_DEPLOYMENT_TOOL_NAMES))
    trading_actual = sorted(set(TRADING_TOOL_NAMES))
    trading_missing = sorted(set(trading_expected) - set(trading_actual))

    strategy_expected = sorted(set(_STRATEGY_ARTIFACT_OPS_TOOL_NAMES))
    strategy_actual = sorted(set(STRATEGY_TOOL_NAMES))
    strategy_missing = sorted(set(strategy_expected) - set(strategy_actual))

    backtest_expected = sorted(set(_BACKTEST_FEEDBACK_TOOL_NAMES))
    backtest_actual = sorted(set(BACKTEST_TOOL_NAMES))
    backtest_missing = sorted(set(backtest_expected) - set(backtest_actual))

    market_expected = sorted(set(_MARKET_DATA_MINIMAL_TOOL_NAMES))
    market_actual = sorted(set(MARKET_DATA_TOOL_NAMES))
    market_missing = sorted(set(market_expected) - set(market_actual))

    signature = inspect.signature(trading_create_paper_deployment)
    create_params = list(signature.parameters.keys())

    return {
        "trading_runtime_policy_expected_tools": trading_expected,
        "trading_mcp_registered_tools": trading_actual,
        "trading_missing_in_mcp": trading_missing,
        "strategy_runtime_policy_expected_tools": strategy_expected,
        "strategy_mcp_registered_tools": strategy_actual,
        "strategy_missing_in_mcp": strategy_missing,
        "backtest_runtime_policy_expected_tools": backtest_expected,
        "backtest_mcp_registered_tools": backtest_actual,
        "backtest_missing_in_mcp": backtest_missing,
        "market_runtime_policy_expected_tools": market_expected,
        "market_mcp_registered_tools": market_actual,
        "market_missing_in_mcp": market_missing,
        "deployment_skill_tools_listed": deployment_skill_tools,
        "strategy_skill_tools_listed_subset": strategy_skill_tools,
        "trading_create_paper_deployment_signature": create_params,
        "deployment_skill_mentions_auto_start": "auto_start" in deployment_skill_md.read_text(
            encoding="utf-8"
        ),
    }


async def _fetch_user_and_binding_snapshot(*, user_email: str) -> dict[str, Any]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == user_email))
        if user is None:
            raise RuntimeError(f"user not found: {user_email}")

        broker_accounts = (
            await db.scalars(select(BrokerAccount).where(BrokerAccount.user_id == user.id))
        ).all()
        bindings = (
            await db.scalars(
                select(SocialConnectorBinding).where(
                    SocialConnectorBinding.user_id == user.id,
                    SocialConnectorBinding.provider == "telegram",
                )
            )
        ).all()
        return {
            "user_id": str(user.id),
            "broker_accounts": [
                {
                    "broker_account_id": str(item.id),
                    "provider": item.provider,
                    "mode": item.mode,
                    "status": item.status,
                    "last_validated_status": item.last_validated_status,
                }
                for item in broker_accounts
            ],
            "telegram_bindings": [
                {
                    "binding_id": str(item.id),
                    "status": item.status,
                    "external_chat_id": item.external_chat_id,
                    "external_user_id": item.external_user_id,
                }
                for item in bindings
            ],
        }


async def _fetch_runtime_snapshot(*, deployment_id: str) -> dict[str, Any]:
    deployment_uuid = UUID(deployment_id)
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        deployment = await db.scalar(
            select(Deployment)
            .options(
                selectinload(Deployment.strategy),
                selectinload(Deployment.deployment_runs),
            )
            .where(Deployment.id == deployment_uuid)
        )
        if deployment is None:
            raise RuntimeError(f"deployment not found in db: {deployment_id}")

        latest_run = None
        if deployment.deployment_runs:
            latest_run = sorted(
                deployment.deployment_runs,
                key=lambda row: row.created_at,
                reverse=True,
            )[0]

        order_count = await db.scalar(
            select(func.count(Order.id)).where(Order.deployment_id == deployment_uuid)
        )
        fill_count = await db.scalar(
            select(func.count(Fill.id))
            .join(Order, Order.id == Fill.order_id)
            .where(Order.deployment_id == deployment_uuid)
        )
        signal_count = await db.scalar(
            select(func.count(SignalEvent.id)).where(SignalEvent.deployment_id == deployment_uuid)
        )
        position_count = await db.scalar(
            select(func.count(Position.id)).where(Position.deployment_id == deployment_uuid)
        )
        pnl_snapshot_count = await db.scalar(
            select(func.count(PnlSnapshot.id)).where(PnlSnapshot.deployment_id == deployment_uuid)
        )
        approval_count = await db.scalar(
            select(func.count(TradeApprovalRequest.id)).where(
                TradeApprovalRequest.deployment_id == deployment_uuid
            )
        )
        latest_signal = await db.scalar(
            select(SignalEvent)
            .where(SignalEvent.deployment_id == deployment_uuid)
            .order_by(desc(SignalEvent.created_at), desc(SignalEvent.id))
            .limit(1)
        )

        return {
            "deployment_status": deployment.status,
            "deployment_mode": deployment.mode,
            "strategy_id": str(deployment.strategy_id),
            "strategy_status": deployment.strategy.status if deployment.strategy is not None else None,
            "run": {
                "deployment_run_id": str(latest_run.id) if latest_run is not None else None,
                "status": latest_run.status if latest_run is not None else None,
                "last_bar_time": (
                    latest_run.last_bar_time.isoformat()
                    if latest_run is not None and latest_run.last_bar_time is not None
                    else None
                ),
                "runtime_state": (
                    latest_run.runtime_state
                    if latest_run is not None and isinstance(latest_run.runtime_state, dict)
                    else {}
                ),
            },
            "counts": {
                "orders": int(order_count or 0),
                "fills": int(fill_count or 0),
                "signals": int(signal_count or 0),
                "positions": int(position_count or 0),
                "pnl_snapshots": int(pnl_snapshot_count or 0),
                "approvals": int(approval_count or 0),
            },
            "latest_signal": (
                {
                    "signal_event_id": str(latest_signal.id),
                    "signal": latest_signal.signal,
                    "reason": latest_signal.reason,
                    "symbol": latest_signal.symbol,
                    "timeframe": latest_signal.timeframe,
                    "bar_time": latest_signal.bar_time.isoformat(),
                    "metadata": latest_signal.metadata_ if isinstance(latest_signal.metadata_, dict) else {},
                }
                if latest_signal is not None
                else None
            ),
        }


async def _validate_provider_order_on_alpaca(
    *,
    user_email: str,
    deployment_id: str,
    provider_order_id: str,
) -> dict[str, Any]:
    from src.engine.execution.adapters.alpaca_trading import AlpacaTradingAdapter
    from src.engine.execution.credentials import CredentialCipher

    deployment_uuid = UUID(deployment_id)
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == user_email))
        if user is None:
            raise RuntimeError(f"user not found: {user_email}")

        deployment = await db.scalar(
            select(Deployment)
            .options(selectinload(Deployment.deployment_runs))
            .where(Deployment.id == deployment_uuid, Deployment.user_id == user.id)
        )
        if deployment is None:
            raise RuntimeError(f"deployment not found: {deployment_id}")

        if not deployment.deployment_runs:
            raise RuntimeError(f"deployment run missing: {deployment_id}")
        latest_run = sorted(
            deployment.deployment_runs,
            key=lambda row: row.created_at,
            reverse=True,
        )[0]

        account = await db.scalar(
            select(BrokerAccount).where(BrokerAccount.id == latest_run.broker_account_id)
        )
        if account is None:
            raise RuntimeError("broker account missing for deployment run.")

        decrypted = CredentialCipher().decrypt(account.encrypted_credentials)
        api_key = str(
            decrypted.get("APCA-API-KEY-ID")
            or decrypted.get("api_key")
            or decrypted.get("key")
            or ""
        ).strip()
        api_secret = str(
            decrypted.get("APCA-API-SECRET-KEY")
            or decrypted.get("api_secret")
            or decrypted.get("secret")
            or ""
        ).strip()
        trading_base_url = str(
            decrypted.get("trading_base_url") or decrypted.get("base_url") or ""
        ).strip()
        if not trading_base_url:
            from src.config import settings

            trading_base_url = settings.alpaca_paper_trading_base_url

    adapter = AlpacaTradingAdapter(
        api_key=api_key,
        api_secret=api_secret,
        trading_base_url=trading_base_url,
    )
    try:
        provider_order = await adapter.fetch_order(provider_order_id)
        account_state = await adapter.fetch_account_state()
    finally:
        await adapter.aclose()

    return {
        "provider_order_found": provider_order is not None,
        "provider_order_status": provider_order.status if provider_order is not None else None,
        "provider_order_symbol": provider_order.symbol if provider_order is not None else None,
        "provider_order_qty": str(provider_order.qty) if provider_order is not None else None,
        "alpaca_account_snapshot": {
            "cash": str(account_state.cash),
            "equity": str(account_state.equity),
            "buying_power": str(account_state.buying_power),
            "margin_used": str(account_state.margin_used),
        },
    }


async def _trigger_backtest_completion_for_notifications(
    *,
    user_email: str,
    strategy_id: str,
) -> dict[str, Any]:
    from src.engine.backtest.service import create_backtest_job, execute_backtest_job
    from src.engine.data.data_loader import DataLoader

    strategy_uuid = UUID(strategy_id)
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == user_email))
        if user is None:
            raise RuntimeError(f"user not found: {user_email}")
        strategy = await db.scalar(
            select(Strategy).where(Strategy.id == strategy_uuid, Strategy.user_id == user.id)
        )
        if strategy is None:
            raise RuntimeError(f"strategy not found for user: {strategy_id}")

        payload = strategy.dsl_payload if isinstance(strategy.dsl_payload, dict) else {}
        universe = payload.get("universe") if isinstance(payload.get("universe"), dict) else {}
        raw_market = str(universe.get("market") or "us_stocks").strip()
        tickers = universe.get("tickers") if isinstance(universe.get("tickers"), list) else []
        symbol = str(tickers[0]).strip() if tickers else "SPY"

        loader = DataLoader()
        metadata = loader.get_symbol_metadata(raw_market, symbol)
        timerange = metadata.get("available_timerange") if isinstance(metadata, dict) else {}
        start_raw = str(timerange.get("start") or "").strip()
        end_raw = str(timerange.get("end") or "").strip()
        if not start_raw or not end_raw:
            raise RuntimeError("unable to resolve market data timerange for backtest notification probe.")

        end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(UTC)
        start_dt = end_dt - timedelta(days=5)
        start_floor = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(UTC)
        if start_dt < start_floor:
            start_dt = start_floor

        receipt = await create_backtest_job(
            db,
            strategy_id=strategy.id,
            user_id=user.id,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
            auto_commit=True,
        )
        view = await execute_backtest_job(
            db,
            job_id=receipt.job_id,
            auto_commit=True,
        )
        return {
            "job_id": str(receipt.job_id),
            "status": view.status,
            "progress": view.progress,
            "completed_at": view.completed_at.isoformat() if view.completed_at else None,
            "error": view.error,
        }


async def _collect_notification_snapshot(
    *,
    user_email: str,
    created_after: datetime,
) -> dict[str, Any]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None
    async with db_module.AsyncSessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == user_email))
        if user is None:
            raise RuntimeError(f"user not found: {user_email}")

        rows = (
            await db.scalars(
                select(NotificationOutbox)
                .where(
                    NotificationOutbox.user_id == user.id,
                    NotificationOutbox.created_at >= created_after,
                )
                .order_by(NotificationOutbox.created_at.asc()),
            )
        ).all()
        outbox_ids = [row.id for row in rows]
        attempts: list[NotificationDeliveryAttempt] = []
        if outbox_ids:
            attempts = (
                await db.scalars(
                    select(NotificationDeliveryAttempt)
                    .where(NotificationDeliveryAttempt.outbox_id.in_(outbox_ids))
                    .order_by(NotificationDeliveryAttempt.attempted_at.asc()),
                )
            ).all()

        return {
            "outbox_count": len(rows),
            "outbox": [
                {
                    "outbox_id": str(row.id),
                    "channel": row.channel,
                    "event_type": row.event_type,
                    "event_key": row.event_key,
                    "status": row.status,
                    "retry_count": row.retry_count,
                    "last_error": row.last_error,
                    "scheduled_at": row.scheduled_at.isoformat(),
                    "next_retry_at": row.next_retry_at.isoformat()
                    if row.next_retry_at is not None
                    else None,
                    "sent_at": row.sent_at.isoformat() if row.sent_at is not None else None,
                    "payload": row.payload if isinstance(row.payload, dict) else {},
                }
                for row in rows
            ],
            "attempt_count": len(attempts),
            "attempts": [
                {
                    "attempt_id": str(item.id),
                    "outbox_id": str(item.outbox_id),
                    "provider": item.provider,
                    "success": item.success,
                    "error_code": item.error_code,
                    "error_message": item.error_message,
                    "attempted_at": item.attempted_at.isoformat(),
                    "response_payload": (
                        item.response_payload if isinstance(item.response_payload, dict) else {}
                    ),
                }
                for item in attempts
            ],
        }


def _wait_for_deployment_active(
    client: FrontendLikeClient,
    *,
    deployment_id: str,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        latest = client.get_deployment(deployment_id)
        status = str(latest.get("status", "")).strip().lower()
        if status == "active":
            return latest
        time.sleep(3.0)
    if latest is None:
        raise RuntimeError("deployment active wait failed: no response payload.")
    return latest


def _wait_for_first_order(
    client: FrontendLikeClient,
    *,
    deployment_id: str,
    baseline_count: int,
    timeout_seconds: float = 240.0,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        latest = client.get_orders(deployment_id)
        if len(latest) > baseline_count:
            return latest
        time.sleep(5.0)
    return latest


def _pick_target_deployment(
    *,
    deployments_payload: list[dict[str, Any]],
    strategy_id: str | None,
    deployment_id_hint: str | None,
) -> dict[str, Any]:
    if deployment_id_hint:
        for item in deployments_payload:
            if str(item.get("deployment_id")) == deployment_id_hint:
                return item
    if strategy_id:
        for item in deployments_payload:
            if str(item.get("strategy_id")) == strategy_id:
                return item
    if deployments_payload:
        return deployments_payload[0]
    raise RuntimeError("No deployments found for current user after deployment turn.")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(UTC)

    client = FrontendLikeClient(
        base_url=args.base_url,
        email=args.email,
        password=args.password,
        language=args.language,
    )

    report: dict[str, Any] = {
        "started_at": started_at.isoformat(),
        "config": {
            "base_url": args.base_url,
            "email": args.email,
            "language": args.language,
        },
        "static_sync_checks": {},
        "preflight": {},
        "step_1_strategy_to_deployment": {},
        "step_2_paper_trading_runtime": {},
        "step_3_notifications": {},
        "errors": [],
    }

    report["static_sync_checks"] = collect_static_sync_checks()
    report["preflight"] = asyncio.run(_fetch_user_and_binding_snapshot(user_email=args.email))

    login_payload = client.login()
    thread_payload = client.create_thread(metadata={"label": "audit_frontend_sim_chain"})
    session_id = str(thread_payload["session_id"])

    report["step_1_strategy_to_deployment"]["thread"] = {
        "session_id": session_id,
        "initial_phase": thread_payload.get("phase"),
        "auth_user_id": login_payload.get("user_id"),
    }

    pre_strategy_message = (
        "我想交易 us_stocks 市场，标的是 SPY，机会频率 daily，持有周期 swing_days。"
        "请直接按流程推进。"
    )
    if UUID_PATTERN.search(pre_strategy_message):
        raise RuntimeError("pre_strategy message unexpectedly contains UUID.")
    turn_1 = client.send_turn(
        session_id=session_id,
        message=pre_strategy_message,
        max_stream_seconds=args.max_turn_seconds,
    )

    strategy_message = (
        "请直接生成一个可执行的最简策略 DSL，并且立刻保存，不要等待我点击任何确认按钮。"
        "要求：ticker=SPY，timeframe=1m，只做 long；开仓条件 price.close > 0；平仓条件 price.close < 0；"
        "position sizing 固定 1 手。保存后请把流程自动推进到 deployment 阶段。"
    )
    if UUID_PATTERN.search(strategy_message):
        raise RuntimeError("strategy message unexpectedly contains UUID.")
    turn_2 = client.send_turn(
        session_id=session_id,
        message=strategy_message,
        max_stream_seconds=args.max_turn_seconds,
    )

    deployment_message = (
        "现在请你在聊天里直接完成 paper deployment 并启动运行。"
        "不要让我提供 strategy_id 或 deployment_id，也不要让我点击按钮。"
    )
    if UUID_PATTERN.search(deployment_message):
        raise RuntimeError("deployment message unexpectedly contains UUID.")
    turn_3 = client.send_turn(
        session_id=session_id,
        message=deployment_message,
        max_stream_seconds=args.max_turn_seconds,
    )

    turn_2_session = turn_2.session_snapshot
    turn_3_session = turn_3.session_snapshot
    turn_2_phase = str(turn_2_session.get("current_phase", ""))
    turn_3_phase = str(turn_3_session.get("current_phase", ""))

    strategy_profile = (
        (((turn_2_session.get("artifacts") or {}).get("strategy") or {}).get("profile"))
        if isinstance(turn_2_session.get("artifacts"), dict)
        else {}
    )
    deployment_profile = (
        (((turn_3_session.get("artifacts") or {}).get("deployment") or {}).get("profile"))
        if isinstance(turn_3_session.get("artifacts"), dict)
        else {}
    )
    strategy_id = None
    if isinstance(strategy_profile, dict):
        raw_strategy_id = strategy_profile.get("strategy_id")
        if isinstance(raw_strategy_id, str) and raw_strategy_id.strip():
            strategy_id = raw_strategy_id.strip()
    if not strategy_id and isinstance(deployment_profile, dict):
        raw_strategy_id = deployment_profile.get("strategy_id")
        if isinstance(raw_strategy_id, str) and raw_strategy_id.strip():
            strategy_id = raw_strategy_id.strip()

    turn_2_tools = _extract_tool_names_from_message(turn_2.assistant_message)
    turn_3_tools = _extract_tool_names_from_message(turn_3.assistant_message)

    deployment_id_from_tools = _extract_deployment_id_from_tool_calls(turn_3.assistant_message)
    deployments_payload = client.list_deployments()
    target_deployment = _pick_target_deployment(
        deployments_payload=deployments_payload,
        strategy_id=strategy_id,
        deployment_id_hint=deployment_id_from_tools,
    )
    deployment_id = str(target_deployment["deployment_id"])

    report["step_1_strategy_to_deployment"] = {
        "turn_1": {
            "message": pre_strategy_message,
            "stream_timed_out": turn_1.stream_timed_out,
            "done_payload": turn_1.done_payload,
            "assistant_text": turn_1.assistant_message.get("content"),
            "phase_after_turn": turn_1.session_snapshot.get("current_phase"),
            "tool_calls": _extract_tool_names_from_message(turn_1.assistant_message),
        },
        "turn_2": {
            "message": strategy_message,
            "stream_timed_out": turn_2.stream_timed_out,
            "done_payload": turn_2.done_payload,
            "assistant_text": turn_2.assistant_message.get("content"),
            "phase_after_turn": turn_2_phase,
            "tool_calls": turn_2_tools,
            "strategy_profile": strategy_profile if isinstance(strategy_profile, dict) else {},
        },
        "turn_3": {
            "message": deployment_message,
            "stream_timed_out": turn_3.stream_timed_out,
            "done_payload": turn_3.done_payload,
            "assistant_text": turn_3.assistant_message.get("content"),
            "phase_after_turn": turn_3_phase,
            "tool_calls": turn_3_tools,
            "deployment_profile": deployment_profile if isinstance(deployment_profile, dict) else {},
        },
        "assertions": {
            "strategy_to_deployment_phase_transition": turn_2_phase == "deployment",
            "strategy_id_present_in_artifacts": bool(strategy_id),
            "deployment_tools_called": any(
                name in {
                    "trading_create_paper_deployment",
                    "trading_start_deployment",
                    "trading_list_deployments",
                }
                for name in turn_3_tools
            ),
            "manual_ids_sent_in_user_messages": False,
        },
        "resolved_ids": {
            "strategy_id": strategy_id,
            "deployment_id": deployment_id,
        },
    }

    active_payload = _wait_for_deployment_active(
        client,
        deployment_id=deployment_id,
        timeout_seconds=args.max_wait_deployment_active_seconds,
    )
    baseline_orders = client.get_orders(deployment_id)
    orders_after = _wait_for_first_order(
        client,
        deployment_id=deployment_id,
        baseline_count=len(baseline_orders),
        timeout_seconds=args.max_wait_orders_seconds,
    )
    positions_after = client.get_positions(deployment_id)
    portfolio_after = client.get_portfolio(deployment_id)
    symbol = "SPY"
    if isinstance(active_payload.get("symbols"), list) and active_payload["symbols"]:
        symbol = str(active_payload["symbols"][0]).strip() or symbol
    market_bars_payload = client.get_market_bars(
        symbol=symbol,
        market="stocks",
        timeframe="1m",
        limit=20,
    )

    db_runtime_snapshot = asyncio.run(_fetch_runtime_snapshot(deployment_id=deployment_id))
    first_order = orders_after[0] if orders_after else None
    provider_order_id = str(first_order.get("provider_order_id", "")).strip() if first_order else ""

    alpaca_provider_check = {
        "provider_order_id": provider_order_id or None,
        "provider_order_id_looks_local_simulated": provider_order_id.startswith("paper-"),
        "provider_query": None,
    }
    if provider_order_id and not provider_order_id.startswith("paper-"):
        try:
            alpaca_provider_check["provider_query"] = asyncio.run(
                _validate_provider_order_on_alpaca(
                    user_email=args.email,
                    deployment_id=deployment_id,
                    provider_order_id=provider_order_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            alpaca_provider_check["provider_query"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    report["step_2_paper_trading_runtime"] = {
        "deployment_api": active_payload,
        "orders_before_count": len(baseline_orders),
        "orders_after_count": len(orders_after),
        "first_order": first_order,
        "positions_count": len(positions_after),
        "positions_preview": positions_after[:5],
        "portfolio": portfolio_after,
        "market_data_bars_count": len((market_bars_payload.get("bars") or [])),
        "market_data_sample": (market_bars_payload.get("bars") or [])[-3:],
        "db_runtime_snapshot": db_runtime_snapshot,
        "alpaca_provider_check": alpaca_provider_check,
        "assertions": {
            "deployment_active": str(active_payload.get("status", "")).lower() == "active",
            "orders_created": len(orders_after) > len(baseline_orders),
            "positions_updated": len(positions_after) >= 1,
            "portfolio_has_equity": isinstance(portfolio_after.get("equity"), float | int),
            "celery_runtime_signal_present": int(
                ((db_runtime_snapshot.get("counts") or {}).get("signals") or 0)
            )
            > 0,
            "runtime_scheduler_state_present": isinstance(
                (((db_runtime_snapshot.get("run") or {}).get("runtime_state") or {}).get("scheduler")),
                dict,
            ),
            "order_sent_to_real_alpaca": bool(
                provider_order_id
                and not provider_order_id.startswith("paper-")
                and isinstance(alpaca_provider_check.get("provider_query"), dict)
                and alpaca_provider_check["provider_query"].get("provider_order_found") is True
            ),
        },
    }

    backtest_probe: dict[str, Any] | None = None
    if strategy_id:
        try:
            backtest_probe = asyncio.run(
                _trigger_backtest_completion_for_notifications(
                    user_email=args.email,
                    strategy_id=strategy_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            backtest_probe = {"error": f"{type(exc).__name__}: {exc}"}

    time.sleep(args.notification_settle_seconds)

    notification_snapshot = asyncio.run(
        _collect_notification_snapshot(
            user_email=args.email,
            created_after=started_at,
        )
    )

    outbox = notification_snapshot.get("outbox") if isinstance(notification_snapshot, dict) else []
    outbox = outbox if isinstance(outbox, list) else []
    event_types = [str(item.get("event_type")) for item in outbox if isinstance(item, dict)]
    has_deployment_started = "DEPLOYMENT_STARTED" in event_types
    has_position_opened = "POSITION_OPENED" in event_types
    has_backtest_completed = "BACKTEST_COMPLETED" in event_types
    statuses = {str(item.get("status")) for item in outbox if isinstance(item, dict)}
    delivery_attempts = notification_snapshot.get("attempts")
    delivery_attempts = delivery_attempts if isinstance(delivery_attempts, list) else []
    telegram_attempt_errors = sorted(
        {
            str(item.get("error_code"))
            for item in delivery_attempts
            if isinstance(item, dict)
            and str(item.get("provider", "")).lower() == "telegram"
            and not bool(item.get("success"))
            and item.get("error_code")
        }
    )

    report["step_3_notifications"] = {
        "backtest_probe": backtest_probe,
        "notification_snapshot": notification_snapshot,
        "assertions": {
            "deployment_started_event_enqueued": has_deployment_started,
            "position_opened_event_enqueued": has_position_opened,
            "backtest_completed_event_enqueued": has_backtest_completed,
            "outbox_rows_dispatched_or_failed": bool(statuses & {"sent", "failed", "dead"}),
            "telegram_delivery_success": any(
                isinstance(item, dict)
                and str(item.get("provider", "")).lower() == "telegram"
                and bool(item.get("success"))
                for item in delivery_attempts
            ),
            "telegram_delivery_failure_codes": telegram_attempt_errors,
        },
    }

    report["finished_at"] = datetime.now(UTC).isoformat()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backend full-chain audit via frontend-like flow.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Backend API base url.",
    )
    parser.add_argument("--email", default="2@test.com", help="Login email.")
    parser.add_argument("--password", default="123456", help="Login password.")
    parser.add_argument("--language", default="zh", help="Chat language query parameter.")
    parser.add_argument(
        "--max-turn-seconds",
        type=float,
        default=240.0,
        help="Max seconds to read one SSE turn before forced close.",
    )
    parser.add_argument(
        "--max-wait-deployment-active-seconds",
        type=float,
        default=180.0,
        help="Wait window for deployment to become active.",
    )
    parser.add_argument(
        "--max-wait-orders-seconds",
        type=float,
        default=240.0,
        help="Wait window for runtime-generated order appearance.",
    )
    parser.add_argument(
        "--notification-settle-seconds",
        type=float,
        default=15.0,
        help="Wait time for notifications worker to dispatch outbox rows.",
    )
    parser.add_argument(
        "--report-path",
        default="artifacts/audit_run/frontend_sim_chain_latest.json",
        help="Output report JSON path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = run_audit(args)
    except Exception as exc:  # noqa: BLE001
        error_report = {
            "started_at": datetime.now(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "trace_hint": "Run with backend/celery services up and inspect logs for details.",
        }
        report_path.write_text(json.dumps(error_report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps(error_report, ensure_ascii=True, indent=2))
        return 1
    finally:
        try:
            asyncio.run(db_module.close_postgres())
        except Exception:
            pass

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
