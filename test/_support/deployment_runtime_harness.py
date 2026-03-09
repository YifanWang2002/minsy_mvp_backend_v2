from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from test._support.live_helpers import (
    BACKEND_DIR,
    COMPOSE_FILE,
    compose_ps,
    parse_sse_payloads,
    run_command,
    wait_http_ok,
)

_HARNESS_REPORT_DIR = BACKEND_DIR / "runtime" / "harness_reports"
_DEFAULT_PASSWORD = "test1234"
_DEFAULT_MCP_TIMEOUT_SECONDS = 60

_DEPLOYMENT_COUNT_QUERIES: dict[str, str] = {
    "deployments": "select count(*) from deployments where id = :deployment_id",
    "deployment_runs": (
        "select count(*) from deployment_runs where deployment_id = :deployment_id"
    ),
    "manual_trade_actions": (
        "select count(*) from manual_trade_actions where deployment_id = :deployment_id"
    ),
    "orders": "select count(*) from orders where deployment_id = :deployment_id",
    "fills": (
        "select count(*) from fills f "
        "join orders o on o.id = f.order_id "
        "where o.deployment_id = :deployment_id"
    ),
    "positions": "select count(*) from positions where deployment_id = :deployment_id",
    "pnl_snapshots": (
        "select count(*) from pnl_snapshots where deployment_id = :deployment_id"
    ),
    "signal_events": (
        "select count(*) from signal_events where deployment_id = :deployment_id"
    ),
    "trade_approval_requests": (
        "select count(*) from trade_approval_requests where deployment_id = :deployment_id"
    ),
    "trading_event_outbox": (
        "select count(*) from trading_event_outbox where deployment_id = :deployment_id"
    ),
}

_USER_COUNT_QUERIES: dict[str, str] = {
    "notification_outbox": (
        "select count(*) from notification_outbox where user_id = :user_id"
    ),
    "notification_delivery_attempts": (
        "select count(*) from notification_delivery_attempts nda "
        "join notification_outbox no on no.id = nda.outbox_id "
        "where no.user_id = :user_id"
    ),
    "social_connector_bindings": (
        "select count(*) from social_connector_bindings where user_id = :user_id"
    ),
    "social_connector_link_intents": (
        "select count(*) from social_connector_link_intents where user_id = :user_id"
    ),
    "social_connector_activities": (
        "select count(*) from social_connector_activities where user_id = :user_id"
    ),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    return str(value)


def _render_markdown_value(value: Any) -> str:
    normalized = _safe_json(value)
    if isinstance(normalized, str):
        return normalized
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def _extract_first_sse_payload(raw_text: str) -> dict[str, Any]:
    payloads = parse_sse_payloads(raw_text)
    if payloads:
        return payloads[0]
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        return decoded
    raise AssertionError(f"No SSE payload found in response: {raw_text[:300]}")


def _parse_sse_frames(raw_text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    blocks = [item.strip() for item in raw_text.split("\n\n") if item.strip()]
    for block in blocks:
        event_name = "message"
        payload: dict[str, Any] | None = None
        raw_data: str | None = None
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("event:"):
                event_name = stripped.removeprefix("event:").strip() or "message"
            elif stripped.startswith("data:"):
                raw_data = stripped.removeprefix("data:").strip()
        if raw_data is not None:
            try:
                decoded = json.loads(raw_data)
            except json.JSONDecodeError:
                decoded = {"raw": raw_data}
            if isinstance(decoded, dict):
                payload = decoded
            else:
                payload = {"value": decoded}
        frames.append(
            {
                "event": event_name,
                "payload": payload or {},
            }
        )
    return frames


def _build_deployable_dsl() -> dict[str, object]:
    suffix = uuid4().hex[:8]
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": f"Harness Deploy {suffix}",
            "description": "Deployment runtime harness live scenario",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTC/USD"],
        },
        "timeframe": "1m",
        "factors": {
            "ema_9": {
                "type": "ema",
                "params": {"period": 9, "source": "close"},
            },
            "ema_21": {
                "type": "ema",
                "params": {"period": 21, "source": "close"},
            },
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_9"},
                            "op": "cross_above",
                            "b": {"ref": "ema_21"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_cross_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_9"},
                                "op": "cross_below",
                                "b": {"ref": "ema_21"},
                            }
                        },
                    },
                    {
                        "type": "bracket_rr",
                        "name": "protective_bracket",
                        "stop": {"kind": "pct", "value": 0.01},
                        "risk_reward": 2.0,
                    },
                ],
            }
        },
    }


def _resolve_mcp_context_helpers() -> tuple[str, Any]:
    from packages.infra.auth.mcp_context import (
        MCP_CONTEXT_HEADER,
        create_mcp_context_token,
    )

    return MCP_CONTEXT_HEADER, create_mcp_context_token


def _resolve_db_session_module():
    from packages.infra.db import session as db_session_module

    return db_session_module


def _resolve_sync_redis_client():
    from packages.infra.redis.client import get_sync_redis_client

    return get_sync_redis_client


def _resolve_celery_app():
    from packages.infra.queue.celery_app import celery_app

    return celery_app


def _resolve_runtime_settings():
    from packages.shared_settings.schema.settings import settings

    return settings


def _resolve_social_connector_service():
    from packages.domain.user.services.social_connector_service import (
        SocialConnectorService,
    )

    return SocialConnectorService


def _resolve_trade_approval_service():
    from packages.domain.trading.services.trade_approval_service import (
        TradeApprovalService,
    )

    return TradeApprovalService


def _resolve_telegram_approval_codec():
    from packages.domain.trading.services.telegram_approval_codec import (
        TelegramApprovalCodec,
    )

    return TelegramApprovalCodec


def _resolve_notification_task():
    from apps.worker.io.tasks.notification import dispatch_pending_notifications_task

    return dispatch_pending_notifications_task


def _resolve_trade_approval_task():
    from apps.worker.io.tasks.trade_approval import execute_approved_open_task

    return execute_approved_open_task


@dataclass(slots=True)
class HarnessStep:
    name: str
    driver: str
    method: str
    target: str
    duration_ms: int
    ok: bool
    request: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    status_code: int | None = None
    note: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "driver": self.driver,
            "method": self.method,
            "target": self.target,
            "duration_ms": self.duration_ms,
            "ok": self.ok,
            "request": _safe_json(self.request),
            "response": _safe_json(self.response),
            "status_code": self.status_code,
            "note": self.note,
            "error": self.error,
        }


@dataclass(slots=True)
class DbSnapshot:
    table_counts: dict[str, int]
    deployment_status: str | None
    runtime_status: str | None
    latest_manual_action: dict[str, Any] | None
    latest_trade_approval: dict[str, Any] | None
    latest_trading_event: dict[str, Any] | None
    latest_notification: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_counts": _safe_json(self.table_counts),
            "deployment_status": self.deployment_status,
            "runtime_status": self.runtime_status,
            "latest_manual_action": _safe_json(self.latest_manual_action),
            "latest_trade_approval": _safe_json(self.latest_trade_approval),
            "latest_trading_event": _safe_json(self.latest_trading_event),
            "latest_notification": _safe_json(self.latest_notification),
        }


@dataclass(slots=True)
class ScenarioContext:
    user_id: UUID
    auth_headers: dict[str, str]
    session_id: str
    strategy_id: str
    broker_account_id: str
    deployment_id: str
    symbol: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": str(self.user_id),
            "session_id": self.session_id,
            "strategy_id": self.strategy_id,
            "broker_account_id": self.broker_account_id,
            "deployment_id": self.deployment_id,
            "symbol": self.symbol,
        }


@dataclass(slots=True)
class HarnessReport:
    scenario_name: str
    started_at: datetime
    completed_at: datetime
    context: ScenarioContext
    steps: list[HarnessStep]
    db_before: DbSnapshot | None
    db_after: DbSnapshot | None
    redis_probe: dict[str, Any]
    celery_probe: dict[str, Any]
    artifacts: dict[str, Any]
    report_json_path: Path
    report_markdown_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "started_at": _safe_json(self.started_at),
            "completed_at": _safe_json(self.completed_at),
            "context": self.context.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "db_before": self.db_before.to_dict() if self.db_before else None,
            "db_after": self.db_after.to_dict() if self.db_after else None,
            "redis_probe": _safe_json(self.redis_probe),
            "celery_probe": _safe_json(self.celery_probe),
            "artifacts": _safe_json(self.artifacts),
            "report_json_path": str(self.report_json_path),
            "report_markdown_path": str(self.report_markdown_path),
        }


class ApiDriver:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    def register_test_user(self) -> tuple[dict[str, str], UUID, HarnessStep]:
        email = f"deployment_harness_{uuid4().hex[:12]}@test.com"
        payload = {
            "email": email,
            "password": _DEFAULT_PASSWORD,
            "name": "Deployment Harness",
        }
        started = time.perf_counter()
        response = self._client.post("/api/v1/auth/register", json=payload)
        duration_ms = int((time.perf_counter() - started) * 1000)
        body = response.json()
        step = HarnessStep(
            name="register_test_user",
            driver="api",
            method="POST",
            target="/api/v1/auth/register",
            duration_ms=duration_ms,
            ok=response.status_code in {200, 201},
            request=payload,
            response=body,
            status_code=response.status_code,
        )
        assert response.status_code in {200, 201}, response.text
        access_token = str(body["access_token"])
        user_id = UUID(str(body["user_id"]))
        return {"Authorization": f"Bearer {access_token}"}, user_id, step

    def login_user(
        self,
        *,
        email: str,
        password: str,
    ) -> tuple[dict[str, str], UUID, HarnessStep]:
        payload = {
            "email": email.strip(),
            "password": password,
        }
        started = time.perf_counter()
        response = self._client.post("/api/v1/auth/login", json=payload)
        duration_ms = int((time.perf_counter() - started) * 1000)
        body = response.json()
        step = HarnessStep(
            name="login_seeded_user",
            driver="api",
            method="POST",
            target="/api/v1/auth/login",
            duration_ms=duration_ms,
            ok=response.status_code == 200,
            request={"email": payload["email"]},
            response=body,
            status_code=response.status_code,
            error=response.text if response.status_code != 200 else None,
        )
        assert response.status_code == 200, response.text
        access_token = str(body["access_token"])
        user_id = UUID(str(body["user_id"]))
        return {"Authorization": f"Bearer {access_token}"}, user_id, step

    def request(
        self,
        *,
        name: str,
        method: str,
        path: str,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expected_status: int | set[int] = 200,
        note: str | None = None,
    ) -> tuple[Any, HarnessStep]:
        started = time.perf_counter()
        response = self._client.request(
            method.upper(),
            path,
            headers=headers,
            json=json_body,
            params=params,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        allowed = (
            expected_status
            if isinstance(expected_status, set)
            else {int(expected_status)}
        )
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            payload = response.text
        step = HarnessStep(
            name=name,
            driver="api",
            method=method.upper(),
            target=path,
            duration_ms=duration_ms,
            ok=response.status_code in allowed,
            request={
                "headers": sorted(headers.keys()),
                "params": params or {},
                "json": json_body or {},
            },
            response=payload,
            status_code=response.status_code,
            note=note,
            error=response.text if response.status_code not in allowed else None,
        )
        assert response.status_code in allowed, response.text
        return payload, step


class ChatDriver:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    def send_message_stream(
        self,
        *,
        headers: dict[str, str],
        message: str,
        language: str = "en",
        session_id: str | None = None,
    ) -> tuple[dict[str, Any], HarnessStep]:
        query = f"?language={language}"
        if session_id is not None and session_id.strip():
            query = f"{query}&session_id={session_id.strip()}"
        path = f"/api/v1/chat/send-openai-stream{query}"
        payload = {"message": message}
        if session_id is not None and session_id.strip():
            payload["session_id"] = session_id.strip()
        started = time.perf_counter()
        response = self._client.post(path, headers=headers, json=payload)
        duration_ms = int((time.perf_counter() - started) * 1000)
        sse_payloads = parse_sse_payloads(response.text)
        text_delta = "".join(
            str(item.get("delta", ""))
            for item in sse_payloads
            if item.get("type") == "text_delta"
        )
        done_payload = next(
            (item for item in sse_payloads if item.get("type") == "done"),
            None,
        )
        summary = {
            "event_count": len(sse_payloads),
            "done": done_payload or {},
            "text_preview": text_delta[:500],
        }
        step = HarnessStep(
            name="chat_stream_turn",
            driver="chat",
            method="POST",
            target=path,
            duration_ms=duration_ms,
            ok=response.status_code == 200 and done_payload is not None,
            request=payload,
            response=summary,
            status_code=response.status_code,
            error=response.text[:500] if response.status_code != 200 else None,
        )
        assert response.status_code == 200, response.text
        assert done_payload is not None, response.text[:500]
        return summary, step


class McpDriver:
    def __init__(self, *, base_url: str = "http://127.0.0.1:8110") -> None:
        self._base_url = base_url.rstrip("/")

    def call_trading_tool(
        self,
        *,
        tool_name: str,
        user_id: UUID,
        session_id: str | None,
        arguments: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], HarnessStep]:
        mcp_context_header, create_mcp_context_token = _resolve_mcp_context_helpers()
        token = create_mcp_context_token(
            user_id=user_id,
            session_id=UUID(session_id) if session_id else None,
            phase="deployment",
            trace_id=f"deployment-harness-{uuid4().hex[:8]}",
        )
        request_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        request = Request(
            f"{self._base_url}/trading/mcp",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                mcp_context_header: token,
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=_DEFAULT_MCP_TIMEOUT_SECONDS) as response:
                raw_text = response.read().decode("utf-8")
                http_status = int(response.status)
        except HTTPError as exc:
            raw_text = exc.read().decode("utf-8")
            http_status = int(exc.code)
        except URLError as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            step = HarnessStep(
                name=tool_name,
                driver="mcp",
                method="POST",
                target="/trading/mcp",
                duration_ms=duration_ms,
                ok=False,
                request=request_payload,
                response=None,
                status_code=None,
                error=str(exc),
            )
            raise AssertionError(f"MCP tool call failed: {exc}") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        envelope = _extract_first_sse_payload(raw_text)
        result = envelope.get("result")
        assert isinstance(result, dict), envelope
        content = result.get("content")
        decoded_payload: dict[str, Any] | None = None
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                text_payload = first.get("text")
                if isinstance(text_payload, str) and text_payload.strip():
                    decoded_payload = json.loads(text_payload)
        if decoded_payload is None:
            decoded_payload = {"raw_result": result}
        step = HarnessStep(
            name=tool_name,
            driver="mcp",
            method="POST",
            target="/trading/mcp",
            duration_ms=duration_ms,
            ok=http_status == 200 and bool(decoded_payload.get("ok", True)),
            request=request_payload,
            response=decoded_payload,
            status_code=http_status,
            error=raw_text[:500] if http_status != 200 else None,
        )
        assert http_status == 200, raw_text
        return decoded_payload, step


class DomainDriver:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    def seed_trade_approval_request(
        self,
        *,
        user_id: UUID,
        deployment_id: str,
        symbol: str,
        side: str = "long",
        qty: str = "1",
        mark_price: str = "100",
        timeframe: str = "1m",
        approval_channel: str = "telegram",
        timeout_seconds: int = 180,
    ) -> tuple[dict[str, Any], HarnessStep]:
        deployment_uuid = UUID(deployment_id)
        TradeApprovalService = _resolve_trade_approval_service()
        TelegramApprovalCodec = _resolve_telegram_approval_codec()
        db_session_module = _resolve_db_session_module()

        async def _seed() -> dict[str, Any]:
            from decimal import Decimal

            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                service = TradeApprovalService(db)
                request, created = await service.create_or_get_open_request(
                    user_id=user_id,
                    deployment_id=deployment_uuid,
                    signal="OPEN_LONG" if side == "long" else "OPEN_SHORT",
                    side=side,
                    symbol=symbol,
                    qty=Decimal(str(qty)),
                    mark_price=Decimal(str(mark_price)),
                    reason="deployment_runtime_harness_seed",
                    timeframe=timeframe,
                    bar_time=None,
                    approval_channel=approval_channel,
                    approval_timeout_seconds=timeout_seconds,
                    intent_payload={
                        "source": "deployment_runtime_harness",
                        "symbol": symbol,
                        "side": side,
                        "qty": str(qty),
                        "mark_price": str(mark_price),
                    },
                )
                await db.commit()
                codec = TelegramApprovalCodec()
                approve_callback = codec.encode(
                    request_id=request.id,
                    action="approve",
                    expires_at=request.expires_at,
                )
                reject_callback = codec.encode(
                    request_id=request.id,
                    action="reject",
                    expires_at=request.expires_at,
                )
                return {
                    "request_id": str(request.id),
                    "created": created,
                    "status": request.status,
                    "expires_at": request.expires_at.isoformat(),
                    "approval_channel": request.approval_channel,
                    "approve_callback_data": approve_callback,
                    "reject_callback_data": reject_callback,
                }

        started = time.perf_counter()
        assert self._client.portal is not None
        payload = self._client.portal.call(_seed)
        duration_ms = int((time.perf_counter() - started) * 1000)
        step = HarnessStep(
            name="seed_trade_approval_request",
            driver="domain",
            method="SEED",
            target="trade_approval_requests",
            duration_ms=duration_ms,
            ok=bool(payload.get("request_id")),
            request={
                "deployment_id": deployment_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "mark_price": mark_price,
                "approval_channel": approval_channel,
            },
            response=payload,
            note="Deterministically seeds a pending approval request and notification outbox row.",
        )
        return payload, step

    def seed_telegram_binding(
        self,
        *,
        user_id: UUID,
        chat_id: str,
        external_user_id: str,
        username: str,
        locale: str = "en",
    ) -> tuple[dict[str, Any], HarnessStep]:
        SocialConnectorService = _resolve_social_connector_service()
        db_session_module = _resolve_db_session_module()

        async def _seed() -> dict[str, Any]:
            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                service = SocialConnectorService(db)
                binding = await service.upsert_telegram_binding(
                    user_id=user_id,
                    telegram_chat_id=chat_id,
                    telegram_user_id=external_user_id,
                    telegram_username=username,
                    locale=locale,
                )
                await db.commit()
                return {
                    "binding_id": str(binding.id),
                    "provider": binding.provider,
                    "status": binding.status,
                    "chat_id": binding.external_chat_id,
                    "external_user_id": binding.external_user_id,
                    "external_username": binding.external_username,
                }

        started = time.perf_counter()
        assert self._client.portal is not None
        payload = self._client.portal.call(_seed)
        duration_ms = int((time.perf_counter() - started) * 1000)
        step = HarnessStep(
            name="seed_telegram_binding",
            driver="domain",
            method="SEED",
            target="social_connector_bindings",
            duration_ms=duration_ms,
            ok=bool(payload.get("binding_id")),
            request={
                "chat_id": chat_id,
                "external_user_id": external_user_id,
                "username": username,
                "locale": locale,
            },
            response=payload,
            note="Fallback path when Telegram connect-link/webhook is unavailable.",
        )
        return payload, step

    def get_telegram_binding(
        self,
        *,
        user_id: UUID,
    ) -> dict[str, Any] | None:
        SocialConnectorService = _resolve_social_connector_service()
        db_session_module = _resolve_db_session_module()

        async def _load() -> dict[str, Any] | None:
            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                service = SocialConnectorService(db)
                binding = await service.get_connected_binding_for_user(
                    user_id=user_id,
                    provider="telegram",
                )
                if binding is None:
                    return None
                return {
                    "binding_id": str(binding.id),
                    "provider": binding.provider,
                    "status": binding.status,
                    "chat_id": binding.external_chat_id,
                    "external_user_id": binding.external_user_id,
                    "external_username": binding.external_username,
                }

        assert self._client.portal is not None
        return self._client.portal.call(_load)

    def get_trade_approval(
        self,
        *,
        request_id: str,
    ) -> dict[str, Any] | None:
        TradeApprovalService = _resolve_trade_approval_service()
        db_session_module = _resolve_db_session_module()
        request_uuid = UUID(request_id)

        async def _load() -> dict[str, Any] | None:
            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                service = TradeApprovalService(db)
                row = await service.get_by_id(request_id=request_uuid)
                if row is None:
                    return None
                return {
                    "request_id": str(row.id),
                    "status": row.status,
                    "approved_via": row.approved_via,
                    "execution_error": row.execution_error,
                    "execution_order_id": (
                        str(row.execution_order_id)
                        if row.execution_order_id is not None
                        else None
                    ),
                    "approved_at": (
                        row.approved_at.isoformat() if row.approved_at is not None else None
                    ),
                    "executed_at": (
                        row.executed_at.isoformat() if row.executed_at is not None else None
                    ),
                }

        assert self._client.portal is not None
        return self._client.portal.call(_load)

    def run_notification_dispatch(
        self,
        *,
        limit: int = 10,
    ) -> tuple[dict[str, Any], HarnessStep]:
        task = _resolve_notification_task()
        started = time.perf_counter()
        payload = dict(task.run(limit=limit))
        duration_ms = int((time.perf_counter() - started) * 1000)
        step = HarnessStep(
            name="dispatch_pending_notifications",
            driver="worker",
            method="RUN",
            target="notifications.dispatch_pending",
            duration_ms=duration_ms,
            ok=True,
            request={"limit": limit},
            response=payload,
        )
        return payload, step

    def run_execute_approved_open(
        self,
        *,
        request_id: str,
    ) -> tuple[dict[str, Any], HarnessStep]:
        task = _resolve_trade_approval_task()
        started = time.perf_counter()
        payload = dict(task.run(request_id))
        duration_ms = int((time.perf_counter() - started) * 1000)
        step = HarnessStep(
            name="execute_approved_open_inline",
            driver="worker",
            method="RUN",
            target="trade_approval.execute_approved_open",
            duration_ms=duration_ms,
            ok=payload.get("status") not in {None, "missing", "failed"},
            request={"request_id": request_id},
            response=payload,
        )
        return payload, step


class WebhookDriver:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    def telegram_update(
        self,
        *,
        payload: dict[str, Any],
        expected_status: int | set[int] = 200,
    ) -> tuple[dict[str, Any], HarnessStep]:
        settings = _resolve_runtime_settings()
        headers: dict[str, str] = {}
        secret = str(settings.telegram_webhook_secret_token).strip()
        if secret:
            headers["X-Telegram-Bot-Api-Secret-Token"] = secret
        started = time.perf_counter()
        response = self._client.post(
            "/api/v1/social/webhooks/telegram",
            headers=headers,
            json=payload,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        allowed = (
            expected_status
            if isinstance(expected_status, set)
            else {int(expected_status)}
        )
        body = response.json()
        step = HarnessStep(
            name="telegram_webhook_update",
            driver="webhook",
            method="POST",
            target="/api/v1/social/webhooks/telegram",
            duration_ms=duration_ms,
            ok=response.status_code in allowed,
            request=payload,
            response=body,
            status_code=response.status_code,
            error=response.text if response.status_code not in allowed else None,
        )
        assert response.status_code in allowed, response.text
        return body, step


class DbObserver:
    def __init__(self, client: TestClient) -> None:
        self._client = client

    def capture(self, *, deployment_id: str, user_id: UUID) -> DbSnapshot:
        deployment_uuid = UUID(deployment_id)
        db_session_module = _resolve_db_session_module()

        async def _load() -> DbSnapshot:
            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                table_counts: dict[str, int] = {}

                for label, sql in _DEPLOYMENT_COUNT_QUERIES.items():
                    value = await db.scalar(
                        text(sql),
                        {"deployment_id": deployment_uuid},
                    )
                    table_counts[label] = int(value or 0)

                for label, sql in _USER_COUNT_QUERIES.items():
                    value = await db.scalar(text(sql), {"user_id": user_id})
                    table_counts[label] = int(value or 0)

                deployment_status = await db.scalar(
                    text(
                        "select status from deployments "
                        "where id = :deployment_id limit 1"
                    ),
                    {"deployment_id": deployment_uuid},
                )
                runtime_status = await db.scalar(
                    text(
                        "select status from deployment_runs "
                        "where deployment_id = :deployment_id "
                        "order by created_at desc limit 1"
                    ),
                    {"deployment_id": deployment_uuid},
                )
                manual_row = (
                    (
                        await db.execute(
                            text(
                                "select action, status, payload, created_at "
                                "from manual_trade_actions "
                                "where deployment_id = :deployment_id "
                                "order by created_at desc limit 1"
                            ),
                            {"deployment_id": deployment_uuid},
                        )
                    )
                    .mappings()
                    .first()
                )
                approval_row = (
                    (
                        await db.execute(
                            text(
                                "select status, signal, side, symbol, qty, "
                                "approval_channel, requested_at, executed_at "
                                "from trade_approval_requests "
                                "where deployment_id = :deployment_id "
                                "order by requested_at desc limit 1"
                            ),
                            {"deployment_id": deployment_uuid},
                        )
                    )
                    .mappings()
                    .first()
                )
                event_row = (
                    (
                        await db.execute(
                            text(
                                "select event_seq, event_type, payload, occurred_at "
                                "from trading_event_outbox "
                                "where deployment_id = :deployment_id "
                                "order by event_seq desc limit 1"
                            ),
                            {"deployment_id": deployment_uuid},
                        )
                    )
                    .mappings()
                    .first()
                )
                notification_row = (
                    (
                        await db.execute(
                            text(
                                "select channel, event_type, status, retry_count, "
                                "scheduled_at, sent_at "
                                "from notification_outbox "
                                "where user_id = :user_id "
                                "order by created_at desc limit 1"
                            ),
                            {"user_id": user_id},
                        )
                    )
                    .mappings()
                    .first()
                )

                return DbSnapshot(
                    table_counts=table_counts,
                    deployment_status=(
                        str(deployment_status) if deployment_status is not None else None
                    ),
                    runtime_status=(
                        str(runtime_status) if runtime_status is not None else None
                    ),
                    latest_manual_action=dict(manual_row) if manual_row else None,
                    latest_trade_approval=dict(approval_row) if approval_row else None,
                    latest_trading_event=dict(event_row) if event_row else None,
                    latest_notification=dict(notification_row)
                    if notification_row
                    else None,
                )

        assert self._client.portal is not None
        return self._client.portal.call(_load)


class RedisObserver:
    def probe(self, *, deployment_id: str) -> dict[str, Any]:
        redis = _resolve_sync_redis_client()()
        runtime_key = f"paper_trading:runtime_state:{deployment_id}"
        queue_lengths: dict[str, int] = {}
        for queue_name in (
            "paper_trading",
            "trade_approval",
            "notifications",
            "market_data",
            "maintenance",
        ):
            try:
                queue_lengths[queue_name] = int(redis.llen(queue_name))
            except Exception:  # noqa: BLE001
                queue_lengths[queue_name] = -1
        runtime_state_raw = redis.get(runtime_key)
        runtime_state: dict[str, Any] | str | None
        if isinstance(runtime_state_raw, str) and runtime_state_raw.strip():
            try:
                runtime_state = json.loads(runtime_state_raw)
            except json.JSONDecodeError:
                runtime_state = runtime_state_raw
        else:
            runtime_state = None
        health_raw = redis.get("paper_trading:runtime_state:__live_trading_health__")
        health_payload: dict[str, Any] | None = None
        if isinstance(health_raw, str) and health_raw.strip():
            try:
                decoded = json.loads(health_raw)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                health_payload = decoded
        return {
            "runtime_key": runtime_key,
            "runtime_state": runtime_state,
            "live_trading_health": health_payload,
            "queue_lengths": queue_lengths,
        }


class CeleryObserver:
    def probe(self) -> dict[str, Any]:
        inspect = _resolve_celery_app().control.inspect(timeout=1.5)
        try:
            ping = inspect.ping() or {}
        except Exception as exc:  # noqa: BLE001
            ping = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            stats = inspect.stats() or {}
        except Exception as exc:  # noqa: BLE001
            stats = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "ping": _safe_json(ping),
            "stats_keys": sorted(stats.keys()) if isinstance(stats, dict) else [],
        }


class HarnessReporter:
    def write(self, report: HarnessReport) -> None:
        _HARNESS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report.report_json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report.report_markdown_path.write_text(
            self._to_markdown(report),
            encoding="utf-8",
        )

    def _to_markdown(self, report: HarnessReport) -> str:
        lines = [
            f"# {report.scenario_name}",
            "",
            f"- Started: {_render_markdown_value(report.started_at)}",
            f"- Completed: {_render_markdown_value(report.completed_at)}",
            f"- Deployment: {report.context.deployment_id}",
            f"- User: {report.context.user_id}",
            "",
            "## Steps",
        ]
        for step in report.steps:
            lines.append(
                (
                    f"- {step.name}: ok={step.ok} driver={step.driver} "
                    f"method={step.method} target={step.target} "
                    f"status={step.status_code} duration_ms={step.duration_ms}"
                )
            )
            if step.note:
                lines.append(f"  note: {step.note}")
            if step.error:
                lines.append(f"  error: {step.error}")
        if report.db_before is not None:
            lines.extend(
                [
                    "",
                    "## DB Before",
                    "```json",
                    json.dumps(
                        report.db_before.to_dict(),
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        if report.db_after is not None:
            lines.extend(
                [
                    "",
                    "## DB After",
                    "```json",
                    json.dumps(
                        report.db_after.to_dict(),
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        lines.extend(
            [
                "",
                "## Redis Probe",
                "```json",
                json.dumps(
                    _safe_json(report.redis_probe),
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "## Celery Probe",
                "```json",
                json.dumps(
                    _safe_json(report.celery_probe),
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        )
        if report.artifacts:
            lines.extend(
                [
                    "",
                    "## Artifacts",
                    "```json",
                    json.dumps(
                        _safe_json(report.artifacts),
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                ]
            )
        return "\n".join(lines) + "\n"


class DeploymentRuntimeHarness:
    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.api = ApiDriver(client)
        self.chat = ChatDriver(client)
        self.mcp = McpDriver()
        self.domain = DomainDriver(client)
        self.webhook = WebhookDriver(client)
        self.db = DbObserver(client)
        self.redis = RedisObserver()
        self.celery = CeleryObserver()
        self.reporter = HarnessReporter()

    def _report_paths(self, scenario_name: str) -> tuple[Path, Path]:
        slug = scenario_name.lower().replace(".", "_").replace(" ", "_")
        suffix = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        json_path = _HARNESS_REPORT_DIR / f"{slug}_{suffix}.json"
        md_path = _HARNESS_REPORT_DIR / f"{slug}_{suffix}.md"
        return json_path, md_path

    def prepare_context(self) -> tuple[ScenarioContext, list[HarnessStep]]:
        steps: list[HarnessStep] = []
        auth_headers, user_id, register_step = self.api.register_test_user()
        steps.append(register_step)

        thread_payload, thread_step = self.api.request(
            name="create_chat_thread",
            method="POST",
            path="/api/v1/chat/new-thread",
            headers=auth_headers,
            json_body={"metadata": {"source": "deployment-runtime-harness"}},
            expected_status=201,
        )
        steps.append(thread_step)
        session_id = str(thread_payload["session_id"])

        strategy_payload, strategy_step = self.api.request(
            name="confirm_strategy",
            method="POST",
            path="/api/v1/strategies/confirm",
            headers=auth_headers,
            json_body={
                "session_id": session_id,
                "dsl_json": _build_deployable_dsl(),
                "auto_start_backtest": False,
                "language": "en",
            },
        )
        steps.append(strategy_step)
        strategy_id = str(strategy_payload["strategy_id"])

        broker_payload, broker_step = self.api.request(
            name="create_builtin_sandbox_broker",
            method="POST",
            path="/api/v1/broker-accounts/builtin-sandbox",
            headers=auth_headers,
            json_body={
                "starting_cash": "25000",
                "fee_bps": "3",
                "metadata": {"source": "deployment-runtime-harness"},
            },
            expected_status=201,
        )
        steps.append(broker_step)
        broker_account_id = str(broker_payload["broker_account_id"])

        preference_payload, preference_step = self.api.request(
            name="set_trading_preference_auto_execute",
            method="PUT",
            path="/api/v1/trading/preferences",
            headers=auth_headers,
            json_body={
                "execution_mode": "auto_execute",
                "approval_scope": "open_and_close",
            },
        )
        steps.append(preference_step)
        assert preference_payload["execution_mode"] == "auto_execute"

        deployment_payload, deployment_step = self.api.request(
            name="create_deployment",
            method="POST",
            path="/api/v1/deployments",
            headers=auth_headers,
            json_body={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": 10000,
                "risk_limits": {"order_qty": 1},
                "runtime_state": {"source": "deployment-runtime-harness"},
            },
            expected_status=201,
        )
        steps.append(deployment_step)
        deployment_id = str(deployment_payload["deployment_id"])

        context = ScenarioContext(
            user_id=user_id,
            auth_headers=auth_headers,
            session_id=session_id,
            strategy_id=strategy_id,
            broker_account_id=broker_account_id,
            deployment_id=deployment_id,
            symbol="BTC/USD",
        )
        return context, steps

    def _find_existing_deployable_strategy_id(
        self,
        *,
        user_id: UUID,
    ) -> str | None:
        db_session_module = _resolve_db_session_module()

        async def _load() -> str | None:
            assert db_session_module.AsyncSessionLocal is not None
            async with db_session_module.AsyncSessionLocal() as db:
                value = await db.scalar(
                    text(
                        "select id::text from strategies "
                        "where user_id = :user_id "
                        "and lower(status) in ('validated', 'backtested', 'deployed') "
                        "order by updated_at desc nulls last, created_at desc "
                        "limit 1"
                    ),
                    {"user_id": user_id},
                )
                if value is None:
                    return None
                return str(value)

        assert self._client.portal is not None
        return self._client.portal.call(_load)

    def prepare_context_for_existing_user(
        self,
        *,
        email: str,
        password: str,
    ) -> tuple[ScenarioContext, list[HarnessStep]]:
        steps: list[HarnessStep] = []
        auth_headers, user_id, login_step = self.api.login_user(
            email=email,
            password=password,
        )
        steps.append(login_step)

        preference_payload, preference_step = self.api.request(
            name="set_trading_preference_auto_execute",
            method="PUT",
            path="/api/v1/trading/preferences",
            headers=auth_headers,
            json_body={
                "execution_mode": "auto_execute",
                "approval_scope": "open_and_close",
            },
        )
        steps.append(preference_step)
        assert preference_payload["execution_mode"] == "auto_execute"

        deployments_payload, deployments_step = self.api.request(
            name="reuse_existing_deployment_list",
            method="GET",
            path="/api/v1/deployments",
            headers=auth_headers,
            expected_status=200,
            note=(
                "Primary path for seeded user: reuse an existing deployment "
                "so quota-limited accounts can still run realtime E2E."
            ),
        )
        steps.append(deployments_step)

        if isinstance(deployments_payload, list):
            deployment_rows = [row for row in deployments_payload if isinstance(row, dict)]
        elif isinstance(deployments_payload, dict):
            raw_rows = deployments_payload.get("deployments")
            deployment_rows = (
                [row for row in raw_rows if isinstance(row, dict)]
                if isinstance(raw_rows, list)
                else []
            )
        else:
            deployment_rows = []

        def _deployment_rank(row: dict[str, Any]) -> tuple[int, int]:
            status_order = {
                "active": 0,
                "paused": 1,
                "pending": 2,
                "stopped": 3,
                "error": 4,
            }
            raw_status = str(row.get("status", "")).strip().lower()
            rank = status_order.get(raw_status, 99)
            market_bonus = (
                0 if str(row.get("market", "")).strip().lower() == "crypto" else 1
            )
            return market_bonus, rank

        candidates = [
            row
            for row in deployment_rows
            if str(row.get("deployment_id", "")).strip()
        ]
        candidates.sort(key=_deployment_rank)
        selected = candidates[0] if candidates else None

        if selected is not None:
            session_id = str(uuid4())
            deployment_id = str(selected["deployment_id"])
            strategy_id = str(selected.get("strategy_id", "")).strip()
            broker_account_id = ""
            run_payload = selected.get("run")
            if isinstance(run_payload, dict):
                broker_account_id = str(run_payload.get("broker_account_id", "")).strip()
            if not strategy_id:
                fallback_strategy_id = self._find_existing_deployable_strategy_id(
                    user_id=user_id
                )
                strategy_id = str(fallback_strategy_id or "")
            context_symbol = "BTCUSD"
            selected_symbols = selected.get("symbols")
            if isinstance(selected_symbols, list) and selected_symbols:
                first_symbol = str(selected_symbols[0]).strip()
                if first_symbol:
                    context_symbol = first_symbol
            steps.append(
                HarnessStep(
                    name="select_existing_deployment_context",
                    driver="api",
                    method="SELECT",
                    target="/api/v1/deployments",
                    duration_ms=0,
                    ok=True,
                    request={"strategy_preference": "crypto-first"},
                    response={
                        "deployment_id": deployment_id,
                        "strategy_id": strategy_id,
                        "broker_account_id": broker_account_id,
                        "symbol": context_symbol,
                        "status": selected.get("status"),
                    },
                    note="Reuses seeded-user deployment (real strategy) for deterministic E2E.",
                )
            )
            context = ScenarioContext(
                user_id=user_id,
                auth_headers=auth_headers,
                session_id=session_id,
                strategy_id=strategy_id,
                broker_account_id=broker_account_id,
                deployment_id=deployment_id,
                symbol=context_symbol,
            )
            return context, steps

        # No reusable deployment: create minimal new context as fallback.
        thread_payload, thread_step = self.api.request(
            name="create_chat_thread",
            method="POST",
            path="/api/v1/chat/new-thread",
            headers=auth_headers,
            json_body={
                "metadata": {
                    "source": "deployment-runtime-harness-seeded-user",
                }
            },
            expected_status=201,
        )
        steps.append(thread_step)
        session_id = str(thread_payload["session_id"])
        strategy_payload, strategy_step = self.api.request(
            name="confirm_strategy",
            method="POST",
            path="/api/v1/strategies/confirm",
            headers=auth_headers,
            json_body={
                "session_id": session_id,
                "dsl_json": _build_deployable_dsl(),
                "auto_start_backtest": False,
                "language": "en",
            },
        )
        steps.append(strategy_step)
        strategy_id = str(strategy_payload["strategy_id"])
        broker_payload, broker_step = self.api.request(
            name="create_builtin_sandbox_broker",
            method="POST",
            path="/api/v1/broker-accounts/builtin-sandbox",
            headers=auth_headers,
            json_body={
                "starting_cash": "25000",
                "fee_bps": "3",
                "metadata": {"source": "deployment-runtime-harness-seeded-user"},
            },
            expected_status=201,
        )
        steps.append(broker_step)
        broker_account_id = str(broker_payload["broker_account_id"])
        deployment_payload, deployment_step = self.api.request(
            name="create_deployment",
            method="POST",
            path="/api/v1/deployments",
            headers=auth_headers,
            json_body={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": "paper",
                "capital_allocated": 10000,
                "risk_limits": {"order_qty": 1},
                "runtime_state": {"source": "deployment-runtime-harness-seeded-user"},
            },
            expected_status=201,
        )
        steps.append(deployment_step)
        deployment_id = str(deployment_payload["deployment_id"])
        context_symbol = "BTC/USD"

        context = ScenarioContext(
            user_id=user_id,
            auth_headers=auth_headers,
            session_id=session_id,
            strategy_id=strategy_id,
            broker_account_id=broker_account_id,
            deployment_id=deployment_id,
            symbol=context_symbol,
        )
        return context, steps

    @staticmethod
    def _extract_access_token(headers: dict[str, str]) -> str:
        auth_value = str(headers.get("Authorization", "")).strip()
        bearer_prefix = "Bearer "
        if auth_value.startswith(bearer_prefix):
            return auth_value.removeprefix(bearer_prefix).strip()
        return auth_value

    @staticmethod
    def _ws_receive_json(
        ws: Any,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                raw = ws.receive_text()
                result_queue.put(("ok", raw))
            except Exception as exc:  # noqa: BLE001
                result_queue.put(("err", exc))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            state, value = result_queue.get(timeout=max(0.1, timeout_seconds))
        except queue.Empty as exc:
            raise AssertionError(
                f"WebSocket receive timed out after {timeout_seconds:.2f}s"
            ) from exc
        if state == "err":
            raise AssertionError(f"WebSocket receive failed: {value!r}")
        raw_text = str(value)
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"WebSocket frame is not JSON: {raw_text[:200]}") from exc
        if not isinstance(payload, dict):
            raise AssertionError(f"WebSocket frame must be JSON object: {payload!r}")
        return payload

    @classmethod
    def _ws_wait_for_event(
        cls,
        ws: Any,
        *,
        event_name: str,
        timeout_seconds: float,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        ignored_events: set[str] | None = None,
        observed: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ignored = set(ignored_events or {"heartbeat", "pong"})
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            slice_timeout = min(2.0, remaining)
            try:
                message = cls._ws_receive_json(ws, timeout_seconds=slice_timeout)
            except AssertionError as exc:
                if "timed out" in str(exc).lower():
                    continue
                raise
            if observed is not None:
                observed.append(message)
            current_event = str(message.get("event", "")).strip()
            if current_event in ignored:
                continue
            if current_event != event_name:
                continue
            if predicate is None or bool(predicate(message)):
                return message
        observed_events = [str(item.get("event")) for item in (observed or [])[-12:]]
        raise AssertionError(
            f"Timed out waiting for WS event={event_name}. recent_events={observed_events}"
        )

    @staticmethod
    def _wait_for_redis_healthy(*, timeout_seconds: float = 60.0) -> None:
        wait_http_ok("http://127.0.0.1:8000/api/v1/health", timeout_seconds=60)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            rows = compose_ps()
            for row in rows:
                if str(row.get("Service", "")).strip() != "redis":
                    continue
                state = str(row.get("State", "")).strip().lower()
                health = str(row.get("Health", "")).strip().lower()
                if state == "running" and health in {"healthy", ""}:
                    return
            time.sleep(1.0)
        raise AssertionError("Redis container did not become healthy in time.")

    @staticmethod
    def _publish_synthetic_bar_pubsub(
        *,
        market: str,
        symbol: str,
        timeframe: str,
        bar: dict[str, Any],
    ) -> dict[str, Any]:
        from packages.domain.market_data.bar_event_pubsub import bar_event_channel

        redis = _resolve_sync_redis_client()()
        payload = {
            "event": "bar_update",
            "market": market.strip().lower(),
            "symbol": symbol.strip().upper(),
            "timeframe": timeframe.strip().lower(),
            "bars": [bar],
            "published_at": datetime.now(UTC).isoformat(),
        }
        redis.publish(
            bar_event_channel(market=market, symbol=symbol),
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )
        return payload

    @staticmethod
    def _extract_start_token(connect_url: str) -> str | None:
        text = str(connect_url).strip()
        marker = "?start="
        if marker not in text:
            return None
        token = text.split(marker, 1)[1].strip()
        return token or None

    @staticmethod
    def _telegram_start_message_payload(
        *,
        start_token: str,
        chat_id: str,
        external_user_id: str,
        username: str,
        update_id: int,
    ) -> dict[str, Any]:
        now_epoch = int(time.time())
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": now_epoch,
                "chat": {
                    "id": int(chat_id),
                    "type": "private",
                    "username": username,
                },
                "from": {
                    "id": int(external_user_id),
                    "is_bot": False,
                    "first_name": "Harness",
                    "username": username,
                    "language_code": "en",
                },
                "text": f"/start {start_token}",
            },
        }

    @staticmethod
    def _telegram_callback_payload(
        *,
        callback_data: str,
        chat_id: str,
        external_user_id: str,
        username: str,
        update_id: int,
    ) -> dict[str, Any]:
        now_epoch = int(time.time())
        return {
            "update_id": update_id,
            "callback_query": {
                "id": f"harness-callback-{update_id}",
                "from": {
                    "id": int(external_user_id),
                    "is_bot": False,
                    "first_name": "Harness",
                    "username": username,
                    "language_code": "en",
                },
                "message": {
                    "message_id": update_id,
                    "date": now_epoch,
                    "chat": {
                        "id": int(chat_id),
                        "type": "private",
                        "username": username,
                    },
                    "text": "Approval request",
                },
                "data": callback_data,
            },
        }

    def _ensure_telegram_binding(
        self,
        *,
        context: ScenarioContext,
        steps: list[HarnessStep],
        artifacts: dict[str, Any],
        chat_id: str,
        external_user_id: str,
        username: str,
    ) -> tuple[bool, str]:
        connect_payload, connect_step = self.api.request(
            name="create_telegram_connect_link",
            method="POST",
            path="/api/v1/social/connectors/telegram/connect-link",
            headers=context.auth_headers,
            json_body={"locale": "en"},
            expected_status={200, 503},
        )
        steps.append(connect_step)
        artifacts["telegram_connect_link"] = connect_payload

        if connect_step.status_code == 200:
            connect_url = str(connect_payload.get("connect_url", "")).strip()
            token = self._extract_start_token(connect_url)
            if token:
                webhook_payload = self._telegram_start_message_payload(
                    start_token=token,
                    chat_id=chat_id,
                    external_user_id=external_user_id,
                    username=username,
                    update_id=900001,
                )
                try:
                    _, webhook_step = self.webhook.telegram_update(
                        payload=webhook_payload,
                    )
                    webhook_step.name = "telegram_webhook_start_link"
                    webhook_step.note = (
                        "Exercises the real Telegram /start binding path when the connector is enabled."
                    )
                    steps.append(webhook_step)
                    binding_summary = self.domain.get_telegram_binding(
                        user_id=context.user_id,
                    )
                    if binding_summary is not None:
                        artifacts["telegram_binding_via_webhook"] = binding_summary
                        artifacts["telegram_binding_path"] = "webhook_start"
                        return True, "webhook_start"
                    artifacts["telegram_binding_path"] = "webhook_start_no_binding"
                except AssertionError as exc:
                    fallback_note = f"webhook_start_failed:{type(exc).__name__}"
                    artifacts["telegram_binding_path"] = fallback_note

        binding_payload, binding_step = self.domain.seed_telegram_binding(
            user_id=context.user_id,
            chat_id=chat_id,
            external_user_id=external_user_id,
            username=username,
            locale="en",
        )
        steps.append(binding_step)
        artifacts["seeded_telegram_binding"] = binding_payload
        artifacts["telegram_binding_path"] = "seeded_binding"
        return False, "seeded_binding"

    def run_v2_portfolio_realtime_resilience_scenario(
        self,
        *,
        email: str,
        password: str,
    ) -> HarnessReport:
        started_at = _utc_now()
        context, steps = self.prepare_context_for_existing_user(
            email=email,
            password=password,
        )
        artifacts: dict[str, Any] = {
            "seeded_user_email": email,
            "broker_mode": "builtin_sandbox",
            "frontend_request_matrix": [
                "GET /deployments",
                "GET /trading/preferences",
                "GET /trade-approvals",
                "GET /deployments/{id}/portfolio",
                "GET /deployments/{id}/orders",
                "GET /deployments/{id}/fills",
                "GET /market-data/bars",
                "GET /stream/deployments/{id}",
                "WS /ws/trading subscribe + subscribe_bars",
                "POST /deployments/{id}/manual-actions (open/close)",
            ],
        }

        db_before = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )

        _, start_step = self.api.request(
            name="start_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/start",
            headers=context.auth_headers,
            expected_status={200, 409},
            note="Matches portfolio card 'Start' action.",
        )
        steps.append(start_step)

        _, list_deployments_step = self.api.request(
            name="portfolio_list_deployments",
            method="GET",
            path="/api/v1/deployments",
            headers=context.auth_headers,
        )
        steps.append(list_deployments_step)

        _, get_preference_step = self.api.request(
            name="portfolio_get_trading_preference",
            method="GET",
            path="/api/v1/trading/preferences",
            headers=context.auth_headers,
        )
        steps.append(get_preference_step)

        _, list_approvals_step = self.api.request(
            name="portfolio_list_trade_approvals",
            method="GET",
            path="/api/v1/trade-approvals",
            headers=context.auth_headers,
            params={
                "deployment_id": context.deployment_id,
                "status": (
                    "pending,approved,rejected,expired,executing,"
                    "executed,failed,cancelled"
                ),
                "limit": 100,
            },
        )
        steps.append(list_approvals_step)

        _, get_portfolio_step = self.api.request(
            name="portfolio_get_portfolio",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/portfolio",
            headers=context.auth_headers,
        )
        steps.append(get_portfolio_step)

        _, get_orders_step = self.api.request(
            name="portfolio_get_orders",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/orders",
            headers=context.auth_headers,
        )
        steps.append(get_orders_step)

        _, get_fills_step = self.api.request(
            name="portfolio_get_fills",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/fills",
            headers=context.auth_headers,
        )
        steps.append(get_fills_step)

        bars_symbol = context.symbol.replace("/", "")
        _, bars_step = self.api.request(
            name="portfolio_get_recent_bars",
            method="GET",
            path="/api/v1/market-data/bars",
            headers=context.auth_headers,
            params={
                "symbol": bars_symbol,
                "market": "crypto",
                "timeframe": "1m",
                "limit": 120,
                "refresh_if_empty": "true",
            },
        )
        steps.append(bars_step)

        access_token = self._extract_access_token(context.auth_headers)
        ws_path = f"/api/v1/ws/trading?token={access_token}"
        observed_ws_events: list[dict[str, Any]] = []
        redis_stopped = False
        initial_mode = "unknown"
        fallback_mode_seen = False
        recovered_mode_seen = False

        with self._client.websocket_connect(ws_path) as ws:
            subscribe_request = {
                "action": "subscribe",
                "deployment_ids": [context.deployment_id],
                "cursors": {},
            }
            ws_subscribe_started = time.perf_counter()
            ws.send_text(json.dumps(subscribe_request, ensure_ascii=True))
            subscribed_msg = self._ws_wait_for_event(
                ws,
                event_name="subscribed",
                timeout_seconds=20.0,
                observed=observed_ws_events,
            )
            ws_subscribe_duration = int((time.perf_counter() - ws_subscribe_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_subscribe_deployment",
                    driver="ws",
                    method="SEND",
                    target=ws_path,
                    duration_ms=ws_subscribe_duration,
                    ok=True,
                    request=subscribe_request,
                    response=subscribed_msg,
                    note="Matches frontend websocket subscribe envelope.",
                )
            )
            initial_mode = str(subscribed_msg.get("transport_mode", "unknown"))
            if initial_mode != "pubsub":
                ws_wait_pubsub_started = time.perf_counter()
                ws_pubsub_mode_msg = self._ws_wait_for_event(
                    ws,
                    event_name="transport_mode",
                    timeout_seconds=45.0,
                    predicate=lambda msg: str(msg.get("mode", "")) == "pubsub",
                    observed=observed_ws_events,
                )
                ws_wait_pubsub_duration = int(
                    (time.perf_counter() - ws_wait_pubsub_started) * 1000
                )
                steps.append(
                    HarnessStep(
                        name="ws_wait_pubsub_before_fault_injection",
                        driver="ws",
                        method="RECV",
                        target=ws_path,
                        duration_ms=ws_wait_pubsub_duration,
                        ok=True,
                        request={"expected_mode": "pubsub"},
                        response=ws_pubsub_mode_msg,
                        note=(
                            "Ensures baseline mode is pubsub before forcing redis outage, "
                            "so fallback transition is meaningful."
                        ),
                    )
                )
                initial_mode = "pubsub"

            subscribe_bars_request = {
                "action": "subscribe_bars",
                "market": "crypto",
                "symbol": bars_symbol,
                "timeframe": "1m",
                "limit": 120,
            }
            ws_bars_started = time.perf_counter()
            ws.send_text(json.dumps(subscribe_bars_request, ensure_ascii=True))
            bar_snapshot_msg = self._ws_wait_for_event(
                ws,
                event_name="bar_snapshot",
                timeout_seconds=30.0,
                observed=observed_ws_events,
            )
            bars_subscribed_msg = self._ws_wait_for_event(
                ws,
                event_name="bars_subscribed",
                timeout_seconds=20.0,
                observed=observed_ws_events,
            )
            ws_bars_duration = int((time.perf_counter() - ws_bars_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_subscribe_bars",
                    driver="ws",
                    method="SEND",
                    target=ws_path,
                    duration_ms=ws_bars_duration,
                    ok=True,
                    request=subscribe_bars_request,
                    response={
                        "bar_snapshot_count": len(bar_snapshot_msg.get("bars", [])),
                        "bars_subscribed": bars_subscribed_msg,
                    },
                    note="Matches frontend chart stream subscribe path.",
                )
            )

            snapshot_bars = bar_snapshot_msg.get("bars", [])
            last_ts = 0
            last_close = 100.0
            if isinstance(snapshot_bars, list):
                for row in snapshot_bars:
                    if not isinstance(row, dict):
                        continue
                    raw_ts = row.get("t")
                    try:
                        ts = int(raw_ts)
                    except (TypeError, ValueError):
                        continue
                    last_ts = max(last_ts, ts)
                    raw_close = row.get("c")
                    try:
                        last_close = float(raw_close)
                    except (TypeError, ValueError):
                        continue
            synthetic_bar = {
                "t": max(last_ts + 60, int(time.time())),
                "o": round(last_close, 6),
                "h": round(last_close * 1.0008, 6),
                "l": round(last_close * 0.9992, 6),
                "c": round(last_close * 1.0001, 6),
                "v": 0.123456,
            }
            synthetic_payload = self._publish_synthetic_bar_pubsub(
                market="crypto",
                symbol=bars_symbol,
                timeframe="1m",
                bar=synthetic_bar,
            )
            ws_pubsub_started = time.perf_counter()
            bar_update_msg = self._ws_wait_for_event(
                ws,
                event_name="bar_update",
                timeout_seconds=20.0,
                predicate=lambda msg: any(
                    isinstance(item, dict) and int(item.get("t", 0)) == synthetic_bar["t"]
                    for item in (msg.get("bars") if isinstance(msg.get("bars"), list) else [])
                ),
                observed=observed_ws_events,
            )
            ws_pubsub_duration = int((time.perf_counter() - ws_pubsub_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_receive_pubsub_bar_update",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_pubsub_duration,
                    ok=True,
                    request={"published_payload": synthetic_payload},
                    response=bar_update_msg,
                    note="Proves active pub/sub channel delivery for chart updates.",
                )
            )

            manual_open_payload = {
                "action": "open",
                "payload": {
                    "symbol": context.symbol,
                    "qty": 1,
                    "side": "long",
                    "mark_price": 100,
                },
            }
            _, manual_open_step = self.api.request(
                name="manual_open_long",
                method="POST",
                path=f"/api/v1/deployments/{context.deployment_id}/manual-actions",
                headers=context.auth_headers,
                json_body=manual_open_payload,
                note="Matches frontend manual open-long action.",
            )
            steps.append(manual_open_step)
            ws_open_started = time.perf_counter()
            ws_open_event = self._ws_wait_for_event(
                ws,
                event_name="manual_action_update",
                timeout_seconds=25.0,
                predicate=lambda msg: str(msg.get("deployment_id", "")) == context.deployment_id,
                observed=observed_ws_events,
            )
            ws_open_duration = int((time.perf_counter() - ws_open_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_receive_manual_open_update",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_open_duration,
                    ok=True,
                    request={"action": "open"},
                    response=ws_open_event,
                )
            )
            open_event_seq = int(ws_open_event.get("event_seq", 0))

            manual_close_payload = {
                "action": "close",
                "payload": {
                    "symbol": context.symbol,
                    "qty": 1,
                    "mark_price": 100,
                },
            }
            _, manual_close_step = self.api.request(
                name="manual_close_position",
                method="POST",
                path=f"/api/v1/deployments/{context.deployment_id}/manual-actions",
                headers=context.auth_headers,
                json_body=manual_close_payload,
                note="Matches frontend manual close action.",
            )
            steps.append(manual_close_step)
            ws_close_started = time.perf_counter()
            ws_close_event = self._ws_wait_for_event(
                ws,
                event_name="manual_action_update",
                timeout_seconds=25.0,
                predicate=lambda msg: int(msg.get("event_seq", 0)) > open_event_seq,
                observed=observed_ws_events,
            )
            ws_close_duration = int((time.perf_counter() - ws_close_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_receive_manual_close_update",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_close_duration,
                    ok=True,
                    request={"action": "close"},
                    response=ws_close_event,
                )
            )
            close_event_seq = int(ws_close_event.get("event_seq", 0))

            sse_started = time.perf_counter()
            sse_response = self._client.get(
                f"/api/v1/stream/deployments/{context.deployment_id}",
                headers=context.auth_headers,
                params={
                    "cursor": max(0, close_event_seq - 1),
                    "poll_seconds": 0.5,
                    "heartbeat_seconds": 1.0,
                    "max_events": 4,
                },
            )
            sse_duration = int((time.perf_counter() - sse_started) * 1000)
            assert sse_response.status_code == 200, sse_response.text[:400]
            sse_frames = _parse_sse_frames(sse_response.text)
            steps.append(
                HarnessStep(
                    name="sse_stream_replay_after_ws",
                    driver="sse",
                    method="GET",
                    target=f"/api/v1/stream/deployments/{context.deployment_id}",
                    duration_ms=sse_duration,
                    ok=bool(sse_frames),
                    request={"cursor": max(0, close_event_seq - 1)},
                    response={
                        "frame_count": len(sse_frames),
                        "frames": sse_frames[:6],
                    },
                    status_code=sse_response.status_code,
                    note="Matches frontend SSE fallback stream path with cursor continuity.",
                )
            )

            stop_redis_started = time.perf_counter()
            run_command(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(COMPOSE_FILE),
                    "stop",
                    "redis",
                ],
                cwd=BACKEND_DIR,
                timeout=180,
            )
            redis_stopped = True
            stop_redis_duration = int((time.perf_counter() - stop_redis_started) * 1000)
            steps.append(
                HarnessStep(
                    name="simulate_pubsub_disconnect_stop_redis",
                    driver="infra",
                    method="docker compose stop",
                    target="redis",
                    duration_ms=stop_redis_duration,
                    ok=True,
                    request={},
                    response={"redis_state": "stopped"},
                    note="Fault injection: force pub/sub read failure on active WS.",
                )
            )

            ws_fallback_started = time.perf_counter()
            fallback_mode_msg = self._ws_wait_for_event(
                ws,
                event_name="transport_mode",
                timeout_seconds=30.0,
                predicate=lambda msg: str(msg.get("mode", "")) == "polling_fallback",
                observed=observed_ws_events,
            )
            ws_fallback_duration = int((time.perf_counter() - ws_fallback_started) * 1000)
            fallback_mode_seen = True
            steps.append(
                HarnessStep(
                    name="ws_switch_to_polling_fallback",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_fallback_duration,
                    ok=True,
                    request={"expected_mode": "polling_fallback"},
                    response=fallback_mode_msg,
                )
            )

            _, manual_open_fallback_step = self.api.request(
                name="manual_open_long_during_fallback",
                method="POST",
                path=f"/api/v1/deployments/{context.deployment_id}/manual-actions",
                headers=context.auth_headers,
                json_body=manual_open_payload,
                note=(
                    "Creates outbox events while redis is down; WS should keep receiving via polling fallback."
                ),
            )
            steps.append(manual_open_fallback_step)

            ws_polling_started = time.perf_counter()
            fallback_manual_msg = self._ws_wait_for_event(
                ws,
                event_name="manual_action_update",
                timeout_seconds=30.0,
                predicate=lambda msg: int(msg.get("event_seq", 0)) > close_event_seq,
                observed=observed_ws_events,
            )
            ws_polling_duration = int((time.perf_counter() - ws_polling_started) * 1000)
            steps.append(
                HarnessStep(
                    name="ws_receive_event_during_polling_fallback",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_polling_duration,
                    ok=True,
                    request={"after_event_seq": close_event_seq},
                    response=fallback_manual_msg,
                    note="Evidence that fallback path continues incremental outbox delivery.",
                )
            )

            start_redis_started = time.perf_counter()
            run_command(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(COMPOSE_FILE),
                    "start",
                    "redis",
                ],
                cwd=BACKEND_DIR,
                timeout=180,
            )
            redis_stopped = False
            self._wait_for_redis_healthy(timeout_seconds=60.0)
            start_redis_duration = int((time.perf_counter() - start_redis_started) * 1000)
            steps.append(
                HarnessStep(
                    name="restore_pubsub_start_redis",
                    driver="infra",
                    method="docker compose start",
                    target="redis",
                    duration_ms=start_redis_duration,
                    ok=True,
                    request={},
                    response={"redis_state": "healthy"},
                )
            )

            ws_recover_started = time.perf_counter()
            recovered_mode_msg = self._ws_wait_for_event(
                ws,
                event_name="transport_mode",
                timeout_seconds=45.0,
                predicate=lambda msg: str(msg.get("mode", "")) == "pubsub",
                observed=observed_ws_events,
            )
            ws_recover_duration = int((time.perf_counter() - ws_recover_started) * 1000)
            recovered_mode_seen = True
            steps.append(
                HarnessStep(
                    name="ws_switch_back_to_pubsub",
                    driver="ws",
                    method="RECV",
                    target=ws_path,
                    duration_ms=ws_recover_duration,
                    ok=True,
                    request={"expected_mode": "pubsub"},
                    response=recovered_mode_msg,
                    note="Auto-recovery from fallback to pub/sub after redis is restored.",
                )
            )

        if redis_stopped:
            run_command(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(COMPOSE_FILE),
                    "start",
                    "redis",
                ],
                cwd=BACKEND_DIR,
                timeout=180,
            )
            self._wait_for_redis_healthy(timeout_seconds=60.0)

        _, stop_step = self.api.request(
            name="stop_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/stop",
            headers=context.auth_headers,
            expected_status={200, 409},
            note="Final cleanup and parity with frontend terminate action.",
        )
        steps.append(stop_step)

        db_after = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )
        redis_probe = self.redis.probe(deployment_id=context.deployment_id)
        celery_probe = self.celery.probe()
        artifacts["ws_initial_mode"] = initial_mode
        artifacts["ws_fallback_mode_seen"] = fallback_mode_seen
        artifacts["ws_recovered_mode_seen"] = recovered_mode_seen
        artifacts["ws_observed_event_count"] = len(observed_ws_events)
        artifacts["ws_observed_events_tail"] = observed_ws_events[-15:]

        json_path, md_path = self._report_paths("deployment_runtime_v2_portfolio_realtime")
        report = HarnessReport(
            scenario_name="deployment_runtime_v2_portfolio_realtime",
            started_at=started_at,
            completed_at=_utc_now(),
            context=context,
            steps=steps,
            db_before=db_before,
            db_after=db_after,
            redis_probe=redis_probe,
            celery_probe=celery_probe,
            artifacts=artifacts,
            report_json_path=json_path,
            report_markdown_path=md_path,
        )
        self.reporter.write(report)
        return report

    def run_v1_rest_scenario(self) -> HarnessReport:
        started_at = _utc_now()
        context, steps = self.prepare_context()
        db_before = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )

        _, start_step = self.api.request(
            name="start_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/start",
            headers=context.auth_headers,
        )
        steps.append(start_step)

        _, portfolio_step = self.api.request(
            name="get_portfolio",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/portfolio",
            headers=context.auth_headers,
            note="Captures synchronous runtime refresh latency on the portfolio poll path.",
        )
        steps.append(portfolio_step)

        _, orders_step = self.api.request(
            name="get_orders",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/orders",
            headers=context.auth_headers,
        )
        steps.append(orders_step)

        _, fills_step = self.api.request(
            name="get_fills",
            method="GET",
            path=f"/api/v1/deployments/{context.deployment_id}/fills",
            headers=context.auth_headers,
        )
        steps.append(fills_step)

        _, manual_step = self.api.request(
            name="manual_open_long",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/manual-actions",
            headers=context.auth_headers,
            json_body={
                "action": "open",
                "payload": {
                    "symbol": context.symbol,
                    "qty": 1,
                    "side": "long",
                },
            },
            note="Covers UI-equivalent open long action after deployment start.",
        )
        steps.append(manual_step)

        _, pause_step = self.api.request(
            name="pause_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/pause",
            headers=context.auth_headers,
        )
        steps.append(pause_step)

        _, stop_step = self.api.request(
            name="stop_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/stop",
            headers=context.auth_headers,
        )
        steps.append(stop_step)

        db_after = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )
        redis_probe = self.redis.probe(deployment_id=context.deployment_id)
        celery_probe = self.celery.probe()
        json_path, md_path = self._report_paths("deployment_runtime_v1")
        report = HarnessReport(
            scenario_name="deployment_runtime_v1",
            started_at=started_at,
            completed_at=_utc_now(),
            context=context,
            steps=steps,
            db_before=db_before,
            db_after=db_after,
            redis_probe=redis_probe,
            celery_probe=celery_probe,
            artifacts={},
            report_json_path=json_path,
            report_markdown_path=md_path,
        )
        self.reporter.write(report)
        return report

    def run_v1_1_chat_mcp_scenario(self) -> HarnessReport:
        started_at = _utc_now()
        context, steps = self.prepare_context()
        db_before = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )

        chat_summary, chat_step = self.chat.send_message_stream(
            headers=context.auth_headers,
            session_id=context.session_id,
            language="en",
            message=(
                "In one short sentence, confirm you can continue with deployment "
                "setup. Do not call any tools."
            ),
        )
        steps.append(chat_step)
        assert int(chat_summary["event_count"]) > 0

        _, list_step = self.mcp.call_trading_tool(
            tool_name="trading_list_deployments",
            user_id=context.user_id,
            session_id=context.session_id,
        )
        steps.append(list_step)

        _, start_step = self.mcp.call_trading_tool(
            tool_name="trading_start_deployment",
            user_id=context.user_id,
            session_id=context.session_id,
            arguments={"deployment_id": context.deployment_id},
        )
        steps.append(start_step)

        _, orders_step = self.mcp.call_trading_tool(
            tool_name="trading_get_orders",
            user_id=context.user_id,
            session_id=context.session_id,
            arguments={"deployment_id": context.deployment_id},
        )
        steps.append(orders_step)

        _, pause_step = self.mcp.call_trading_tool(
            tool_name="trading_pause_deployment",
            user_id=context.user_id,
            session_id=context.session_id,
            arguments={"deployment_id": context.deployment_id},
        )
        steps.append(pause_step)

        _, stop_step = self.mcp.call_trading_tool(
            tool_name="trading_stop_deployment",
            user_id=context.user_id,
            session_id=context.session_id,
            arguments={"deployment_id": context.deployment_id},
        )
        steps.append(stop_step)

        db_after = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )
        redis_probe = self.redis.probe(deployment_id=context.deployment_id)
        celery_probe = self.celery.probe()
        json_path, md_path = self._report_paths("deployment_runtime_v1_1")
        report = HarnessReport(
            scenario_name="deployment_runtime_v1_1",
            started_at=started_at,
            completed_at=_utc_now(),
            context=context,
            steps=steps,
            db_before=db_before,
            db_after=db_after,
            redis_probe=redis_probe,
            celery_probe=celery_probe,
            artifacts={},
            report_json_path=json_path,
            report_markdown_path=md_path,
        )
        self.reporter.write(report)
        return report

    def run_v1_2_approval_notification_scenario(self) -> HarnessReport:
        started_at = _utc_now()
        context, steps = self.prepare_context()
        artifacts: dict[str, Any] = {}

        _, notification_pref_step = self.api.request(
            name="set_notification_preferences",
            method="PUT",
            path="/api/v1/notifications/preferences",
            headers=context.auth_headers,
            json_body={
                "telegram_enabled": True,
                "deployment_started_enabled": True,
                "position_opened_enabled": True,
            },
        )
        steps.append(notification_pref_step)

        _, preference_step = self.api.request(
            name="set_trading_preference_approval_required",
            method="PUT",
            path="/api/v1/trading/preferences",
            headers=context.auth_headers,
            json_body={
                "execution_mode": "approval_required",
                "approval_channel": "telegram",
                "approval_timeout_seconds": 180,
                "approval_scope": "open_only",
            },
        )
        steps.append(preference_step)

        _, start_step = self.api.request(
            name="start_deployment",
            method="POST",
            path=f"/api/v1/deployments/{context.deployment_id}/start",
            headers=context.auth_headers,
        )
        steps.append(start_step)

        db_before = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )

        telegram_identity_seed = str(int(uuid4().hex[:12], 16))
        chat_id = telegram_identity_seed
        external_user_id = telegram_identity_seed
        username = f"harness_{uuid4().hex[:8]}"
        telegram_webhook_used, binding_path = self._ensure_telegram_binding(
            context=context,
            steps=steps,
            artifacts=artifacts,
            chat_id=chat_id,
            external_user_id=external_user_id,
            username=username,
        )
        artifacts["telegram_webhook_used"] = telegram_webhook_used

        approval_seed_payload, approval_seed_step = self.domain.seed_trade_approval_request(
            user_id=context.user_id,
            deployment_id=context.deployment_id,
            symbol=context.symbol,
            side="long",
            qty="1",
            mark_price="100",
            approval_channel="telegram",
            timeout_seconds=180,
        )
        steps.append(approval_seed_step)
        artifacts["seeded_trade_approval"] = approval_seed_payload

        approval_list_payload, approval_list_step = self.api.request(
            name="list_trade_approvals",
            method="GET",
            path="/api/v1/trade-approvals",
            headers=context.auth_headers,
            params={
                "deployment_id": context.deployment_id,
                "status": "pending,approved,rejected,expired,executing,executed,failed,cancelled",
                "limit": 50,
            },
        )
        steps.append(approval_list_step)
        artifacts["approval_list_count"] = len(approval_list_payload)

        request_id = str(approval_seed_payload["request_id"])
        if binding_path == "webhook_start":
            callback_payload = self._telegram_callback_payload(
                callback_data=str(approval_seed_payload["approve_callback_data"]),
                chat_id=chat_id,
                external_user_id=external_user_id,
                username=username,
                update_id=900002,
            )
            _, callback_step = self.webhook.telegram_update(payload=callback_payload)
            callback_step.name = "telegram_webhook_approval_callback"
            callback_step.note = "Exercises Telegram callback approval flow."
            steps.append(callback_step)
            approval_after_callback = self.domain.get_trade_approval(
                request_id=request_id,
            )
            artifacts["approval_after_callback"] = approval_after_callback
            if (
                approval_after_callback is not None
                and approval_after_callback.get("status") != "pending"
            ):
                artifacts["approval_decision_path"] = "telegram_callback"
            else:
                approve_payload, approve_step = self.api.request(
                    name="approve_trade_approval_fallback",
                    method="POST",
                    path=f"/api/v1/trade-approvals/{request_id}/approve",
                    headers=context.auth_headers,
                    json_body={"note": "deployment-runtime-harness-fallback"},
                )
                steps.append(approve_step)
                artifacts["approval_decision_path"] = "telegram_callback_then_api_fallback"
                artifacts["approve_response"] = approve_payload
        else:
            approve_payload, approve_step = self.api.request(
                name="approve_trade_approval",
                method="POST",
                path=f"/api/v1/trade-approvals/{request_id}/approve",
                headers=context.auth_headers,
                json_body={"note": "deployment-runtime-harness"},
            )
            steps.append(approve_step)
            artifacts["approval_decision_path"] = "api_approve"
            artifacts["approve_response"] = approve_payload

        dispatch_payload, dispatch_step = self.domain.run_notification_dispatch(limit=10)
        steps.append(dispatch_step)
        artifacts["notification_dispatch_result"] = dispatch_payload

        execute_payload, execute_step = self.domain.run_execute_approved_open(
            request_id=request_id,
        )
        steps.append(execute_step)
        artifacts["approval_execution_result"] = execute_payload

        connectors_payload, connectors_step = self.api.request(
            name="list_social_connectors",
            method="GET",
            path="/api/v1/social/connectors",
            headers=context.auth_headers,
        )
        steps.append(connectors_step)
        artifacts["social_connectors"] = connectors_payload

        activities_payload, activities_step = self.api.request(
            name="list_telegram_activities",
            method="GET",
            path="/api/v1/social/connectors/telegram/activities",
            headers=context.auth_headers,
            params={"limit": 20},
        )
        steps.append(activities_step)
        artifacts["telegram_activities_count"] = len(
            activities_payload.get("items", [])
        )

        db_after = self.db.capture(
            deployment_id=context.deployment_id,
            user_id=context.user_id,
        )
        redis_probe = self.redis.probe(deployment_id=context.deployment_id)
        celery_probe = self.celery.probe()
        json_path, md_path = self._report_paths("deployment_runtime_v1_2")
        report = HarnessReport(
            scenario_name="deployment_runtime_v1_2",
            started_at=started_at,
            completed_at=_utc_now(),
            context=context,
            steps=steps,
            db_before=db_before,
            db_after=db_after,
            redis_probe=redis_probe,
            celery_probe=celery_probe,
            artifacts=artifacts,
            report_json_path=json_path,
            report_markdown_path=md_path,
        )
        self.reporter.write(report)
        return report
