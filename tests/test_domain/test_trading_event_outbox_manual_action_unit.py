"""Unit tests for manual_action_update trading outbox payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from packages.domain.trading.services import trading_event_outbox_service
from packages.domain.trading.services.trading_event_outbox_service import (
    TradingEventSnapshot,
)


class _FakeScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return list(self._rows)


class _FakeDb:
    def __init__(self, *, latest_rows: list[object] | None = None) -> None:
        self.latest_rows = latest_rows or []
        self.added: list[object] = []
        self.commit_count = 0

    async def scalars(self, _stmt):
        return _FakeScalarResult(self.latest_rows)

    async def scalar(self, _stmt):
        return None

    def add(self, row: object) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.commit_count += 1


def _snapshot_payload(deployment_id: str) -> dict[str, object]:
    return {
        "deployment_id": deployment_id,
        "status": "active",
        "run": None,
        "pnl": {
            "equity": 1000.0,
            "cash": 900.0,
            "margin_used": 100.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "snapshot_time": datetime.now(UTC).isoformat(),
        },
        "pnl_source": "platform_estimate",
        "broker_account": None,
        "positions": [],
        "orders": [],
        "fills": [],
        "approvals": [],
        "manual_actions": [
            {
                "manual_trade_action_id": "ma-1",
                "deployment_id": deployment_id,
                "action": "open",
                "status": "executing",
                "payload": {
                    "_execution": {
                        "status": "executing",
                        "reason": "queued",
                    }
                },
                "created_at": "2026-03-06T10:00:00+00:00",
                "updated_at": "2026-03-06T10:00:10+00:00",
            }
        ],
    }


@pytest.mark.asyncio
async def test_append_snapshot_emits_manual_action_update(monkeypatch) -> None:
    deployment_id = uuid4()
    deployment = SimpleNamespace(id=deployment_id, updated_at=datetime.now(UTC))
    db = _FakeDb()

    async def _fake_build_snapshot(_db, *, deployment):
        return TradingEventSnapshot(
            payload=_snapshot_payload(str(deployment.id)),
            pnl_snapshot=SimpleNamespace(),
            pnl_source="platform_estimate",
            broker_account=None,
        )

    monkeypatch.setattr(
        trading_event_outbox_service,
        "build_trading_event_snapshot",
        _fake_build_snapshot,
    )

    result = await trading_event_outbox_service.append_trading_event_snapshot(
        db,
        deployment=deployment,
    )

    assert result["status"] == "ok"
    manual_rows = [
        row
        for row in db.added
        if getattr(row, "event_type", None) == "manual_action_update"
    ]
    assert len(manual_rows) == 1
    payload = manual_rows[0].payload
    assert payload["deployment_id"] == str(deployment_id)
    assert payload["latest_manual_action_id"] == "ma-1"
    assert isinstance(payload["manual_actions"], list)
    assert db.commit_count == 1


@pytest.mark.asyncio
async def test_append_snapshot_dedupes_manual_action_update(monkeypatch) -> None:
    deployment_id = uuid4()
    deployment = SimpleNamespace(id=deployment_id, updated_at=datetime.now(UTC))
    payload = _snapshot_payload(str(deployment_id))
    manual_event_payload = {
        "deployment_id": str(deployment_id),
        "manual_actions": payload["manual_actions"],
        "latest_manual_action_id": "ma-1",
        "updated_at": "2026-03-06T10:00:10+00:00",
    }
    latest_row = SimpleNamespace(
        event_type="manual_action_update",
        payload=manual_event_payload,
    )
    db = _FakeDb(latest_rows=[latest_row])

    async def _fake_build_snapshot(_db, *, deployment):
        return TradingEventSnapshot(
            payload=payload,
            pnl_snapshot=SimpleNamespace(),
            pnl_source="platform_estimate",
            broker_account=None,
        )

    monkeypatch.setattr(
        trading_event_outbox_service,
        "build_trading_event_snapshot",
        _fake_build_snapshot,
    )
    monkeypatch.setattr(
        trading_event_outbox_service,
        "_EVENT_TYPES",
        ("manual_action_update",),
    )

    result = await trading_event_outbox_service.append_trading_event_snapshot(
        db,
        deployment=deployment,
    )

    assert result["inserted"] == 0
    assert db.added == []
    assert db.commit_count == 0
