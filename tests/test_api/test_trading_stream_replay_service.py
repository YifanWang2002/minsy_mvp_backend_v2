"""Tests for shared trading stream replay service helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from apps.api.services.trading_stream_replay_service import (
    load_owned_deployment,
    poll_outbox_events,
    resolve_replay_cursor,
)


@pytest.mark.asyncio
async def test_load_owned_deployment_raises_404_when_missing() -> None:
    db = AsyncMock()
    db.scalar = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await load_owned_deployment(db, deployment_id="dep-1", user_id="user-1")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_resolve_replay_cursor_prefers_requested_cursor() -> None:
    db = AsyncMock()
    db.scalar = AsyncMock(return_value=999)

    resolved = await resolve_replay_cursor(
        db,
        deployment_id="dep-1",
        requested_cursor=123,
    )

    assert resolved.deployment_id == "dep-1"
    assert resolved.cursor == 123
    db.scalar.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_replay_cursor_falls_back_to_latest_outbox_seq() -> None:
    db = AsyncMock()
    db.scalar = AsyncMock(return_value="456")

    resolved = await resolve_replay_cursor(
        db,
        deployment_id="dep-1",
        requested_cursor=None,
    )

    assert resolved.cursor == 456
    db.scalar.assert_awaited()


@pytest.mark.asyncio
async def test_poll_outbox_events_returns_ordered_rows_list() -> None:
    db = AsyncMock()
    rows = [
        SimpleNamespace(event_seq=11, event_type="order_update", payload={"x": 1}),
        SimpleNamespace(event_seq=12, event_type="fill_update", payload={"x": 2}),
    ]
    scalar_result = MagicMock()
    scalar_result.all.return_value = rows
    db.scalars = AsyncMock(return_value=scalar_result)

    result = await poll_outbox_events(
        db,
        deployment_id="dep-1",
        cursor=10,
        limit=300,
    )

    assert result == rows
    db.scalars.assert_awaited()
