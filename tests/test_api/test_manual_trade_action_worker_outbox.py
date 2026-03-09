"""Worker tests ensuring manual action state transitions emit outbox snapshots."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.worker.io.tasks import manual_trade_action as worker_module


class _FakeSession:
    def __init__(self, action: SimpleNamespace) -> None:
        self._action = action
        self.commit_count = 0
        self.refresh_count = 0

    async def scalar(self, _stmt):
        return self._action

    async def commit(self) -> None:
        self.commit_count += 1

    async def refresh(self, _obj) -> None:
        self.refresh_count += 1


class _FakeSessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self) -> _FakeSessionContext:
        return _FakeSessionContext(self._session)


@pytest.mark.asyncio
async def test_worker_exception_path_emits_outbox_snapshot(monkeypatch) -> None:
    action = SimpleNamespace(
        id=uuid4(),
        deployment_id=uuid4(),
        status="pending",
        payload={},
    )
    fake_session = _FakeSession(action)
    append_calls: list[tuple[object, object]] = []

    async def _fake_close_postgres() -> None:
        return None

    async def _fake_init_postgres(*, ensure_schema: bool) -> None:
        assert ensure_schema is False

    async def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("boom")

    async def _fake_append_snapshot(db_session, *, deployment_id):
        append_calls.append((db_session, deployment_id))

    monkeypatch.setattr(worker_module.db_module, "close_postgres", _fake_close_postgres)
    monkeypatch.setattr(worker_module.db_module, "init_postgres", _fake_init_postgres)
    monkeypatch.setattr(
        worker_module.db_module,
        "AsyncSessionLocal",
        _FakeSessionFactory(fake_session),
    )
    monkeypatch.setattr(
        worker_module,
        "execute_manual_trade_action",
        _raise_runtime_error,
    )
    monkeypatch.setattr(
        worker_module,
        "append_trading_event_snapshot",
        _fake_append_snapshot,
    )

    result = await worker_module._run_execute_manual_trade_action(action.id)

    assert result["status"] == "failed"
    assert result["deployment_id"] == str(action.deployment_id)
    assert len(append_calls) >= 2
    assert all(call[0] is fake_session for call in append_calls)
    assert all(call[1] == action.deployment_id for call in append_calls)
