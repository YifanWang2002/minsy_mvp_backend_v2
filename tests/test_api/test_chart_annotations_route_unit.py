"""Unit coverage for canonical chart annotation routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from apps.api.main import create_app
from apps.api.routes import chart_annotations
from packages.domain.chart_annotations.service import ChartAnnotationConflictError


@asynccontextmanager
async def _noop_lifespan(_):
    yield


@pytest.fixture()
def app(monkeypatch):
    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr("apps.api.main.lifespan", _noop_lifespan)
        test_app = create_app()
        yield test_app


@pytest.fixture()
def client(app):
    return TestClient(app)


def _override_dependencies(*, app, user, db):
    async def _override_user():
        return user

    async def _override_db():
        yield db

    app.dependency_overrides[chart_annotations.get_current_user] = _override_user
    app.dependency_overrides[chart_annotations.get_db] = _override_db


def test_get_chart_annotations_merges_snapshot_and_projected_rows(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    snapshot_mock = AsyncMock(
        return_value={
            "annotations": [{"id": "ann-1", "semantic": {"kind": "note"}}],
            "cursor": 17,
        }
    )
    execution_mock = AsyncMock(return_value=[{"id": "exec-1"}])
    backtest_mock = AsyncMock(return_value=[{"id": "bt-1"}])
    monkeypatch.setattr(
        chart_annotations,
        "append_chart_annotation_snapshot",
        snapshot_mock,
    )
    monkeypatch.setattr(
        chart_annotations,
        "project_execution_annotations_for_scope",
        execution_mock,
    )
    monkeypatch.setattr(
        chart_annotations,
        "project_backtest_annotations_for_scope",
        backtest_mock,
    )

    response = client.get(
        "/api/v1/chart-annotations",
        params={"market": "stocks", "symbol": "AAPL", "timeframe": "1m"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cursor"] == 17
    assert [item["id"] for item in payload["annotations"]] == [
        "ann-1",
        "exec-1",
        "bt-1",
    ]
    snapshot_mock.assert_awaited_once()
    execution_mock.assert_awaited_once()
    backtest_mock.assert_awaited_once()


def test_update_chart_annotation_returns_409_for_version_conflict(
    client,
    app,
    monkeypatch,
):
    annotation_id = uuid4()
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    latest = {"id": str(annotation_id), "version": 3}
    monkeypatch.setattr(
        chart_annotations,
        "update_chart_annotation",
        AsyncMock(side_effect=ChartAnnotationConflictError(latest=latest)),
    )

    response = client.patch(
        f"/api/v1/chart-annotations/{annotation_id}",
        json={
            "base_version": 2,
            "annotation": {
                "id": str(annotation_id),
                "source": {"type": "user_manual"},
                "scope": {
                    "market": "stocks",
                    "symbol": "AAPL",
                    "timeframe": "1m",
                },
                "semantic": {"kind": "note", "role": "markup"},
                "tool": {"family": "text", "vendor_type": "text"},
            },
        },
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_VERSION_CONFLICT"
    assert payload["detail"]["latest"] == latest


def test_create_chart_annotation_returns_422_for_invalid_payload(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    monkeypatch.setattr(
        chart_annotations,
        "create_chart_annotation",
        AsyncMock(side_effect=ValueError("invalid fib vendor type")),
    )

    response = client.post(
        "/api/v1/chart-annotations",
        json={
            "annotation": {
                "source": {"type": "user_manual"},
                "scope": {
                    "market": "crypto",
                    "symbol": "BTCUSD",
                    "timeframe": "15m",
                },
                "semantic": {"kind": "note", "role": "markup"},
                "tool": {"family": "fib", "vendor_type": "fib_retracement"},
            },
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_INVALID_PAYLOAD"
    assert payload["detail"]["message"] == "invalid fib vendor type"


def test_update_chart_annotation_returns_422_for_invalid_payload(
    client,
    app,
    monkeypatch,
):
    annotation_id = uuid4()
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    monkeypatch.setattr(
        chart_annotations,
        "update_chart_annotation",
        AsyncMock(side_effect=ValueError("vendor_native.state must be object")),
    )

    response = client.patch(
        f"/api/v1/chart-annotations/{annotation_id}",
        json={
            "base_version": 1,
            "annotation": {
                "id": str(annotation_id),
                "source": {"type": "user_manual"},
                "scope": {
                    "market": "crypto",
                    "symbol": "BTCUSD",
                    "timeframe": "15m",
                },
                "semantic": {"kind": "note", "role": "markup"},
                "tool": {"family": "gann", "vendor_type": "gannbox"},
            },
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_INVALID_PAYLOAD"
    assert payload["detail"]["message"] == "vendor_native.state must be object"


def test_batch_upsert_chart_annotations_returns_422_for_invalid_payload(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    monkeypatch.setattr(
        chart_annotations,
        "batch_upsert_chart_annotations",
        AsyncMock(side_effect=ValueError("unsupported gann vendor type")),
    )

    response = client.post(
        "/api/v1/chart-annotations/batch-upsert",
        json={
            "annotations": [
                {
                    "source": {"type": "system"},
                    "scope": {
                        "market": "crypto",
                        "symbol": "BTCUSD",
                        "timeframe": "15m",
                    },
                    "semantic": {"kind": "note", "role": "markup"},
                    "tool": {"family": "gann", "vendor_type": "gannbox"},
                }
            ]
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_INVALID_PAYLOAD"
    assert payload["detail"]["message"] == "unsupported gann vendor type"


def test_create_chart_annotation_returns_422_for_invalid_profile_payload(
    client,
    app,
    monkeypatch,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    monkeypatch.setattr(
        chart_annotations,
        "create_chart_annotation",
        AsyncMock(
            side_effect=ValueError(
                "tool.vendor_type 'fib_retracement' is not supported for family 'profile'"
            )
        ),
    )

    response = client.post(
        "/api/v1/chart-annotations",
        json={
            "annotation": {
                "source": {"type": "user_manual"},
                "scope": {
                    "market": "crypto",
                    "symbol": "BTCUSD",
                    "timeframe": "15m",
                },
                "semantic": {"kind": "profile", "role": "markup"},
                "tool": {"family": "profile", "vendor_type": "anchored_vwap"},
            },
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_INVALID_PAYLOAD"
    assert "family 'profile'" in payload["detail"]["message"]


@pytest.mark.parametrize(
    "family,error_message",
    [
        (
            "pattern",
            "tool.vendor_type 'pitchfork' is not supported for family 'pattern'",
        ),
        (
            "fork",
            "tool.vendor_type 'anchored_vwap' is not supported for family 'fork'",
        ),
        (
            "measurement",
            "tool.vendor_type 'pitchfork' is not supported for family 'measurement'",
        ),
        (
            "cycle",
            "tool.vendor_type 'price_range' is not supported for family 'cycle'",
        ),
        (
            "forecast",
            "tool.vendor_type 'pitchfork' is not supported for family 'forecast'",
        ),
    ],
)
def test_create_chart_annotation_returns_422_for_invalid_additional_line_tool_payload(
    client,
    app,
    monkeypatch,
    family,
    error_message,
):
    _override_dependencies(
        app=app,
        user=SimpleNamespace(id=uuid4()),
        db=SimpleNamespace(),
    )
    monkeypatch.setattr(
        chart_annotations,
        "create_chart_annotation",
        AsyncMock(side_effect=ValueError(error_message)),
    )

    response = client.post(
        "/api/v1/chart-annotations",
        json={
            "annotation": {
                "source": {"type": "user_manual"},
                "scope": {
                    "market": "crypto",
                    "symbol": "BTCUSD",
                    "timeframe": "15m",
                },
                "semantic": {"kind": family, "role": "markup"},
                "tool": {"family": family, "vendor_type": family},
            },
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"]["code"] == "CHART_ANNOTATION_INVALID_PAYLOAD"
    assert f"family '{family}'" in payload["detail"]["message"]
