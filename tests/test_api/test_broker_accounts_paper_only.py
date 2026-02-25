from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.engine.execution.alpaca_account_probe import AlpacaAccountProbeResult
from src.main import app


def _register_and_get_token(client: TestClient) -> str:
    email = f"broker_paper_only_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Broker Tester"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def test_create_broker_account_rejects_live_mode() -> None:
    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/api/v1/broker-accounts",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "live",
                "credentials": {"api_key": "demo", "api_secret": "demo"},
                "metadata": {},
            },
        )
    assert response.status_code == 422


def test_create_broker_account_validates_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok_probe(_: dict[str, str]) -> AlpacaAccountProbeResult:
        return AlpacaAccountProbeResult(
            ok=True,
            status="paper_probe_ok",
            message="ok",
            metadata={"paper_http_status": 200, "live_http_status": 401},
        )

    monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _ok_probe)

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/api/v1/broker-accounts",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "demo", "api_secret": "demo"},
                "metadata": {"label": "primary"},
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["last_validated_status"] == "paper_probe_ok"
    assert payload["validation_metadata"]["paper_http_status"] == 200
    assert payload["validation_metadata"]["live_http_status"] == 401
    assert payload["last_validated_at"] is not None


def test_create_broker_account_validate_false_skips_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _should_not_call(_: dict[str, str]) -> AlpacaAccountProbeResult:
        raise AssertionError("Probe should not be called when validate=false")

    monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _should_not_call)

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/api/v1/broker-accounts?validate=false",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "demo", "api_secret": "demo"},
                "metadata": {},
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["last_validated_status"] == "validation_skipped"
    assert payload["validation_metadata"]["validate_requested"] is False
    assert payload["last_validated_at"] is None


def test_validate_endpoint_updates_status_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _ok_probe(_: dict[str, str]) -> AlpacaAccountProbeResult:
        return AlpacaAccountProbeResult(
            ok=True,
            status="paper_probe_ok",
            message="ok",
            metadata={"paper_http_status": 200, "live_http_status": 401},
        )

    monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _ok_probe)

    with TestClient(app) as client:
        token = _register_and_get_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        create = client.post(
            "/api/v1/broker-accounts?validate=false",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "demo", "api_secret": "demo"},
                "metadata": {},
            },
        )
        assert create.status_code == 201
        broker_account_id = create.json()["broker_account_id"]

        async def _failed_probe(_: dict[str, str]) -> AlpacaAccountProbeResult:
            return AlpacaAccountProbeResult(
                ok=False,
                status="paper_probe_failed",
                message="Invalid Alpaca credentials or wrong endpoint.",
                metadata={"paper_http_status": 401, "live_http_status": 401},
            )

        monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _failed_probe)
        validate_resp = client.post(f"/api/v1/broker-accounts/{broker_account_id}/validate", headers=headers)
        assert validate_resp.status_code == 422

        detail = client.get(f"/api/v1/broker-accounts/{broker_account_id}", headers=headers)
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["status"] == "error"
        assert payload["last_validated_status"] == "paper_probe_failed"
        assert payload["validation_metadata"]["paper_http_status"] == 401
