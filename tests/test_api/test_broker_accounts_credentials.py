from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.engine.execution.alpaca_account_probe import AlpacaAccountProbeResult
from src.main import app
from src.models import database as db_module
from src.models.broker_account_audit_log import BrokerAccountAuditLog


def _register_and_get_token(client: TestClient) -> str:
    email = f"broker_credentials_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Broker Credential User"},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def test_broker_account_credentials_rotation_and_audit_log(
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

        created = client.post(
            "/api/v1/broker-accounts",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "first-key", "api_secret": "first-secret"},
                "metadata": {"label": "rotation"},
            },
        )
        assert created.status_code == 201
        body = created.json()
        broker_account_id = body["broker_account_id"]
        first_fingerprint = body["key_fingerprint"]
        assert body["encryption_version"] == "fernet_v1"
        assert body["updated_source"] == "api"
        assert "credentials" not in body

        listed = client.get("/api/v1/broker-accounts", headers=headers)
        assert listed.status_code == 200
        assert listed.json()[0]["key_fingerprint"] == first_fingerprint
        assert "credentials" not in listed.json()[0]

        rotated = client.patch(
            f"/api/v1/broker-accounts/{broker_account_id}/credentials?validate=false",
            headers=headers,
            json={"credentials": {"api_key": "second-key", "api_secret": "second-secret"}},
        )
        assert rotated.status_code == 200
        rotated_body = rotated.json()
        assert rotated_body["key_fingerprint"] != first_fingerprint
        assert rotated_body["last_validated_status"] == "validation_skipped"

        validated = client.post(f"/api/v1/broker-accounts/{broker_account_id}/validate", headers=headers)
        assert validated.status_code == 200
        assert validated.json()["last_validated_status"] == "paper_probe_ok"

        deactivated = client.post(
            f"/api/v1/broker-accounts/{broker_account_id}/deactivate",
            headers=headers,
        )
        assert deactivated.status_code == 200
        assert deactivated.json()["status"] == "inactive"

        async def _load_audit_actions() -> list[str]:
            assert db_module.AsyncSessionLocal is not None
            async with db_module.AsyncSessionLocal() as db:
                rows = (
                    await db.scalars(
                        select(BrokerAccountAuditLog.action)
                        .where(BrokerAccountAuditLog.broker_account_id == UUID(broker_account_id))
                        .order_by(BrokerAccountAuditLog.created_at.asc()),
                    )
                ).all()
            return list(rows)

        actions = client.portal.call(_load_audit_actions)
        assert actions == ["create", "update", "validate", "deactivate"]


def test_rotate_credentials_rejects_invalid_probe_result(
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
        created = client.post(
            "/api/v1/broker-accounts?validate=false",
            headers=headers,
            json={
                "provider": "alpaca",
                "mode": "paper",
                "credentials": {"api_key": "stable-key", "api_secret": "stable-secret"},
                "metadata": {},
            },
        )
        assert created.status_code == 201
        broker_account_id = created.json()["broker_account_id"]
        previous_fingerprint = created.json()["key_fingerprint"]

        async def _failed_probe(_: dict[str, str]) -> AlpacaAccountProbeResult:
            return AlpacaAccountProbeResult(
                ok=False,
                status="paper_probe_failed",
                message="Invalid Alpaca credentials or wrong endpoint.",
                metadata={"paper_http_status": 401, "live_http_status": 401},
            )

        monkeypatch.setattr("src.api.routers.broker_accounts._probe_alpaca_credentials", _failed_probe)
        rotated = client.patch(
            f"/api/v1/broker-accounts/{broker_account_id}/credentials",
            headers=headers,
            json={"credentials": {"api_key": "bad-key", "api_secret": "bad-secret"}},
        )
        assert rotated.status_code == 422

        detail = client.get(f"/api/v1/broker-accounts/{broker_account_id}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["key_fingerprint"] == previous_fingerprint
