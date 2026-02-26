from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_EXAMPLE_STRATEGY_PATH = (
    _BACKEND_DIR / "packages" / "domain" / "strategy" / "assets" / "example_strategy.json"
)


def _load_example_dsl() -> dict[str, object]:
    return json.loads(_EXAMPLE_STRATEGY_PATH.read_text(encoding="utf-8"))


def _create_thread(api_test_client: TestClient, auth_headers: dict[str, str]) -> str:
    response = api_test_client.post(
        "/api/v1/chat/new-thread",
        headers=auth_headers,
        json={"metadata": {"source": "pytest-strategy-confirm-live"}},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _confirm_strategy(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    *,
    session_id: str,
    dsl_json: dict[str, object],
    strategy_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": session_id,
        "dsl_json": dsl_json,
        "auto_start_backtest": False,
        "language": "en",
    }
    if strategy_id is not None:
        payload["strategy_id"] = strategy_id

    response = api_test_client.post(
        "/api/v1/strategies/confirm",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_000_accessibility_strategy_confirm_persists_dsl(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)
    dsl = _load_example_dsl()
    strategy_name = f"Pytest Confirm {uuid4().hex[:8]}"
    dsl["strategy"]["name"] = strategy_name  # type: ignore[index]

    confirmed = _confirm_strategy(
        api_test_client,
        auth_headers,
        session_id=session_id,
        dsl_json=dsl,
    )

    strategy_id = str(confirmed["strategy_id"])
    assert str(confirmed["session_id"]) == session_id
    assert confirmed["phase"] in {"strategy", "stress_test"}

    detail = api_test_client.get(
        f"/api/v1/strategies/{strategy_id}",
        headers=auth_headers,
    )
    assert detail.status_code == 200, detail.text
    detail_payload = detail.json()
    assert str(detail_payload["strategy_id"]) == strategy_id
    assert detail_payload["dsl_json"]["strategy"]["name"] == strategy_name


def test_010_strategy_confirm_twice_creates_version_diff(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    session_id = _create_thread(api_test_client, auth_headers)
    dsl_v1 = _load_example_dsl()
    dsl_v1["strategy"]["name"] = f"Pytest Version {uuid4().hex[:8]}"  # type: ignore[index]

    first = _confirm_strategy(
        api_test_client,
        auth_headers,
        session_id=session_id,
        dsl_json=dsl_v1,
    )
    strategy_id = str(first["strategy_id"])
    UUID(strategy_id)

    dsl_v2 = deepcopy(dsl_v1)
    dsl_v2["strategy"]["description"] = f"pytest-description-{uuid4().hex[:8]}"  # type: ignore[index]
    second = _confirm_strategy(
        api_test_client,
        auth_headers,
        session_id=session_id,
        dsl_json=dsl_v2,
        strategy_id=strategy_id,
    )
    assert str(second["strategy_id"]) == strategy_id

    versions_response = api_test_client.get(
        f"/api/v1/strategies/{strategy_id}/versions",
        headers=auth_headers,
        params={"limit": 20},
    )
    assert versions_response.status_code == 200, versions_response.text
    versions = versions_response.json()
    assert isinstance(versions, list)
    assert len(versions) >= 2, versions

    version_values = sorted({int(item["version"]) for item in versions})
    from_version = version_values[0]
    to_version = version_values[-1]
    assert to_version > from_version, version_values

    diff_response = api_test_client.get(
        f"/api/v1/strategies/{strategy_id}/diff",
        headers=auth_headers,
        params={
            "from_version": from_version,
            "to_version": to_version,
        },
    )
    assert diff_response.status_code == 200, diff_response.text
    diff_payload = diff_response.json()
    assert int(diff_payload["patch_op_count"]) >= 1
    assert isinstance(diff_payload.get("diff_items"), list)
    assert len(diff_payload["diff_items"]) >= 1
