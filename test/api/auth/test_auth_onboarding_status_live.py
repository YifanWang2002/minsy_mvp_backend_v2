from __future__ import annotations

from fastapi.testclient import TestClient

_ALLOWED = {"pending", "completed", "canceled"}
_SECTIONS = ("home", "strategies", "portfolio")


def _get_status(
    api_test_client: TestClient, auth_headers: dict[str, str]
) -> dict[str, str]:
    response = api_test_client.get(
        "/api/v1/auth/onboarding-status",
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in _SECTIONS:
        assert payload[key] in _ALLOWED
    return {key: payload[key] for key in _SECTIONS}


def _set_status(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
    *,
    section: str,
    status: str,
) -> dict[str, str]:
    response = api_test_client.put(
        "/api/v1/auth/onboarding-status",
        headers=auth_headers,
        json={"section": section, "status": status},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    return {key: payload[key] for key in _SECTIONS}


def test_000_accessibility_onboarding_status_readable(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    _get_status(api_test_client, auth_headers)


def test_010_onboarding_status_update_reset_and_restore(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    original = _get_status(api_test_client, auth_headers)

    target = "completed" if original["home"] != "completed" else "canceled"
    updated = _set_status(
        api_test_client,
        auth_headers,
        section="home",
        status=target,
    )
    assert updated["home"] == target

    reset_response = api_test_client.post(
        "/api/v1/auth/onboarding-status/reset",
        headers=auth_headers,
    )
    assert reset_response.status_code == 200, reset_response.text
    reset_payload = reset_response.json()
    for key in _SECTIONS:
        assert reset_payload[key] == "pending"

    for key in _SECTIONS:
        _set_status(
            api_test_client,
            auth_headers,
            section=key,
            status=original[key],
        )
