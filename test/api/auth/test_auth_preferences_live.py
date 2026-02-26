from __future__ import annotations

from fastapi.testclient import TestClient


def test_000_accessibility_preferences_readable(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = api_test_client.get("/api/v1/auth/preferences", headers=auth_headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["theme_mode"] in {"light", "dark", "system"}
    assert payload["locale"] in {"en", "zh"}
    assert payload["font_scale"] in {"small", "default", "large"}


def test_010_preferences_update_and_restore(
    api_test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    original_response = api_test_client.get("/api/v1/auth/preferences", headers=auth_headers)
    assert original_response.status_code == 200, original_response.text
    original = original_response.json()

    target = {
        "theme_mode": "dark" if original["theme_mode"] != "dark" else "light",
        "locale": "zh" if original["locale"] != "zh" else "en",
        "font_scale": "large" if original["font_scale"] != "large" else "small",
    }

    update_response = api_test_client.put(
        "/api/v1/auth/preferences",
        headers=auth_headers,
        json=target,
    )
    assert update_response.status_code == 200, update_response.text
    updated = update_response.json()
    assert updated["theme_mode"] == target["theme_mode"]
    assert updated["locale"] == target["locale"]
    assert updated["font_scale"] == target["font_scale"]

    # Restore original preferences to avoid mutating persistent user state.
    restore_response = api_test_client.put(
        "/api/v1/auth/preferences",
        headers=auth_headers,
        json={
            "theme_mode": original["theme_mode"],
            "locale": original["locale"],
            "font_scale": original["font_scale"],
        },
    )
    assert restore_response.status_code == 200, restore_response.text
