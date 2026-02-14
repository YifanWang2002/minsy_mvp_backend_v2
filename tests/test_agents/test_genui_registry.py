from __future__ import annotations

from src.agents.genui_registry import (
    normalize_genui_payloads,
    register_genui_normalizer,
)


def test_unknown_genui_type_can_passthrough() -> None:
    payloads = [{"type": "custom_card", "title": "hello"}]

    normalized = normalize_genui_payloads(payloads, allow_passthrough_unregistered=True)

    assert normalized == payloads


def test_registered_genui_normalizer_is_used() -> None:
    register_genui_normalizer(
        "custom_metric",
        lambda payload: {
            "type": "custom_metric",
            "label": str(payload.get("label") or "metric").strip(),
        },
    )

    normalized = normalize_genui_payloads(
        [{"type": "custom_metric", "label": " sharpe "}],
        allow_passthrough_unregistered=False,
    )

    assert normalized == [{"type": "custom_metric", "label": "sharpe"}]
