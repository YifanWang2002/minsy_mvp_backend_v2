from __future__ import annotations

from apps.api.agents.genui_registry import normalize_genui_payloads


def test_choice_prompt_with_no_valid_options_is_dropped() -> None:
    payloads = normalize_genui_payloads(
        [
            {
                "type": "choice_prompt",
                "choice_id": "deployment_confirmation_status",
                "question": "Confirm this deployment summary before execution.",
                "options": [
                    {"id": "", "label": "Missing id"},
                    {"id": "confirmed", "label": ""},
                ],
            }
        ]
    )

    assert payloads == []


def test_choice_prompt_with_valid_options_survives_normalization() -> None:
    payloads = normalize_genui_payloads(
        [
            {
                "type": "choice_prompt",
                "choice_id": "deployment_confirmation_status",
                "question": "Confirm this deployment summary before execution.",
                "options": [
                    {"id": "confirmed", "label": "Confirm deployment"},
                    {"id": "needs_changes", "label": "Make changes"},
                ],
            }
        ]
    )

    assert payloads == [
        {
            "type": "choice_prompt",
            "choice_id": "deployment_confirmation_status",
            "question": "Confirm this deployment summary before execution.",
            "options": [
                {"id": "confirmed", "label": "Confirm deployment"},
                {"id": "needs_changes", "label": "Make changes"},
            ],
        }
    ]
