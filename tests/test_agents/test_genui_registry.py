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


def test_backtest_charts_payload_is_normalized() -> None:
    normalized = normalize_genui_payloads(
        [
            {
                "type": "backtest_charts",
                "job_id": " 8f17881d-1fd3-4306-b6e9-70fa896f0fa6 ",
                "charts": ["equity_curve", "", "monthly_return_table"],
                "sampling": " EOD ",
                "max_points": 99999,
                "window_bars": 252.3,
            }
        ],
        allow_passthrough_unregistered=False,
    )

    assert normalized == [
        {
            "type": "backtest_charts",
            "job_id": "8f17881d-1fd3-4306-b6e9-70fa896f0fa6",
            "charts": ["equity_curve", "monthly_return_table"],
            "sampling": "eod",
            "max_points": 5000,
            "window_bars": 252,
        }
    ]


def test_choice_prompt_payload_with_empty_options_is_preserved_for_handler_fill() -> None:
    normalized = normalize_genui_payloads(
        [
            {
                "type": "choice_prompt",
                "choice_id": "target_instrument",
                "question": "Which instrument?",
                "options": [],
            }
        ],
        allow_passthrough_unregistered=False,
    )

    assert normalized == [
        {
            "type": "choice_prompt",
            "choice_id": "target_instrument",
            "question": "Which instrument?",
            "options": [],
        }
    ]
