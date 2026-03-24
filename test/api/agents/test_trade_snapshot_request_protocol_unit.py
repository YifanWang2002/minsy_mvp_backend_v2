from __future__ import annotations

import json

from apps.api.orchestration import ChatOrchestrator


def test_orchestrator_extracts_trade_snapshot_request_payload() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    payload = {
        "job_id": "8fdc6d6b-33df-4c7d-87fd-860239f67816",
        "trade_index": 12,
        "trade_uid": "12:entry:exit",
        "visible_indicator_keys": ["ema_20", "rsi_14"],
        "filters": {
            "side": "long",
            "outcome": "win",
            "date_range": "30d",
            "pnl_sort": "pnl_pct_desc",
            "exit_reason": "all",
        },
        "lookback_bars": 40,
        "lookforward_bars": 20,
        "user_prompt": "Analyze this trade and suggest changes.",
    }
    message = (
        "Analyze this trade and suggest changes.\n"
        f"<TRADE_SNAPSHOT_REQUEST>{json.dumps(payload)}</TRADE_SNAPSHOT_REQUEST>"
    )

    parsed = orchestrator._extract_trade_snapshot_request_from_message(message)

    assert parsed is not None
    assert parsed["job_id"] == payload["job_id"]
    assert parsed["trade_index"] == 12
    assert parsed["trade_uid"] == "12:entry:exit"
    assert parsed["visible_indicator_keys"] == ["ema_20", "rsi_14"]
    assert parsed["filters"]["side"] == "long"
    assert parsed["lookback_bars"] == 40
    assert parsed["lookforward_bars"] == 20


def test_orchestrator_rejects_invalid_trade_snapshot_request_payload() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    message = (
        "<TRADE_SNAPSHOT_REQUEST>"
        '{"job_id":"job-1","trade_index":-1}'
        "</TRADE_SNAPSHOT_REQUEST>"
    )

    parsed = orchestrator._extract_trade_snapshot_request_from_message(message)

    assert parsed is None


def test_orchestrator_stores_display_message_without_trade_request_token() -> None:
    orchestrator = ChatOrchestrator(None)  # type: ignore[arg-type]
    message = (
        "Please analyze this trade and show evidence bars.\n"
        '<TRADE_SNAPSHOT_REQUEST>{"job_id":"job-1","trade_index":2,'
        '"user_prompt":"Please analyze this trade and show evidence bars."}'
        "</TRADE_SNAPSHOT_REQUEST>"
    )

    stored = orchestrator._resolve_user_message_for_storage(message)

    assert stored == "Please analyze this trade and show evidence bars."
