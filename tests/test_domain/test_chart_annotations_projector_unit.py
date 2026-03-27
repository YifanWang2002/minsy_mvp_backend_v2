from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.chart_annotations.projector import (
    build_backtest_trade_annotation_documents,
    build_execution_annotation_documents,
)


def _dt(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_build_execution_annotation_documents_emits_position_bundle_and_managed_exit_lines() -> None:
    deployment_id = uuid4()
    order_id = uuid4()
    fill_time = _dt(2026, 3, 27, 9, 31)
    docs = build_execution_annotation_documents(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        deployment_id=deployment_id,
        signal_events=[],
        orders=[
            SimpleNamespace(
                id=order_id,
                symbol="BTCUSD",
                side="buy",
                submitted_at=_dt(2026, 3, 27, 9, 30),
                client_order_id="ord-1",
                status="filled",
                price=101.25,
            )
        ],
        fills=[
            SimpleNamespace(
                order_id=order_id,
                fill_price=101.5,
                fill_qty=1.25,
                filled_at=fill_time,
            )
        ],
        positions=[
            SimpleNamespace(
                symbol="BTCUSD",
                side="long",
                avg_entry_price=101.5,
                qty=1.25,
                mark_price=103.0,
            )
        ],
        managed_exit_state={
            "symbol": "BTCUSD",
            "side": "long",
            "stop_price": 96.5,
            "take_price": 110.25,
        },
    )

    docs_by_id = {doc["id"]: doc for doc in docs}
    position_id = f"position:{deployment_id}:BTCUSD"
    stop_loss_id = f"stop_loss:{deployment_id}:BTCUSD"
    take_profit_id = f"take_profit:{deployment_id}:BTCUSD"
    assert position_id in docs_by_id
    assert stop_loss_id in docs_by_id
    assert take_profit_id in docs_by_id

    position_doc = docs_by_id[position_id]
    assert position_doc["semantic"]["kind"] == "position"
    assert position_doc["tool"]["vendor_type"] == "long_position"
    assert position_doc["anchors"]["points"][0] == {
        "time": int(fill_time.timestamp()),
        "price": 101.5,
    }
    assert position_doc["anchors"]["points"][1]["time"] == int(fill_time.timestamp()) + (15 * 60 * 12)
    assert position_doc["vendor_native"]["trade"]["stop_price"] == 96.5
    assert position_doc["vendor_native"]["trade"]["target_price"] == 110.25
    assert position_doc["vendor_native"]["trade"]["qty"] == 1.25
    assert position_doc["content"]["trade"]["direction"] == "long"
    assert position_doc["content"]["trade"]["risk_reward_ratio"] == 1.75
    assert position_doc["relations"]["group_id"] == f"execution:{deployment_id}:BTCUSD:trade_bundle"
    assert position_doc["relations"]["composite_members"] == [
        stop_loss_id,
        take_profit_id,
    ]

    stop_loss_doc = docs_by_id[stop_loss_id]
    assert stop_loss_doc["semantic"]["kind"] == "stop_loss"
    assert stop_loss_doc["semantic"]["role"] == "risk"
    assert stop_loss_doc["content"]["trade"]["stop_price"] == 96.5
    assert stop_loss_doc["anchors"]["points"] == [
        {"time": int(fill_time.timestamp()), "price": 96.5},
        {"time": int(fill_time.timestamp()) + (15 * 60 * 12), "price": 96.5},
    ]

    take_profit_doc = docs_by_id[take_profit_id]
    assert take_profit_doc["semantic"]["kind"] == "take_profit"
    assert take_profit_doc["content"]["trade"]["target_price"] == 110.25
    assert take_profit_doc["anchors"]["points"] == [
        {"time": int(fill_time.timestamp()), "price": 110.25},
        {"time": int(fill_time.timestamp()) + (15 * 60 * 12), "price": 110.25},
    ]


def test_build_execution_annotation_documents_skips_managed_exit_lines_when_state_mismatches_symbol() -> None:
    deployment_id = uuid4()
    order_id = uuid4()
    docs = build_execution_annotation_documents(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        deployment_id=deployment_id,
        signal_events=[],
        orders=[
            SimpleNamespace(
                id=order_id,
                symbol="BTCUSD",
                side="buy",
                submitted_at=_dt(2026, 3, 27, 10, 0),
                client_order_id="ord-2",
                status="filled",
                price=99.0,
            )
        ],
        fills=[
            SimpleNamespace(
                order_id=order_id,
                fill_price=99.0,
                fill_qty=1.0,
                filled_at=_dt(2026, 3, 27, 10, 1),
            )
        ],
        positions=[
            SimpleNamespace(
                symbol="BTCUSD",
                side="long",
                avg_entry_price=99.0,
                qty=1.0,
                mark_price=100.0,
            )
        ],
        managed_exit_state={
            "symbol": "ETHUSD",
            "side": "long",
            "stop_price": 94.0,
            "take_price": 108.0,
        },
    )

    doc_ids = {doc["id"] for doc in docs}
    assert f"position:{deployment_id}:BTCUSD" in doc_ids
    assert f"stop_loss:{deployment_id}:BTCUSD" not in doc_ids
    assert f"take_profit:{deployment_id}:BTCUSD" not in doc_ids


def test_build_execution_annotation_documents_emits_signal_and_order_trade_summaries() -> None:
    deployment_id = uuid4()
    signal_id = uuid4()
    order_id = uuid4()
    bar_time = _dt(2026, 3, 27, 11, 15)
    docs = build_execution_annotation_documents(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        deployment_id=deployment_id,
        signal_events=[
            SimpleNamespace(
                id=signal_id,
                bar_time=bar_time,
                signal="OPEN_LONG",
                reason="breakout_confirmed",
                metadata_={
                    "order_id": str(order_id),
                    "qty": "1.5",
                    "mark_price": 101.75,
                    "stop_price": 96.5,
                    "take_price": 111.25,
                },
            )
        ],
        orders=[
            SimpleNamespace(
                id=order_id,
                symbol="BTCUSD",
                side="buy",
                submitted_at=bar_time,
                client_order_id="ord-3",
                status="filled",
                price=101.25,
                qty=1.5,
            )
        ],
        fills=[
            SimpleNamespace(
                order_id=order_id,
                fill_price=101.5,
                fill_qty=1.5,
                filled_at=_dt(2026, 3, 27, 11, 16),
            )
        ],
        positions=[],
        managed_exit_state=None,
    )

    docs_by_id = {doc["id"]: doc for doc in docs}
    signal_doc = docs_by_id[f"signal:{signal_id}"]
    order_doc = docs_by_id[f"order:{order_id}"]

    assert signal_doc["semantic"]["direction"] == "long"
    assert signal_doc["content"]["trade"]["qty"] == 1.5
    assert signal_doc["content"]["trade"]["mark_price"] == 101.75
    assert signal_doc["content"]["trade"]["stop_price"] == 96.5
    assert signal_doc["content"]["trade"]["target_price"] == 111.25
    assert signal_doc["content"]["trade"]["risk_reward_ratio"] == 1.8095
    assert signal_doc["relations"]["group_id"] == f"execution:{deployment_id}:{order_id}:order_flow"

    assert order_doc["semantic"]["kind"] == "entry"
    assert order_doc["content"]["text"] == "ord-3"
    assert order_doc["content"]["trade"]["direction"] == "long"
    assert order_doc["content"]["trade"]["entry_price"] == 101.5
    assert order_doc["content"]["trade"]["qty"] == 1.5
    assert order_doc["vendor_native"]["trade"]["mark_price"] == 101.5
    assert order_doc["relations"]["group_id"] == f"execution:{deployment_id}:{order_id}:order_flow"


def test_build_execution_annotation_documents_maps_close_sell_order_to_long_exit() -> None:
    deployment_id = uuid4()
    order_id = uuid4()
    docs = build_execution_annotation_documents(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        deployment_id=deployment_id,
        signal_events=[],
        orders=[
            SimpleNamespace(
                id=order_id,
                symbol="BTCUSD",
                side="sell",
                submitted_at=_dt(2026, 3, 27, 12, 0),
                client_order_id="ord-close",
                status="filled",
                price=109.0,
                qty=1.0,
                metadata_={
                    "signal": "CLOSE",
                    "unified_direction": "long",
                },
            )
        ],
        fills=[
            SimpleNamespace(
                order_id=order_id,
                fill_price=109.0,
                fill_qty=1.0,
                filled_at=_dt(2026, 3, 27, 12, 1),
            )
        ],
        positions=[],
        managed_exit_state=None,
    )

    order_doc = {doc["id"]: doc for doc in docs}[f"order:{order_id}"]
    assert order_doc["semantic"]["kind"] == "exit"
    assert order_doc["semantic"]["direction"] == "long"
    assert "entry_price" not in order_doc["content"]["trade"]
    assert order_doc["content"]["trade"]["exit_price"] == 109.0
    assert order_doc["content"]["trade"]["direction"] == "long"


def test_build_backtest_trade_annotation_documents_emits_grouped_risk_reward_bundle() -> None:
    backtest_id = uuid4()
    docs = build_backtest_trade_annotation_documents(
        market="crypto",
        symbol="BTCUSD",
        timeframe="15m",
        backtest_id=backtest_id,
        trade={
            "trade_uid": "trade-1",
            "side": "long",
            "entry_price": 100.0,
            "exit_price": 109.0,
        },
        trade_annotations=[
            {
                "kind": "trade_entry",
                "time": "2026-03-27T09:30:00Z",
                "price": 100.0,
                "label": "Entry",
            },
            {
                "kind": "stop_loss",
                "time": "2026-03-27T09:31:00Z",
                "price": 95.0,
                "label": "Stop",
            },
            {
                "kind": "take_profit",
                "time": "2026-03-27T09:32:00Z",
                "price": 110.0,
                "label": "Take",
            },
            {
                "kind": "trade_exit",
                "time": "2026-03-27T11:00:00Z",
                "price": 109.0,
                "label": "Exit",
            },
        ],
    )

    docs_by_id = {doc["id"]: doc for doc in docs}
    risk_reward = docs_by_id["backtest:trade-1:risk_reward"]
    stop_loss = docs_by_id["backtest:trade-1:stop_loss"]
    take_profit = docs_by_id["backtest:trade-1:take_profit"]

    assert risk_reward["anchors"]["points"] == [
        {"time": int(_dt(2026, 3, 27, 9, 30).timestamp()), "price": 100.0},
        {"time": int(_dt(2026, 3, 27, 11, 0).timestamp()), "price": 100.0},
    ]
    assert risk_reward["relations"]["group_id"] == "backtest:trade-1:trade_bundle"
    assert risk_reward["relations"]["composite_members"] == [
        "backtest:trade-1:stop_loss",
        "backtest:trade-1:take_profit",
    ]
    assert risk_reward["vendor_native"]["trade"]["stop_price"] == 95.0
    assert risk_reward["vendor_native"]["trade"]["target_price"] == 110.0
    assert risk_reward["content"]["trade"]["exit_price"] == 109.0
    assert risk_reward["content"]["trade"]["risk_reward_ratio"] == 2.0

    assert stop_loss["anchors"]["points"] == [
        {"time": int(_dt(2026, 3, 27, 9, 30).timestamp()), "price": 95.0},
        {"time": int(_dt(2026, 3, 27, 11, 0).timestamp()), "price": 95.0},
    ]
    assert stop_loss["relations"]["parent_id"] == "backtest:trade-1:risk_reward"
    assert stop_loss["content"]["trade"]["stop_price"] == 95.0
    assert take_profit["anchors"]["points"] == [
        {"time": int(_dt(2026, 3, 27, 9, 30).timestamp()), "price": 110.0},
        {"time": int(_dt(2026, 3, 27, 11, 0).timestamp()), "price": 110.0},
    ]
    assert take_profit["content"]["trade"]["target_price"] == 110.0
