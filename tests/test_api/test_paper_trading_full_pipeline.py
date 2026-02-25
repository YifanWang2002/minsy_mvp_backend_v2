from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.engine.backtest.factors import prepare_backtest_frame
from src.engine.execution.adapters.base import OhlcvBar, QuoteSnapshot
from src.engine.execution.signal_store import signal_store
from src.engine.market_data.runtime import market_data_runtime
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload
from src.engine.strategy.pipeline import parse_strategy_payload
from src.main import app


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    symbol: str
    variant: str
    order_qty: float
    capital: float
    strategy_payload: dict[str, Any]
    deployment_id: UUID


class FlowRecorder:
    """Collects full-flow events/errors and persists them as JSONL artifacts."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.events_file = self._artifact_dir / "paper_trading_full_flow.events.jsonl"
        self.errors_file = self._artifact_dir / "paper_trading_full_flow.errors.jsonl"

    def event(self, stage: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            **payload,
        }
        self.events.append(row)
        print(f"[FULL_FLOW][EVENT] {json.dumps(row, ensure_ascii=True, default=str)}")

    def error(self, stage: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            **payload,
        }
        self.errors.append(row)
        print(f"[FULL_FLOW][ERROR] {json.dumps(row, ensure_ascii=True, default=str)}")

    def flush(self) -> None:
        self.events_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in self.events) + "\n",
            encoding="utf-8",
        )
        self.errors_file.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in self.errors) + "\n",
            encoding="utf-8",
        )
        print(f"[FULL_FLOW][ARTIFACT] events={self.events_file}")
        print(f"[FULL_FLOW][ARTIFACT] errors={self.errors_file}")


def _register_and_get_token(client: TestClient, recorder: FlowRecorder) -> str:
    email = f"full_pipeline_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Full Pipeline User"},
    )
    recorder.event(
        "auth_register",
        status=response.status_code,
        email=email,
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def _replace_factor_refs(node: Any, factor_id_map: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ref" and isinstance(value, str):
                replacement = value
                for old_id, new_id in factor_id_map.items():
                    if replacement == old_id:
                        replacement = new_id
                        break
                    prefix = f"{old_id}."
                    if replacement.startswith(prefix):
                        replacement = f"{new_id}.{replacement.removeprefix(prefix)}"
                        break
                node[key] = replacement
                continue
            _replace_factor_refs(value, factor_id_map)
        return

    if isinstance(node, list):
        for item in node:
            _replace_factor_refs(item, factor_id_map)


def _rename_factor_ids(payload: dict[str, Any], factor_id_map: dict[str, str]) -> None:
    factors = payload.get("factors")
    if isinstance(factors, dict):
        remapped: dict[str, Any] = {}
        for factor_id, definition in factors.items():
            remapped[factor_id_map.get(str(factor_id), str(factor_id))] = definition
        payload["factors"] = remapped

    _replace_factor_refs(payload.get("trade"), factor_id_map)


def _high_frequency_payload(symbol: str, variant: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["strategy"]["name"] = f"{symbol} {variant} 1m 高频"
    payload["strategy"]["description"] = f"{symbol} {variant} high-frequency 1m long-only synthetic"
    payload["universe"] = {"market": "crypto", "tickers": [symbol]}
    payload["timeframe"] = "1m"

    payload["factors"]["ema_9"]["params"]["period"] = 3
    payload["factors"]["ema_21"]["params"]["period"] = 6
    payload["factors"]["rsi_14"]["params"]["period"] = 4
    payload["factors"]["macd_12_26_9"]["params"]["fast"] = 3
    payload["factors"]["macd_12_26_9"]["params"]["slow"] = 7
    payload["factors"]["macd_12_26_9"]["params"]["signal"] = 3
    payload["factors"]["atr_14"]["params"]["period"] = 4
    _rename_factor_ids(
        payload,
        {
            "ema_9": "ema_3",
            "ema_21": "ema_6",
            "rsi_14": "rsi_4",
            "macd_12_26_9": "macd_3_7_3",
            "atr_14": "atr_4",
        },
    )

    if variant == "trend":
        payload["trade"]["long"]["entry"]["condition"] = {
            "all": [
                {"cmp": {"left": {"ref": "ema_3"}, "op": "gt", "right": {"ref": "ema_6"}}},
                {"cmp": {"left": {"ref": "rsi_4"}, "op": "lt", "right": 85}},
            ]
        }
        payload["trade"]["short"]["entry"]["condition"] = {
            "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}
        }
        payload["trade"]["long"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "trend_long_exit",
                "condition": {
                    "any": [
                        {"cmp": {"left": {"ref": "ema_3"}, "op": "lt", "right": {"ref": "ema_6"}}},
                        {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 70}},
                    ]
                },
            }
        ]
        payload["trade"]["short"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "trend_short_exit_disabled",
                "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
            }
        ]
        return payload

    payload["trade"]["long"]["entry"]["condition"] = {
        "all": [
            {"cmp": {"left": {"ref": "ema_3"}, "op": "lt", "right": {"ref": "ema_6"}}},
            {"cmp": {"left": {"ref": "rsi_4"}, "op": "lt", "right": 45}},
        ]
    }
    payload["trade"]["short"]["entry"]["condition"] = {
        "cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}
    }
    payload["trade"]["long"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "mean_long_exit",
            "condition": {
                "any": [
                    {"cmp": {"left": {"ref": "ema_3"}, "op": "gt", "right": {"ref": "ema_6"}}},
                    {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 58}},
                ]
            },
        }
    ]
    payload["trade"]["short"]["exits"] = [
        {
            "type": "signal_exit",
            "name": "mean_short_exit_disabled",
            "condition": {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}},
        }
    ]
    return payload


def _safe_last_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    if math.isnan(val):
        return None
    return val


def _factor_snapshot(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, float | None]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    snapshot: dict[str, float | None] = {}
    factors = payload.get("factors", {}) if isinstance(payload.get("factors"), dict) else {}
    for factor_id, definition in factors.items():
        outputs = ()
        if isinstance(definition, dict) and isinstance(definition.get("outputs"), list):
            outputs = tuple(str(item) for item in definition["outputs"])
        if outputs:
            for output in outputs:
                col = f"{factor_id}.{output}"
                snapshot[col] = _safe_last_number(row.get(col))
            continue

        if factor_id in frame.columns:
            snapshot[factor_id] = _safe_last_number(row.get(factor_id))
            continue

        prefix = f"{factor_id}."
        prefixed = [col for col in frame.columns if str(col).startswith(prefix)]
        for col in prefixed[:3]:
            snapshot[str(col)] = _safe_last_number(row.get(col))
    return snapshot


def _build_synthetic_bar(
    *,
    symbol: str,
    minute: int,
    previous_close: Decimal,
    ts: datetime,
) -> tuple[OhlcvBar, Decimal]:
    if symbol == "BTCUSD":
        base = 52000.0
        amp = 850.0
        amp2 = 300.0
    else:
        base = 3200.0
        amp = 120.0
        amp2 = 40.0
    close_f = base + amp * math.sin(minute * 0.63) + amp2 * math.cos(minute * 1.37)
    close = Decimal(str(round(close_f, 6)))
    open_ = previous_close
    high = max(open_, close) + Decimal("1.5")
    low = min(open_, close) - Decimal("1.5")
    volume = Decimal(str(round(200 + abs(float(close - open_)) * 2.5 + (minute % 11) * 3, 6)))
    bar = OhlcvBar(
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )
    return bar, close


def _as_frame(bar_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": [float(row["open"]) for row in bar_rows],
            "high": [float(row["high"]) for row in bar_rows],
            "low": [float(row["low"]) for row in bar_rows],
            "close": [float(row["close"]) for row in bar_rows],
            "volume": [float(row["volume"]) for row in bar_rows],
        },
        index=pd.to_datetime([row["timestamp"] for row in bar_rows], utc=True),
    )
    return frame


@pytest.mark.parametrize("minutes", [90])
def test_paper_trading_full_flow_btc_eth_multi_strategy(
    minutes: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Full pipeline test:
    synthetic 1m data acquisition -> DSL factor calculation -> signal -> order/fill/position/pnl.
    """
    recorder = FlowRecorder(tmp_path / "artifacts")
    monkeypatch.setattr(settings, "paper_trading_enqueue_on_start", False)
    monkeypatch.setattr(settings, "paper_trading_execute_orders", False)
    monkeypatch.setattr(settings, "paper_trading_kill_switch_global", False)
    monkeypatch.setattr("src.api.routers.market_data.enqueue_market_data_refresh", lambda **_: "task-id")
    signal_store.clear()
    market_data_runtime.reset()

    try:
        with TestClient(app) as client:
            token = _register_and_get_token(client, recorder)
            headers = {"Authorization": f"Bearer {token}"}

            broker = client.post(
                "/api/v1/broker-accounts?validate=false",
                headers=headers,
                json={
                    "provider": "alpaca",
                    "mode": "paper",
                    "credentials": {"api_key": "demo", "api_secret": "demo"},
                    "metadata": {"label": "full-flow"},
                },
            )
            recorder.event("broker_account_create", status=broker.status_code)
            assert broker.status_code == 201, broker.text
            broker_id = broker.json()["broker_account_id"]

            subscribe = client.post(
                "/api/v1/market-data/subscriptions",
                headers=headers,
                params={"market": "crypto"},
                json={"symbols": ["BTCUSD", "ETHUSD"]},
            )
            recorder.event(
                "market_subscription",
                status=subscribe.status_code,
                body=subscribe.json() if subscribe.status_code == 200 else subscribe.text,
            )
            assert subscribe.status_code == 200, subscribe.text

            deployments: list[DeploymentPlan] = []
            for symbol, variant, order_qty, capital in [
                ("BTCUSD", "trend", 0.01, 20000.0),
                ("BTCUSD", "mean", 0.01, 20000.0),
                ("ETHUSD", "trend", 0.2, 15000.0),
                ("ETHUSD", "mean", 0.2, 15000.0),
            ]:
                payload = _high_frequency_payload(symbol, variant)
                new_thread = client.post("/api/v1/chat/new-thread", headers=headers, json={"metadata": {}})
                recorder.event(
                    "chat_new_thread",
                    symbol=symbol,
                    variant=variant,
                    status=new_thread.status_code,
                )
                assert new_thread.status_code == 201, new_thread.text
                session_id = new_thread.json()["session_id"]

                confirm = client.post(
                    "/api/v1/strategies/confirm",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "dsl_json": payload,
                        "auto_start_backtest": False,
                    },
                )
                recorder.event(
                    "strategy_confirm",
                    symbol=symbol,
                    variant=variant,
                    status=confirm.status_code,
                )
                assert confirm.status_code == 200, confirm.text
                strategy_id = confirm.json()["strategy_id"]

                create_deployment = client.post(
                    "/api/v1/deployments",
                    headers=headers,
                    json={
                        "strategy_id": strategy_id,
                        "broker_account_id": broker_id,
                        "mode": "paper",
                        "capital_allocated": str(capital),
                        "risk_limits": {
                            "order_qty": order_qty,
                            "max_position_notional": 10000,
                            "max_symbol_exposure_pct": 0.9,
                        },
                        "runtime_state": {"seed": "full-flow"},
                    },
                )
                recorder.event(
                    "deployment_create",
                    symbol=symbol,
                    variant=variant,
                    status=create_deployment.status_code,
                )
                assert create_deployment.status_code == 201, create_deployment.text
                deployment_id = UUID(create_deployment.json()["deployment_id"])

                start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
                recorder.event(
                    "deployment_start",
                    symbol=symbol,
                    variant=variant,
                    deployment_id=str(deployment_id),
                    status=start.status_code,
                )
                assert start.status_code == 200, start.text

                deployments.append(
                    DeploymentPlan(
                        symbol=symbol,
                        variant=variant,
                        order_qty=order_qty,
                        capital=capital,
                        strategy_payload=payload,
                        deployment_id=deployment_id,
                    )
                )

            parsed_by_deployment: dict[UUID, Any] = {
                item.deployment_id: parse_strategy_payload(item.strategy_payload) for item in deployments
            }
            total_orders_by_deployment: dict[UUID, int] = {item.deployment_id: 0 for item in deployments}
            signal_counts: dict[UUID, dict[str, int]] = {item.deployment_id: {} for item in deployments}

            previous_close = {
                "BTCUSD": Decimal("52000"),
                "ETHUSD": Decimal("3200"),
            }
            start_ts = datetime(2026, 1, 6, 9, 30, tzinfo=UTC)

            for minute in range(minutes):
                ts = start_ts + timedelta(minutes=minute)
                for symbol in ("BTCUSD", "ETHUSD"):
                    bar, close = _build_synthetic_bar(
                        symbol=symbol,
                        minute=minute,
                        previous_close=previous_close[symbol],
                        ts=ts,
                    )
                    previous_close[symbol] = close
                    market_data_runtime.ingest_1m_bar(market="crypto", symbol=symbol, bar=bar)
                    market_data_runtime.upsert_quote(
                        market="crypto",
                        symbol=symbol,
                        quote=QuoteSnapshot(
                            symbol=symbol,
                            bid=close - Decimal("0.5"),
                            ask=close + Decimal("0.5"),
                            last=close,
                            timestamp=ts,
                        ),
                    )
                    recorder.event(
                        "data_acquisition_synthetic",
                        minute=minute,
                        symbol=symbol,
                        timestamp=ts.isoformat(),
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    )

                    quote_resp = client.get(
                        "/api/v1/market-data/quote",
                        headers=headers,
                        params={"market": "crypto", "symbol": symbol},
                    )
                    if quote_resp.status_code != 200:
                        recorder.error(
                            "market_quote_read_error",
                            minute=minute,
                            symbol=symbol,
                            status=quote_resp.status_code,
                            body=quote_resp.text,
                        )
                    else:
                        body = quote_resp.json()
                        recorder.event(
                            "market_quote_read",
                            minute=minute,
                            symbol=symbol,
                            last=body.get("last"),
                            bid=body.get("bid"),
                            ask=body.get("ask"),
                        )

                for plan in deployments:
                    bars_resp = client.get(
                        "/api/v1/market-data/bars",
                        headers=headers,
                        params={
                            "market": "crypto",
                            "symbol": plan.symbol,
                            "timeframe": "1m",
                            "limit": 120,
                        },
                    )
                    if bars_resp.status_code != 200:
                        recorder.error(
                            "market_bars_read_error",
                            minute=minute,
                            symbol=plan.symbol,
                            deployment_id=str(plan.deployment_id),
                            status=bars_resp.status_code,
                            body=bars_resp.text,
                        )
                        continue

                    bars = bars_resp.json().get("bars", [])
                    if len(bars) >= 8:
                        try:
                            frame = _as_frame(bars)
                            enriched = prepare_backtest_frame(
                                frame,
                                strategy=parsed_by_deployment[plan.deployment_id],
                            )
                            snapshot = _factor_snapshot(enriched, plan.strategy_payload)
                            recorder.event(
                                "dsl_factor_calculation",
                                minute=minute,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                bar_count=len(bars),
                                factors=snapshot,
                            )
                        except Exception as exc:  # noqa: BLE001
                            recorder.error(
                                "dsl_factor_calculation_error",
                                minute=minute,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )

                    process = client.post(
                        f"/api/v1/deployments/{plan.deployment_id}/process-now",
                        headers=headers,
                    )
                    if process.status_code != 200:
                        recorder.error(
                            "signal_process_error",
                            minute=minute,
                            deployment_id=str(plan.deployment_id),
                            symbol=plan.symbol,
                            variant=plan.variant,
                            status=process.status_code,
                            body=process.text,
                        )
                        continue

                    process_body = process.json()
                    assert process_body.get("execution_event_id") is not None
                    signal = str(process_body.get("signal", ""))
                    signal_counts[plan.deployment_id][signal] = signal_counts[plan.deployment_id].get(signal, 0) + 1
                    order_id = process_body.get("order_id")
                    recorder.event(
                        "signal_process",
                        minute=minute,
                        deployment_id=str(plan.deployment_id),
                        symbol=plan.symbol,
                        variant=plan.variant,
                        signal=signal,
                        reason=process_body.get("reason"),
                        order_id=order_id,
                        idempotent_hit=process_body.get("idempotent_hit"),
                    )
                    if order_id:
                        total_orders_by_deployment[plan.deployment_id] += 1

            total_fills = 0
            total_signals_logged = 0
            for plan in deployments:
                orders_resp = client.get(f"/api/v1/deployments/{plan.deployment_id}/orders", headers=headers)
                fills_resp = client.get(f"/api/v1/deployments/{plan.deployment_id}/fills", headers=headers)
                positions_resp = client.get(f"/api/v1/deployments/{plan.deployment_id}/positions", headers=headers)
                pnl_resp = client.get(f"/api/v1/deployments/{plan.deployment_id}/pnl", headers=headers)
                signals_resp = client.get(f"/api/v1/deployments/{plan.deployment_id}/signals", headers=headers)

                for stage, resp in [
                    ("orders_summary", orders_resp),
                    ("fills_summary", fills_resp),
                    ("positions_summary", positions_resp),
                    ("pnl_summary", pnl_resp),
                    ("signals_summary", signals_resp),
                ]:
                    if resp.status_code != 200:
                        recorder.error(
                            f"{stage}_error",
                            deployment_id=str(plan.deployment_id),
                            symbol=plan.symbol,
                            variant=plan.variant,
                            status=resp.status_code,
                            body=resp.text,
                        )
                        continue
                    payload = resp.json()
                    count = len(payload) if isinstance(payload, list) else 1
                    recorder.event(
                        stage,
                        deployment_id=str(plan.deployment_id),
                        symbol=plan.symbol,
                        variant=plan.variant,
                        count=count,
                    )

                if fills_resp.status_code == 200:
                    total_fills += len(fills_resp.json())
                if signals_resp.status_code == 200:
                    total_signals_logged += len(signals_resp.json())

            total_orders = sum(total_orders_by_deployment.values())
            per_symbol_orders: dict[str, int] = {}
            for plan in deployments:
                per_symbol_orders[plan.symbol] = per_symbol_orders.get(plan.symbol, 0) + total_orders_by_deployment[
                    plan.deployment_id
                ]

            recorder.event(
                "final_summary",
                total_minutes=minutes,
                deployment_count=len(deployments),
                total_orders=total_orders,
                total_fills=total_fills,
                total_signals_logged=total_signals_logged,
                orders_by_deployment={str(k): v for k, v in total_orders_by_deployment.items()},
                signals_by_deployment={str(k): v for k, v in signal_counts.items()},
                orders_by_symbol=per_symbol_orders,
                errors=len(recorder.errors),
            )

            for plan in deployments:
                assert total_orders_by_deployment[plan.deployment_id] > 0, (
                    f"No orders created for deployment {plan.deployment_id} "
                    f"({plan.symbol}, {plan.variant})."
                )
                assert signal_counts[plan.deployment_id].get("OPEN_LONG", 0) + signal_counts[plan.deployment_id].get(
                    "OPEN_SHORT", 0
                ) > 0, f"No opening signals for deployment {plan.deployment_id}."
                assert signal_counts[plan.deployment_id].get("OPEN_SHORT", 0) == 0, (
                    f"Long-only strategy emitted OPEN_SHORT for deployment {plan.deployment_id}."
                )
                assert signal_counts[plan.deployment_id].get("CLOSE", 0) > 0, (
                    f"No closing signals for deployment {plan.deployment_id}."
                )

            assert per_symbol_orders.get("BTCUSD", 0) > 0, "BTCUSD strategies never produced orders."
            assert per_symbol_orders.get("ETHUSD", 0) > 0, "ETHUSD strategies never produced orders."
            assert total_orders >= 12, f"Expected at least 12 orders, got {total_orders}."
            assert total_fills >= 12, f"Expected at least 12 fills, got {total_fills}."
            assert total_signals_logged > 0, "Signal log endpoint returned no events."
            assert not recorder.errors, (
                f"Full pipeline recorded {len(recorder.errors)} errors. "
                f"Check {recorder.errors_file}."
            )
    finally:
        recorder.flush()
