#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run live-data paper/live trading full-flow with real Alpaca order placement + cancel probe.

Flow covered in one long-running script:
1) Fetch live multi-symbol 1m bars + quotes from Alpaca market data.
2) Ingest into runtime cache and compute DSL factors.
3) Trigger signal -> order pipeline for multiple symbols/strategies/timeframes.
4) Place real provider orders through Alpaca adapter (using credentials from .env).
5) Submit a separate limit-order cancel probe at intervals.
6) Persist detailed events/errors to JSONL and print real-time logs.

WARNING:
- This script can place/cancel real provider orders.
- Use paper account by default (`ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from dotenv import load_dotenv
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.engine.backtest.factors import prepare_backtest_frame  # noqa: E402
from src.engine.execution.adapters.alpaca_trading import AlpacaTradingAdapter  # noqa: E402
from src.engine.execution.adapters.base import OrderIntent  # noqa: E402
from src.engine.execution.signal_store import signal_store  # noqa: E402
from src.engine.execution.runtime_state_store import runtime_state_store  # noqa: E402
from src.engine.market_data.aggregator import BarAggregator  # noqa: E402
from src.engine.market_data.providers.alpaca_rest import AlpacaRestProvider  # noqa: E402
from src.engine.market_data.runtime import market_data_runtime  # noqa: E402
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload  # noqa: E402
from src.engine.strategy.pipeline import parse_strategy_payload  # noqa: E402
from src.main import app  # noqa: E402

TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "rejected",
    "expired",
    "done_for_day",
}
ALPACA_CRYPTO_MIN_ORDER_NOTIONAL = Decimal("10")
DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "DOGEUSD")
DEFAULT_VARIANTS = ("trend", "mean", "breakout")
DEFAULT_TIMEFRAMES = ("1m", "2m", "5m")


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    symbol: str
    variant: str
    timeframe: str
    deployment_id: UUID
    order_qty: float
    strategy_payload: dict[str, Any]
    parsed_strategy: Any


@dataclass(frozen=True, slots=True)
class SymbolPlan:
    symbol: str
    order_qty: float
    capital_allocated: float


class FlowRecorder:
    """Append-only JSONL recorder for live flow events/errors."""

    def __init__(self, root_dir: Path) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = root_dir / stamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.output_dir / "live_full_flow.events.jsonl"
        self.errors_file = self.output_dir / "live_full_flow.errors.jsonl"
        self._events_fp = self.events_file.open("a", encoding="utf-8")
        self._errors_fp = self.errors_file.open("a", encoding="utf-8")
        self.event_count = 0
        self.error_count = 0

    def event(self, stage: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            **payload,
        }
        text = json.dumps(row, ensure_ascii=True, default=str)
        self._events_fp.write(text + "\n")
        self._events_fp.flush()
        self.event_count += 1
        print(f"[LIVE_FULL_FLOW][EVENT] {text}")

    def error(self, stage: str, **payload: Any) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "stage": stage,
            **payload,
        }
        text = json.dumps(row, ensure_ascii=True, default=str)
        self._errors_fp.write(text + "\n")
        self._errors_fp.flush()
        self.error_count += 1
        print(f"[LIVE_FULL_FLOW][ERROR] {text}")

    def close(self) -> None:
        self._events_fp.close()
        self._errors_fp.close()
        print(f"[LIVE_FULL_FLOW][ARTIFACT] events={self.events_file}")
        print(f"[LIVE_FULL_FLOW][ARTIFACT] errors={self.errors_file}")


def _load_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live multi-symbol multi-strategy multi-timeframe full pipeline with real Alpaca order placement + cancel probe."
    )
    parser.add_argument("--hours", type=float, default=4.0, help="Total runtime in hours.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Loop polling interval in seconds.",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Initial 1m bars per symbol for factor warmup.",
    )
    parser.add_argument(
        "--refresh-bars-limit",
        type=int,
        default=5,
        help="Max bars pulled per symbol each cycle.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Broker account mode used in deployment records.",
    )
    parser.add_argument(
        "--trading-base-url",
        default="",
        help="Optional override for Alpaca trading base URL.",
    )
    parser.add_argument(
        "--enable-real-orders",
        action="store_true",
        help="Required flag to actually place/cancel provider orders.",
    )
    parser.add_argument(
        "--skip-cancel-probe",
        action="store_true",
        help="Skip standalone order cancel probe.",
    )
    parser.add_argument(
        "--cancel-probe-every-cycles",
        type=int,
        default=12,
        help="Run cancel probe every N cycles (default 12 with 5s poll ~= 1 minute).",
    )
    parser.add_argument(
        "--cancel-probe-symbol",
        default="BTCUSD",
        help="Symbol for cancel probe.",
    )
    parser.add_argument(
        "--cancel-probe-qty",
        type=float,
        default=0.0002,
        help="Base quantity for cancel probe limit order (auto-adjusted upward if notional < $10).",
    )
    parser.add_argument(
        "--min-bars-for-factors",
        type=int,
        default=8,
        help="Minimum bar count before logging factor snapshot.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/live_paper_trading",
        help="Directory root for JSONL logs.",
    )
    parser.add_argument(
        "--max-errors-before-stop",
        type=int,
        default=200,
        help="Stop early if recorded errors exceed this number.",
    )
    parser.add_argument(
        "--btc-order-qty",
        type=float,
        default=0.0002,
        help="Per-order qty override for BTC symbols.",
    )
    parser.add_argument(
        "--eth-order-qty",
        type=float,
        default=0.01,
        help="Per-order qty override for ETH symbols (must satisfy Alpaca crypto min order notional, typically >= $10).",
    )
    parser.add_argument(
        "--default-order-notional",
        type=float,
        default=12.0,
        help="Default order notional in USD for non-BTC/ETH symbols; qty auto-derived from live quote.",
    )
    parser.add_argument(
        "--capital-per-deployment",
        type=float,
        default=10000.0,
        help="Capital allocated for each deployment.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated crypto symbols, e.g. BTCUSD,ETHUSD,SOLUSD,LTCUSD,DOGEUSD",
    )
    parser.add_argument(
        "--strategy-variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated strategy variants, e.g. trend,mean,breakout",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(DEFAULT_TIMEFRAMES),
        help="Comma-separated strategy timeframes, e.g. 1m,2m,5m",
    )
    parser.add_argument(
        "--close-positions-on-exit",
        action="store_true",
        help="Issue close manual-actions for remaining local positions before stopping.",
    )
    parser.add_argument(
        "--user-email",
        default="2@test.com",
        help="Target backend user email for binding trading data.",
    )
    parser.add_argument(
        "--user-password",
        default="pass1234",
        help="Target backend user password used for login/register fallback.",
    )
    parser.add_argument(
        "--allow-register",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow auto-register when login fails (default: disabled).",
    )
    parser.add_argument(
        "--stream-probe-every-cycles",
        type=int,
        default=12,
        help="Probe deployment SSE stream every N cycles (0 disables stream probe).",
    )
    parser.add_argument(
        "--validate-broker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to validate broker credentials on create (/broker-accounts?validate=...).",
    )
    return parser.parse_args()


def _require_configuration(args: argparse.Namespace) -> None:
    if not args.enable_real_orders:
        raise SystemExit(
            "Refusing to run real order flow without --enable-real-orders flag."
        )
    if not settings.alpaca_api_key.strip() or not settings.alpaca_api_secret.strip():
        raise SystemExit("ALPACA_API_KEY / ALPACA_API_SECRET missing in .env.")
    if args.hours <= 0:
        raise SystemExit("--hours must be > 0.")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0.")
    if args.cancel_probe_qty <= 0:
        raise SystemExit("--cancel-probe-qty must be > 0.")
    if args.btc_order_qty <= 0 or args.eth_order_qty <= 0:
        raise SystemExit("--btc-order-qty and --eth-order-qty must be > 0.")
    if args.default_order_notional <= 0:
        raise SystemExit("--default-order-notional must be > 0.")
    if args.capital_per_deployment <= 0:
        raise SystemExit("--capital-per-deployment must be > 0.")
    if args.stream_probe_every_cycles < 0:
        raise SystemExit("--stream-probe-every-cycles must be >= 0.")
    if not str(args.user_email).strip():
        raise SystemExit("--user-email cannot be empty.")
    if not str(args.user_password).strip():
        raise SystemExit("--user-password cannot be empty.")


def _parse_csv_tokens(raw: str, *, upper: bool = False) -> list[str]:
    tokens: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        token = token.upper() if upper else token.lower()
        if token not in tokens:
            tokens.append(token)
    return tokens


def _validate_timeframes(timeframes: list[str]) -> list[str]:
    normalized: list[str] = []
    for tf in timeframes:
        if not tf.endswith(("m", "h", "d")):
            raise SystemExit(f"Unsupported timeframe '{tf}'. Use values like 1m,2m,5m.")
        step = tf[:-1]
        if not step.isdigit() or int(step) <= 0:
            raise SystemExit(f"Invalid timeframe '{tf}'.")
        normalized.append(tf)
    return normalized


def _configure_runtime_aggregator_for_timeframes(timeframes: list[str]) -> tuple[str, ...]:
    aggregate_timeframes = tuple(tf for tf in timeframes if tf != "1m")
    market_data_runtime._aggregator = BarAggregator(  # noqa: SLF001
        timeframes=aggregate_timeframes or ("5m",),
        timezone=settings.market_data_aggregate_timezone,
    )
    return aggregate_timeframes


def _safe_last_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(val):
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
                        suffix = replacement[len(prefix) :]
                        replacement = f"{new_id}.{suffix}"
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


def _disabled_short_entry_condition() -> dict[str, Any]:
    return {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": 0}}


def _disabled_short_exit_rules(tag: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "signal_exit",
            "name": f"{tag}_short_exit_disabled",
            "condition": _disabled_short_entry_condition(),
        }
    ]


def _high_frequency_payload(symbol: str, variant: str, timeframe: str) -> dict[str, Any]:
    payload = load_strategy_payload(EXAMPLE_PATH)
    payload["strategy"]["name"] = f"{symbol} {variant} live-{timeframe}"
    payload["strategy"]["description"] = (
        f"{symbol} {variant} live {timeframe} long-only strategy for e2e runtime."
    )
    payload["universe"] = {"market": "crypto", "tickers": [symbol]}
    payload["timeframe"] = timeframe

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
        payload["trade"]["short"]["entry"]["condition"] = _disabled_short_entry_condition()
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
        payload["trade"]["short"]["exits"] = _disabled_short_exit_rules("trend")
        return payload

    if variant == "mean":
        payload["trade"]["long"]["entry"]["condition"] = {
            "all": [
                {"cmp": {"left": {"ref": "ema_3"}, "op": "lt", "right": {"ref": "ema_6"}}},
                {"cmp": {"left": {"ref": "rsi_4"}, "op": "lt", "right": 45}},
            ]
        }
        payload["trade"]["short"]["entry"]["condition"] = _disabled_short_entry_condition()
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
        payload["trade"]["short"]["exits"] = _disabled_short_exit_rules("mean")
        return payload

    if variant == "breakout":
        payload["trade"]["long"]["entry"]["condition"] = {
            "all": [
                {"cmp": {"left": {"ref": "price.close"}, "op": "gt", "right": {"ref": "ema_3"}}},
                {
                    "cmp": {
                        "left": {"ref": "macd_3_7_3.histogram"},
                        "op": "gt",
                        "right": 0,
                    }
                },
                {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 40}},
            ]
        }
        payload["trade"]["short"]["entry"]["condition"] = _disabled_short_entry_condition()
        payload["trade"]["long"]["exits"] = [
            {
                "type": "signal_exit",
                "name": "breakout_long_exit",
                "condition": {
                    "any": [
                        {"cmp": {"left": {"ref": "price.close"}, "op": "lt", "right": {"ref": "ema_6"}}},
                        {"cmp": {"left": {"ref": "rsi_4"}, "op": "gt", "right": 78}},
                    ]
                },
            }
        ]
        payload["trade"]["short"]["exits"] = _disabled_short_exit_rules("breakout")
        return payload

    raise ValueError(f"Unsupported strategy variant: {variant}")
    return payload


def _register_and_get_token(client: TestClient, recorder: FlowRecorder) -> str:
    email = f"live_full_flow_{uuid4().hex}@test.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "pass1234", "name": "Live Full Flow User"},
    )
    recorder.event("auth_register", status=response.status_code, email=email)
    if response.status_code != 201:
        raise RuntimeError(f"register failed: {response.status_code} {response.text}")
    return response.json()["access_token"]


def _login_or_register_and_get_token(
    client: TestClient,
    *,
    email: str,
    password: str,
    allow_register: bool,
    recorder: FlowRecorder,
) -> str:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    recorder.event("auth_login", status=login.status_code, email=email)
    if login.status_code == 200:
        return str(login.json()["access_token"])
    if not allow_register:
        raise RuntimeError(
            "login failed and auto-register is disabled. "
            f"email={email} status={login.status_code} body={login.text}"
        )

    register = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Live Full Flow User"},
    )
    recorder.event("auth_register", status=register.status_code, email=email)
    if register.status_code == 201:
        return str(register.json()["access_token"])

    raise RuntimeError(
        "auth failed for configured user "
        f"{email}: login={login.status_code}, register={register.status_code}",
    )


def _create_broker_account(
    client: TestClient,
    *,
    headers: dict[str, str],
    mode: str,
    trading_base_url: str,
    validate: bool,
    recorder: FlowRecorder,
) -> str:
    listing = client.get("/api/v1/broker-accounts", headers=headers)
    recorder.event("broker_account_list", status=listing.status_code)
    if listing.status_code != 200:
        raise RuntimeError(f"broker account list failed: {listing.status_code} {listing.text}")
    listed_rows = listing.json()
    if isinstance(listed_rows, list):
        for row in listed_rows:
            if not isinstance(row, dict):
                continue
            if row.get("provider") != "alpaca":
                continue
            if row.get("mode") != mode:
                continue
            if row.get("status") != "active":
                continue
            if row.get("last_validated_status") != "paper_probe_ok":
                continue
            broker_account_id = row.get("broker_account_id")
            if broker_account_id:
                recorder.event(
                    "broker_account_reuse",
                    broker_account_id=broker_account_id,
                    validation_status=row.get("last_validated_status"),
                )
                return str(broker_account_id)

    credentials: dict[str, Any] = {
        "api_key": settings.alpaca_api_key,
        "api_secret": settings.alpaca_api_secret,
    }
    if trading_base_url.strip():
        credentials["trading_base_url"] = trading_base_url.strip()

    response = client.post(
        "/api/v1/broker-accounts",
        headers=headers,
        params={"validate": validate},
        json={
            "provider": "alpaca",
            "mode": mode,
            "credentials": credentials,
            "metadata": {"label": "live-full-flow"},
        },
    )
    recorder.event("broker_account_create", status=response.status_code, mode=mode, validate=validate)
    if response.status_code != 201:
        raise RuntimeError(f"broker account create failed: {response.status_code} {response.text}")
    return str(response.json()["broker_account_id"])


def _subscribe_symbols(
    client: TestClient,
    *,
    headers: dict[str, str],
    symbols: list[str],
    recorder: FlowRecorder,
) -> None:
    response = client.post(
        "/api/v1/market-data/subscriptions",
        headers=headers,
        params={"market": "crypto"},
        json={"symbols": symbols},
    )
    recorder.event(
        "market_subscription",
        status=response.status_code,
        body=response.json() if response.status_code == 200 else response.text,
    )
    if response.status_code != 200:
        raise RuntimeError(f"market subscription failed: {response.status_code} {response.text}")


async def _build_symbol_plans(
    *,
    symbols: list[str],
    btc_order_qty: float,
    eth_order_qty: float,
    default_order_notional: float,
    capital_per_deployment: float,
    quote_provider: AlpacaTradingAdapter,
    recorder: FlowRecorder,
) -> list[SymbolPlan]:
    default_notional = Decimal(str(default_order_notional))
    plans: list[SymbolPlan] = []
    for symbol in symbols:
        upper_symbol = symbol.strip().upper()
        if upper_symbol == "BTCUSD":
            qty = Decimal(str(btc_order_qty))
        elif upper_symbol == "ETHUSD":
            qty = Decimal(str(eth_order_qty))
        else:
            quote = await quote_provider.fetch_latest_quote(upper_symbol)
            if quote is None:
                raise RuntimeError(f"Unable to fetch quote for {upper_symbol} when deriving order qty.")
            reference = quote.last
            if reference is None and quote.bid is not None and quote.ask is not None:
                reference = (quote.bid + quote.ask) / Decimal("2")
            if reference is None or reference <= 0:
                raise RuntimeError(f"Invalid quote price for {upper_symbol} when deriving order qty.")
            qty = (default_notional / reference).quantize(Decimal("0.00000001"), rounding=ROUND_UP)

        if qty <= 0:
            raise RuntimeError(f"Derived non-positive qty for {upper_symbol}.")
        plans.append(
            SymbolPlan(
                symbol=upper_symbol,
                order_qty=float(qty),
                capital_allocated=float(capital_per_deployment),
            )
        )
        recorder.event(
            "symbol_plan",
            symbol=upper_symbol,
            order_qty=float(qty),
            capital_allocated=float(capital_per_deployment),
        )
    return plans


def _create_deployments(
    client: TestClient,
    *,
    headers: dict[str, str],
    broker_account_id: str,
    symbol_plans: list[SymbolPlan],
    strategy_variants: list[str],
    timeframes: list[str],
    mode: str,
    recorder: FlowRecorder,
) -> list[DeploymentPlan]:
    plans: list[DeploymentPlan] = []
    presets: list[tuple[str, str, str, float, float]] = []
    for plan in symbol_plans:
        for timeframe in timeframes:
            for variant in strategy_variants:
                presets.append((plan.symbol, variant, timeframe, plan.order_qty, plan.capital_allocated))

    for symbol, variant, timeframe, order_qty, capital in presets:
        payload = _high_frequency_payload(symbol, variant, timeframe)
        thread = client.post("/api/v1/chat/new-thread", headers=headers, json={"metadata": {}})
        recorder.event(
            "chat_new_thread",
            symbol=symbol,
            variant=variant,
            timeframe=timeframe,
            status=thread.status_code,
        )
        if thread.status_code != 201:
            raise RuntimeError(f"new thread failed: {thread.status_code} {thread.text}")
        session_id = thread.json()["session_id"]

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
            timeframe=timeframe,
            status=confirm.status_code,
        )
        if confirm.status_code != 200:
            raise RuntimeError(f"strategy confirm failed: {confirm.status_code} {confirm.text}")
        strategy_id = confirm.json()["strategy_id"]

        deployment = client.post(
            "/api/v1/deployments",
            headers=headers,
            json={
                "strategy_id": strategy_id,
                "broker_account_id": broker_account_id,
                "mode": mode,
                "capital_allocated": str(capital),
                "risk_limits": {
                    "order_qty": order_qty,
                    "max_position_notional": 5000,
                    "max_symbol_exposure_pct": 0.95,
                },
                "runtime_state": {"source": "live_full_flow_script"},
            },
        )
        recorder.event(
            "deployment_create",
            symbol=symbol,
            variant=variant,
            timeframe=timeframe,
            status=deployment.status_code,
        )
        if deployment.status_code != 201:
            raise RuntimeError(f"deployment create failed: {deployment.status_code} {deployment.text}")
        deployment_id = UUID(deployment.json()["deployment_id"])

        start = client.post(f"/api/v1/deployments/{deployment_id}/start", headers=headers)
        recorder.event(
            "deployment_start",
            symbol=symbol,
            variant=variant,
            timeframe=timeframe,
            deployment_id=str(deployment_id),
            status=start.status_code,
        )
        if start.status_code != 200:
            raise RuntimeError(f"deployment start failed: {start.status_code} {start.text}")

        plans.append(
            DeploymentPlan(
                symbol=symbol,
                variant=variant,
                timeframe=timeframe,
                deployment_id=deployment_id,
                order_qty=order_qty,
                strategy_payload=payload,
                parsed_strategy=parse_strategy_payload(payload),
            )
        )

    return plans


async def _warmup_symbol(
    provider: AlpacaRestProvider,
    *,
    symbol: str,
    warmup_bars: int,
    recorder: FlowRecorder,
) -> datetime | None:
    since = datetime.now(UTC) - timedelta(minutes=max(5, warmup_bars + 15))
    bars = await provider.fetch_recent_1m_bars(
        symbol=symbol,
        market="crypto",
        since=since,
        limit=max(2, warmup_bars),
    )
    latest_ts: datetime | None = None
    for bar in sorted(bars, key=lambda item: item.timestamp):
        market_data_runtime.ingest_1m_bar(market="crypto", symbol=symbol, bar=bar)
        latest_ts = bar.timestamp

    quote = None
    try:
        quote = await provider.fetch_quote(symbol=symbol, market="crypto")
    except Exception as exc:  # noqa: BLE001
        recorder.error(
            "warmup_quote_error",
            symbol=symbol,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    if quote is not None:
        market_data_runtime.upsert_quote(market="crypto", symbol=symbol, quote=quote)

    recorder.event(
        "warmup_data_complete",
        symbol=symbol,
        bars_loaded=len(bars),
        last_bar_time=latest_ts.isoformat() if latest_ts else None,
        quote_time=quote.timestamp.isoformat() if quote else None,
        quote_last=float(quote.last) if quote and quote.last is not None else None,
    )
    return latest_ts


async def _refresh_symbol_live_data(
    provider: AlpacaRestProvider,
    *,
    symbol: str,
    last_bar_ts: datetime | None,
    refresh_bars_limit: int,
    cycle: int,
    recorder: FlowRecorder,
) -> tuple[datetime | None, int]:
    since = (last_bar_ts - timedelta(minutes=2)) if last_bar_ts else (datetime.now(UTC) - timedelta(minutes=3))
    bars = await provider.fetch_recent_1m_bars(
        symbol=symbol,
        market="crypto",
        since=since,
        limit=max(2, refresh_bars_limit),
    )
    new_bars = 0
    latest_ts = last_bar_ts
    for bar in sorted(bars, key=lambda item: item.timestamp):
        if latest_ts is not None and bar.timestamp <= latest_ts:
            continue
        market_data_runtime.ingest_1m_bar(market="crypto", symbol=symbol, bar=bar)
        latest_ts = bar.timestamp
        new_bars += 1
        recorder.event(
            "live_data_bar_ingest",
            cycle=cycle,
            symbol=symbol,
            timestamp=bar.timestamp.isoformat(),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )

    quote = None
    try:
        quote = await provider.fetch_quote(symbol=symbol, market="crypto")
    except Exception as exc:  # noqa: BLE001
        recorder.error(
            "live_data_quote_refresh_error",
            cycle=cycle,
            symbol=symbol,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    if quote is not None:
        market_data_runtime.upsert_quote(market="crypto", symbol=symbol, quote=quote)

    recorder.event(
        "live_data_refresh",
        cycle=cycle,
        symbol=symbol,
        fetched_bars=len(bars),
        ingested_bars=new_bars,
        last_bar_time=latest_ts.isoformat() if latest_ts else None,
        quote_time=quote.timestamp.isoformat() if quote else None,
        bid=float(quote.bid) if quote and quote.bid is not None else None,
        ask=float(quote.ask) if quote and quote.ask is not None else None,
        last=float(quote.last) if quote and quote.last is not None else None,
    )
    return latest_ts, new_bars


def _bars_to_frame(bar_rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [float(row["open"]) for row in bar_rows],
            "high": [float(row["high"]) for row in bar_rows],
            "low": [float(row["low"]) for row in bar_rows],
            "close": [float(row["close"]) for row in bar_rows],
            "volume": [float(row["volume"]) for row in bar_rows],
        },
        index=pd.to_datetime([row["timestamp"] for row in bar_rows], utc=True),
    )


def _to_decimal_qty(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def _to_decimal_price(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_DOWN)


def _resolve_reference_price(symbol: str) -> Decimal | None:
    quote = market_data_runtime.get_latest_quote(market="crypto", symbol=symbol)
    if quote is not None:
        if quote.last is not None and quote.last > 0:
            return Decimal(str(quote.last))
        if quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask > 0:
            return (Decimal(str(quote.bid)) + Decimal(str(quote.ask))) / Decimal("2")
        if quote.bid is not None and quote.bid > 0:
            return Decimal(str(quote.bid))
        if quote.ask is not None and quote.ask > 0:
            return Decimal(str(quote.ask))

    bars = market_data_runtime.get_recent_bars(
        market="crypto",
        symbol=symbol,
        timeframe="1m",
        limit=1,
    )
    if not bars:
        return None
    close = Decimal(str(bars[-1].close))
    return close if close > 0 else None


def _validate_min_notional_config(*, deployments: list[DeploymentPlan], recorder: FlowRecorder) -> None:
    for plan in deployments:
        qty = Decimal(str(plan.order_qty))
        if qty <= 0:
            raise RuntimeError(f"invalid non-positive order_qty for {plan.symbol}/{plan.variant}: {plan.order_qty}")

        ref_price = _resolve_reference_price(plan.symbol)
        if ref_price is None:
            recorder.error(
                "order_qty_notional_validation_missing_price",
                symbol=plan.symbol,
                variant=plan.variant,
                timeframe=plan.timeframe,
                order_qty=float(qty),
            )
            continue

        estimated_notional = qty * ref_price
        recorder.event(
            "order_qty_notional_validation",
            symbol=plan.symbol,
            variant=plan.variant,
            timeframe=plan.timeframe,
            order_qty=float(qty),
            reference_price=float(ref_price),
            estimated_notional=float(estimated_notional),
            minimum_notional=float(ALPACA_CRYPTO_MIN_ORDER_NOTIONAL),
        )
        if estimated_notional < ALPACA_CRYPTO_MIN_ORDER_NOTIONAL:
            raise RuntimeError(

                    f"{plan.symbol} {plan.variant} {plan.timeframe} order_qty too small for Alpaca crypto minimum order notional: "
                    f"qty={qty} price={ref_price} estimated_notional={estimated_notional} "
                    f"< {ALPACA_CRYPTO_MIN_ORDER_NOTIONAL}. Increase qty settings."

            )


async def _run_cancel_probe(
    adapter: AlpacaTradingAdapter,
    *,
    symbol: str,
    qty: Decimal,
    cycle: int,
    recorder: FlowRecorder,
) -> None:
    quote = await adapter.fetch_latest_quote(symbol)
    if quote is None or quote.last is None:
        recorder.error(
            "cancel_probe_quote_missing",
            cycle=cycle,
            symbol=symbol,
        )
        return

    limit_price = _to_decimal_price(quote.last * Decimal("0.70"))
    if limit_price <= 0:
        recorder.error(
            "cancel_probe_invalid_limit_price",
            cycle=cycle,
            symbol=symbol,
            quote_last=float(quote.last),
        )
        return

    min_probe_qty = (ALPACA_CRYPTO_MIN_ORDER_NOTIONAL / limit_price).quantize(
        Decimal("0.00000001"),
        rounding=ROUND_UP,
    )
    effective_qty = qty if (qty * limit_price) >= ALPACA_CRYPTO_MIN_ORDER_NOTIONAL else min_probe_qty
    if effective_qty > qty:
        recorder.event(
            "cancel_probe_qty_adjusted",
            cycle=cycle,
            symbol=symbol,
            original_qty=float(qty),
            adjusted_qty=float(effective_qty),
            limit_price=float(limit_price),
            minimum_notional=float(ALPACA_CRYPTO_MIN_ORDER_NOTIONAL),
        )

    intent = OrderIntent(
        client_order_id=f"cancel-probe-{uuid4().hex[:20]}",
        symbol=symbol,
        side="buy",
        qty=effective_qty,
        order_type="limit",
        limit_price=limit_price,
        time_in_force="gtc",
        metadata={"source": "live_full_flow_cancel_probe"},
    )
    order = await adapter.submit_order(intent)
    recorder.event(
        "cancel_probe_submit",
        cycle=cycle,
        symbol=symbol,
        provider_order_id=order.provider_order_id,
        status=order.status,
        qty=float(order.qty),
        limit_price=float(limit_price),
    )

    await asyncio.sleep(1.0)
    latest = await adapter.fetch_order(order.provider_order_id)
    if latest is None:
        recorder.error(
            "cancel_probe_fetch_missing",
            cycle=cycle,
            symbol=symbol,
            provider_order_id=order.provider_order_id,
        )
        return

    status = latest.status.strip().lower()
    if status in TERMINAL_ORDER_STATUSES:
        recorder.event(
            "cancel_probe_terminal_before_cancel",
            cycle=cycle,
            symbol=symbol,
            provider_order_id=latest.provider_order_id,
            status=latest.status,
        )
        return

    canceled = await adapter.cancel_order(latest.provider_order_id)
    recorder.event(
        "cancel_probe_cancel_request",
        cycle=cycle,
        symbol=symbol,
        provider_order_id=latest.provider_order_id,
        canceled=bool(canceled),
    )
    await asyncio.sleep(0.6)
    final_state = await adapter.fetch_order(latest.provider_order_id)
    recorder.event(
        "cancel_probe_post_cancel_state",
        cycle=cycle,
        symbol=symbol,
        provider_order_id=latest.provider_order_id,
        status=final_state.status if final_state else None,
    )


def _collect_endpoint_count(
    client: TestClient,
    *,
    url: str,
    headers: dict[str, str],
    recorder: FlowRecorder,
    stage: str,
    deployment_id: UUID,
    symbol: str,
    variant: str,
    timeframe: str,
) -> int:
    response = client.get(url, headers=headers)
    if response.status_code != 200:
        recorder.error(
            f"{stage}_error",
            deployment_id=str(deployment_id),
            symbol=symbol,
            variant=variant,
            timeframe=timeframe,
            status=response.status_code,
            body=response.text,
        )
        return 0
    payload = response.json()
    count = len(payload) if isinstance(payload, list) else 1
    recorder.event(
        stage,
        deployment_id=str(deployment_id),
        symbol=symbol,
        variant=variant,
        timeframe=timeframe,
        count=count,
    )
    return count


def _collect_order_diagnostics(
    client: TestClient,
    *,
    deployment_id: UUID,
    headers: dict[str, str],
    recorder: FlowRecorder,
    symbol: str,
    variant: str,
    timeframe: str,
) -> tuple[dict[str, int], int]:
    response = client.get(f"/api/v1/deployments/{deployment_id}/orders", headers=headers)
    if response.status_code != 200:
        recorder.error(
            "orders_diagnostics_error",
            deployment_id=str(deployment_id),
            symbol=symbol,
            variant=variant,
            timeframe=timeframe,
            status=response.status_code,
            body=response.text,
        )
        return {}, 0

    rows = response.json()
    if not isinstance(rows, list):
        return {}, 0

    status_breakdown: dict[str, int] = {}
    transition_events = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip().lower() or "unknown"
        status_breakdown[status] = status_breakdown.get(status, 0) + 1
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            transitions = metadata.get("state_transitions")
            if isinstance(transitions, list):
                transition_events += len(transitions)

    recorder.event(
        "orders_diagnostics",
        deployment_id=str(deployment_id),
        symbol=symbol,
        variant=variant,
        timeframe=timeframe,
        status_breakdown=status_breakdown,
        transition_events=transition_events,
    )
    return status_breakdown, transition_events


def _close_positions_if_requested(
    client: TestClient,
    *,
    headers: dict[str, str],
    deployments: list[DeploymentPlan],
    recorder: FlowRecorder,
) -> None:
    for plan in deployments:
        positions_resp = client.get(
            f"/api/v1/deployments/{plan.deployment_id}/positions",
            headers=headers,
        )
        if positions_resp.status_code != 200:
            recorder.error(
                "close_positions_fetch_error",
                deployment_id=str(plan.deployment_id),
                status=positions_resp.status_code,
                body=positions_resp.text,
            )
            continue
        rows = positions_resp.json()
        if not isinstance(rows, list):
            continue
        for row in rows:
            try:
                qty = float(row.get("qty", 0.0))
            except Exception:  # noqa: BLE001
                qty = 0.0
            side = str(row.get("side", "")).strip().lower()
            symbol = str(row.get("symbol", "")).strip().upper()
            if qty <= 0 or side not in {"long", "short"}:
                continue
            close_resp = client.post(
                f"/api/v1/deployments/{plan.deployment_id}/manual-actions",
                headers=headers,
                json={"action": "close", "payload": {"symbol": symbol, "qty": qty}},
            )
            recorder.event(
                "close_position_on_exit",
                deployment_id=str(plan.deployment_id),
                symbol=symbol,
                qty=qty,
                status=close_resp.status_code,
                body=close_resp.json() if close_resp.status_code == 200 else close_resp.text,
            )


def _stop_deployments(
    client: TestClient,
    *,
    headers: dict[str, str],
    deployments: list[DeploymentPlan],
    recorder: FlowRecorder,
) -> None:
    for plan in deployments:
        stop_resp = client.post(f"/api/v1/deployments/{plan.deployment_id}/stop", headers=headers)
        recorder.event(
            "deployment_stop",
            deployment_id=str(plan.deployment_id),
            symbol=plan.symbol,
            variant=plan.variant,
            timeframe=plan.timeframe,
            status=stop_resp.status_code,
        )


def _probe_stream_once(
    client: TestClient,
    *,
    headers: dict[str, str],
    deployment_id: UUID,
    cycle: int,
    recorder: FlowRecorder,
) -> None:
    response = client.get(
        f"/api/v1/stream/deployments/{deployment_id}",
        headers=headers,
        params={"max_events": 8, "poll_seconds": 0.2},
    )
    if response.status_code != 200:
        recorder.error(
            "stream_probe_http_error",
            cycle=cycle,
            deployment_id=str(deployment_id),
            status=response.status_code,
            body=response.text,
        )
        return
    text = response.text
    required = (
        "event: deployment_status",
        "event: order_update",
        "event: fill_update",
        "event: position_update",
        "event: pnl_update",
        "event: heartbeat",
    )
    missing = [event for event in required if event not in text]
    if missing:
        recorder.error(
            "stream_probe_missing_events",
            cycle=cycle,
            deployment_id=str(deployment_id),
            missing=missing,
        )
        return
    recorder.event(
        "stream_probe_ok",
        cycle=cycle,
        deployment_id=str(deployment_id),
        bytes=len(text.encode("utf-8")),
    )


async def run_live_full_flow(args: argparse.Namespace) -> None:
    _require_configuration(args)
    symbols = _parse_csv_tokens(args.symbols, upper=True)
    strategy_variants = _parse_csv_tokens(args.strategy_variants, upper=False)
    timeframes = _validate_timeframes(_parse_csv_tokens(args.timeframes, upper=False))
    if not symbols:
        raise SystemExit("No symbols configured. Use --symbols.")
    if not strategy_variants:
        raise SystemExit("No strategy variants configured. Use --strategy-variants.")
    if not timeframes:
        raise SystemExit("No timeframes configured. Use --timeframes.")
    invalid_variants = [item for item in strategy_variants if item not in {"trend", "mean", "breakout"}]
    if invalid_variants:
        raise SystemExit(
            f"Unsupported strategy variants: {invalid_variants}. Allowed: trend, mean, breakout."
        )
    aggregate_timeframes = _configure_runtime_aggregator_for_timeframes(timeframes)

    settings.paper_trading_execute_orders = True
    settings.paper_trading_enqueue_on_start = False
    settings.paper_trading_enabled = True
    settings.paper_trading_kill_switch_global = False

    recorder = FlowRecorder(Path(args.artifact_dir).resolve())
    recorder.event(
        "config",
        hours=args.hours,
        poll_seconds=args.poll_seconds,
        mode=args.mode,
        trading_base_url=args.trading_base_url or settings.alpaca_trading_base_url,
        warmup_bars=args.warmup_bars,
        refresh_bars_limit=args.refresh_bars_limit,
        cancel_probe_enabled=not args.skip_cancel_probe,
        cancel_probe_every_cycles=args.cancel_probe_every_cycles,
        symbols=symbols,
        strategy_variants=strategy_variants,
        timeframes=timeframes,
        aggregate_timeframes=list(aggregate_timeframes),
        deployment_target_count=len(symbols) * len(strategy_variants) * len(timeframes),
        default_order_notional=args.default_order_notional,
        paper_trading_execute_orders=settings.paper_trading_execute_orders,
    )

    signal_store.clear()
    market_data_runtime.reset()

    trading_base_url = args.trading_base_url.strip() or settings.alpaca_trading_base_url
    probe_adapter = AlpacaTradingAdapter(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        trading_base_url=trading_base_url,
    )
    data_provider = AlpacaRestProvider()

    deployments: list[DeploymentPlan] = []
    token = ""
    stats_orders: dict[UUID, int] = {}
    stats_signals: dict[UUID, dict[str, int]] = {}
    start_monotonic = time.monotonic()
    deadline = start_monotonic + (args.hours * 3600.0)

    try:
        account = await probe_adapter.fetch_account_state()
        recorder.event(
            "preflight_account_state",
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            margin_used=float(account.margin_used),
        )

        with TestClient(app) as client:
            token = _login_or_register_and_get_token(
                client,
                email=str(args.user_email).strip(),
                password=str(args.user_password).strip(),
                allow_register=bool(args.allow_register),
                recorder=recorder,
            )
            headers = {"Authorization": f"Bearer {token}"}
            broker_account_id = _create_broker_account(
                client,
                headers=headers,
                mode=args.mode,
                trading_base_url=trading_base_url,
                validate=args.validate_broker,
                recorder=recorder,
            )
            _subscribe_symbols(client, headers=headers, symbols=symbols, recorder=recorder)
            symbol_plans = await _build_symbol_plans(
                symbols=symbols,
                btc_order_qty=args.btc_order_qty,
                eth_order_qty=args.eth_order_qty,
                default_order_notional=args.default_order_notional,
                capital_per_deployment=args.capital_per_deployment,
                quote_provider=probe_adapter,
                recorder=recorder,
            )

            deployments = _create_deployments(
                client,
                headers=headers,
                broker_account_id=broker_account_id,
                symbol_plans=symbol_plans,
                strategy_variants=strategy_variants,
                timeframes=timeframes,
                mode=args.mode,
                recorder=recorder,
            )
            stats_orders = {plan.deployment_id: 0 for plan in deployments}
            stats_signals = {plan.deployment_id: {} for plan in deployments}

            latest_by_symbol: dict[str, datetime | None] = {}
            for symbol in symbols:
                try:
                    latest_by_symbol[symbol] = await _warmup_symbol(
                        data_provider,
                        symbol=symbol,
                        warmup_bars=args.warmup_bars,
                        recorder=recorder,
                    )
                except Exception as exc:  # noqa: BLE001
                    latest_by_symbol[symbol] = None
                    recorder.error(
                        "warmup_error",
                        symbol=symbol,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

            _validate_min_notional_config(deployments=deployments, recorder=recorder)
            last_processed_bar_by_deployment: dict[UUID, str] = {}

            cycle = 0
            while time.monotonic() < deadline:
                cycle += 1
                cycle_started = time.monotonic()
                new_bars_by_symbol: dict[str, int] = {}

                for symbol in symbols:
                    try:
                        latest_ts, ingested = await _refresh_symbol_live_data(
                            data_provider,
                            symbol=symbol,
                            last_bar_ts=latest_by_symbol.get(symbol),
                            refresh_bars_limit=args.refresh_bars_limit,
                            cycle=cycle,
                            recorder=recorder,
                        )
                        latest_by_symbol[symbol] = latest_ts
                        new_bars_by_symbol[symbol] = ingested
                    except Exception as exc:  # noqa: BLE001
                        new_bars_by_symbol[symbol] = 0
                        recorder.error(
                            "live_data_refresh_error",
                            cycle=cycle,
                            symbol=symbol,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

                any_new_bar = any(count > 0 for count in new_bars_by_symbol.values())
                if not any_new_bar:
                    recorder.event("cycle_no_new_bar", cycle=cycle)

                if any_new_bar:
                    for plan in deployments:
                        if new_bars_by_symbol.get(plan.symbol, 0) <= 0:
                            continue
                        bars_resp = client.get(
                            "/api/v1/market-data/bars",
                            headers=headers,
                            params={
                                "market": "crypto",
                                "symbol": plan.symbol,
                                "timeframe": plan.timeframe,
                                "limit": 200,
                            },
                        )
                        if bars_resp.status_code != 200:
                            recorder.error(
                                "market_bars_read_error",
                                cycle=cycle,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                timeframe=plan.timeframe,
                                status=bars_resp.status_code,
                                body=bars_resp.text,
                            )
                            continue

                        bars = bars_resp.json().get("bars", [])
                        if not bars:
                            recorder.event(
                                "timeframe_no_data",
                                cycle=cycle,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                timeframe=plan.timeframe,
                            )
                            continue
                        latest_bar_ts = str(bars[-1].get("timestamp"))
                        if last_processed_bar_by_deployment.get(plan.deployment_id) == latest_bar_ts:
                            continue
                        last_processed_bar_by_deployment[plan.deployment_id] = latest_bar_ts

                        if len(bars) >= args.min_bars_for_factors:
                            try:
                                frame = _bars_to_frame(bars)
                                enriched = prepare_backtest_frame(
                                    frame,
                                    strategy=plan.parsed_strategy,
                                )
                                snapshot = _factor_snapshot(enriched, plan.strategy_payload)
                                recorder.event(
                                    "dsl_factor_calculation",
                                    cycle=cycle,
                                    deployment_id=str(plan.deployment_id),
                                    symbol=plan.symbol,
                                    variant=plan.variant,
                                    timeframe=plan.timeframe,
                                    bar_count=len(bars),
                                    factors=snapshot,
                                )
                            except Exception as exc:  # noqa: BLE001
                                recorder.error(
                                    "dsl_factor_calculation_error",
                                    cycle=cycle,
                                    deployment_id=str(plan.deployment_id),
                                    symbol=plan.symbol,
                                    variant=plan.variant,
                                    timeframe=plan.timeframe,
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
                                cycle=cycle,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                timeframe=plan.timeframe,
                                status=process.status_code,
                                body=process.text,
                            )
                            continue

                        body = process.json()
                        signal = str(body.get("signal", ""))
                        stats_signals[plan.deployment_id][signal] = (
                            stats_signals[plan.deployment_id].get(signal, 0) + 1
                        )
                        order_id = body.get("order_id")
                        if order_id:
                            stats_orders[plan.deployment_id] += 1
                        recorder.event(
                            "signal_process",
                            cycle=cycle,
                            deployment_id=str(plan.deployment_id),
                            symbol=plan.symbol,
                            variant=plan.variant,
                            timeframe=plan.timeframe,
                            signal=signal,
                            reason=body.get("reason"),
                            order_id=order_id,
                            idempotent_hit=body.get("idempotent_hit"),
                        )
                        reason = str(body.get("reason") or "")
                        if reason.startswith("order_submit_failed:"):
                            recorder.error(
                                "signal_order_submit_failed",
                                cycle=cycle,
                                deployment_id=str(plan.deployment_id),
                                symbol=plan.symbol,
                                variant=plan.variant,
                                timeframe=plan.timeframe,
                                signal=signal,
                                reason=reason,
                            )

                if (
                    not args.skip_cancel_probe
                    and args.cancel_probe_every_cycles > 0
                    and cycle % args.cancel_probe_every_cycles == 0
                ):
                    try:
                        await _run_cancel_probe(
                            probe_adapter,
                            symbol=args.cancel_probe_symbol.strip().upper(),
                            qty=_to_decimal_qty(args.cancel_probe_qty),
                            cycle=cycle,
                            recorder=recorder,
                        )
                    except Exception as exc:  # noqa: BLE001
                        recorder.error(
                            "cancel_probe_error",
                            cycle=cycle,
                            symbol=args.cancel_probe_symbol,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )

                if (
                    args.stream_probe_every_cycles > 0
                    and cycle % args.stream_probe_every_cycles == 0
                    and deployments
                ):
                    _probe_stream_once(
                        client,
                        headers=headers,
                        deployment_id=deployments[0].deployment_id,
                        cycle=cycle,
                        recorder=recorder,
                    )

                if recorder.error_count >= args.max_errors_before_stop:
                    recorder.error(
                        "stop_due_to_error_threshold",
                        cycle=cycle,
                        max_errors=args.max_errors_before_stop,
                    )
                    break

                elapsed = time.monotonic() - cycle_started
                sleep_seconds = max(0.0, args.poll_seconds - elapsed)
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

            total_runtime_seconds = time.monotonic() - start_monotonic
            total_orders = sum(stats_orders.values())
            total_fills = 0
            total_signals = 0
            total_order_transition_events = 0
            order_status_by_deployment: dict[str, dict[str, int]] = {}
            for plan in deployments:
                orders_count = _collect_endpoint_count(
                    client,
                    url=f"/api/v1/deployments/{plan.deployment_id}/orders",
                    headers=headers,
                    recorder=recorder,
                    stage="orders_summary",
                    deployment_id=plan.deployment_id,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                status_breakdown, transition_events = _collect_order_diagnostics(
                    client,
                    deployment_id=plan.deployment_id,
                    headers=headers,
                    recorder=recorder,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                fills_count = _collect_endpoint_count(
                    client,
                    url=f"/api/v1/deployments/{plan.deployment_id}/fills",
                    headers=headers,
                    recorder=recorder,
                    stage="fills_summary",
                    deployment_id=plan.deployment_id,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                _collect_endpoint_count(
                    client,
                    url=f"/api/v1/deployments/{plan.deployment_id}/positions",
                    headers=headers,
                    recorder=recorder,
                    stage="positions_summary",
                    deployment_id=plan.deployment_id,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                _collect_endpoint_count(
                    client,
                    url=f"/api/v1/deployments/{plan.deployment_id}/pnl",
                    headers=headers,
                    recorder=recorder,
                    stage="pnl_summary",
                    deployment_id=plan.deployment_id,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                signals_count = _collect_endpoint_count(
                    client,
                    url=f"/api/v1/deployments/{plan.deployment_id}/signals?limit=1000",
                    headers=headers,
                    recorder=recorder,
                    stage="signals_summary",
                    deployment_id=plan.deployment_id,
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                )
                external_runtime_state = await runtime_state_store.get(plan.deployment_id)
                recorder.event(
                    "runtime_state_snapshot",
                    deployment_id=str(plan.deployment_id),
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                    runtime_state=external_runtime_state if isinstance(external_runtime_state, dict) else {},
                )
                recorder.event(
                    "deployment_runtime_stats",
                    deployment_id=str(plan.deployment_id),
                    symbol=plan.symbol,
                    variant=plan.variant,
                    timeframe=plan.timeframe,
                    runtime_orders=stats_orders[plan.deployment_id],
                    orders_endpoint_count=orders_count,
                    fills_endpoint_count=fills_count,
                    signals_endpoint_count=signals_count,
                    signal_breakdown=stats_signals[plan.deployment_id],
                    order_status_breakdown=status_breakdown,
                    order_transition_events=transition_events,
                )
                total_fills += fills_count
                total_signals += signals_count
                total_order_transition_events += transition_events
                order_status_by_deployment[str(plan.deployment_id)] = status_breakdown

            orders_by_symbol: dict[str, int] = {}
            orders_by_timeframe: dict[str, int] = {}
            for plan in deployments:
                count = stats_orders[plan.deployment_id]
                orders_by_symbol[plan.symbol] = orders_by_symbol.get(plan.symbol, 0) + count
                orders_by_timeframe[plan.timeframe] = orders_by_timeframe.get(plan.timeframe, 0) + count

            recorder.event(
                "final_summary",
                runtime_seconds=round(total_runtime_seconds, 3),
                deployment_count=len(deployments),
                total_orders_from_runtime=total_orders,
                total_fills_endpoint=total_fills,
                total_signals_endpoint=total_signals,
                orders_by_deployment={str(k): v for k, v in stats_orders.items()},
                signals_by_deployment={str(k): v for k, v in stats_signals.items()},
                orders_by_symbol=orders_by_symbol,
                orders_by_timeframe=orders_by_timeframe,
                order_status_by_deployment=order_status_by_deployment,
                total_order_transition_events=total_order_transition_events,
                errors=recorder.error_count,
                events=recorder.event_count,
            )

            if args.close_positions_on_exit:
                _close_positions_if_requested(
                    client,
                    headers=headers,
                    deployments=deployments,
                    recorder=recorder,
                )
            _stop_deployments(client, headers=headers, deployments=deployments, recorder=recorder)
    finally:
        await data_provider.aclose()
        await probe_adapter.aclose()
        recorder.close()


def main() -> None:
    _load_env()
    args = _parse_args()
    asyncio.run(run_live_full_flow(args))


if __name__ == "__main__":
    main()
