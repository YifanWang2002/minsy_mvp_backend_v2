"""Generate one full demo chat session (KYC -> strategy -> deployment) for 2@test.com.

This script does four things end-to-end:
1) Reset user KYC status to incomplete so a new thread starts from KYC.
2) Run real chat turns against /chat/send-openai-stream (OpenAI + MCP in the loop).
3) Ensure one confirmed strategy + completed backtest, then proceed to deployment.
4) Apply demo polish:
   - overwrite one backtest result payload with prettier analytics-friendly data
   - append one final assistant summary turn with MCP/backtest UI payloads

Usage:
  uv run python scripts/generate_demo_full_chat_session.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import httpx


API_BASE = "http://127.0.0.1:8000/api/v1"
DB_DSN = "postgresql://postgres:123456@localhost:5432/minsy_pgsql"


@dataclass(slots=True)
class TurnResult:
    session_id: str
    done_payload: dict[str, Any]
    choice_prompt: dict[str, Any] | None
    assistant_text: str
    mcp_events: list[dict[str, Any]]
    phase_changes: list[dict[str, Any]]


def _is_english(language: str) -> bool:
    return language.strip().lower().startswith("en")


def _parse_sse_payloads(raw_text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in [item.strip() for item in raw_text.split("\n\n") if item.strip()]:
        for line in block.splitlines():
            if not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ").strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                payloads.append(decoded)
    return payloads


def _extract_choice_prompt(payloads: list[dict[str, Any]], assistant_text: str) -> dict[str, Any] | None:
    for payload in payloads:
        if payload.get("type") != "genui":
            continue
        candidate = payload.get("payload")
        if isinstance(candidate, dict) and candidate.get("type") == "choice_prompt":
            return candidate

    matches = list(
        re.finditer(
            r"<\s*AGENT_UI_JSON\s*>([\s\S]*?)</\s*AGENT_UI_JSON\s*>",
            assistant_text,
            flags=re.IGNORECASE,
        )
    )
    for match in reversed(matches):
        raw_json = match.group(1)
        try:
            candidate = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "choice_prompt":
            return candidate
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _choice_msg(choice_id: str, option_id: str, option_label: str) -> str:
    payload = {
        "choice_id": choice_id,
        "selected_option_id": option_id,
        "selected_option_label": option_label,
    }
    return f"<CHOICE_SELECTION>{json.dumps(payload, ensure_ascii=False)}</CHOICE_SELECTION>"


def _desired_option_ids(choice_id: str, *, alpaca_broker_id: str | None) -> list[str]:
    key = choice_id.strip().lower()
    if "trading_years" in key:
        return ["years_5_plus", "years_3_5", "years_1_3"]
    if "risk" in key:
        return ["aggressive", "moderate"]
    if "return" in key:
        return ["high_growth", "growth", "balanced_growth"]
    if "market" in key:
        return ["crypto"]
    if "instrument" in key:
        return ["BTCUSD", "BTC/USD", "BTCUSDT", "XBTUSD"]
    if "frequency" in key:
        return ["multiple_per_day", "daily", "few_per_week"]
    if "holding" in key:
        return ["swing_days", "intraday"]
    if key == "selected_broker_account_id":
        targets: list[str] = []
        if alpaca_broker_id:
            targets.append(alpaca_broker_id)
        targets.extend(["create_builtin_sandbox"])
        return targets
    if key == "deployment_confirmation_status":
        return ["confirmed"]
    return []


def _pick_choice_option(
    prompt: dict[str, Any],
    *,
    alpaca_broker_id: str | None,
) -> tuple[str, str] | None:
    choice_id = _clean_text(prompt.get("choice_id"))
    raw_options = prompt.get("options")
    if not choice_id or not isinstance(raw_options, list):
        return None
    options = [item for item in raw_options if isinstance(item, dict)]
    if not options:
        return None

    desired = [item.lower() for item in _desired_option_ids(choice_id, alpaca_broker_id=alpaca_broker_id)]
    if desired:
        for target in desired:
            for option in options:
                oid = _clean_text(option.get("id"))
                if oid.lower() == target:
                    label = _clean_text(option.get("label")) or oid
                    return oid, label

    first = options[0]
    fallback_id = _clean_text(first.get("id"))
    fallback_label = _clean_text(first.get("label")) or fallback_id
    if fallback_id:
        return fallback_id, fallback_label
    return None


def _extract_strategy_id_from_artifacts(artifacts: dict[str, Any]) -> str | None:
    strategy = artifacts.get("strategy")
    if not isinstance(strategy, dict):
        return None
    profile = strategy.get("profile")
    if not isinstance(profile, dict):
        return None
    strategy_id = _clean_text(profile.get("strategy_id"))
    return strategy_id or None


def _extract_backtest_job_id_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if _clean_text(message.get("role")) != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for item in reversed(tool_calls):
            if not isinstance(item, dict):
                continue
            if _clean_text(item.get("type")) != "mcp_call":
                continue
            name = _clean_text(item.get("name")).lower()
            if name not in {"backtest_get_job", "backtest_create_job"}:
                continue
            output = item.get("output")
            if not isinstance(output, dict):
                continue
            job_id = _clean_text(output.get("job_id"))
            if job_id:
                return job_id
    return None


def _extract_deployment_id_from_session(detail: dict[str, Any]) -> str | None:
    artifacts = detail.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    deployment_block = artifacts.get("deployment")
    if not isinstance(deployment_block, dict):
        return None
    runtime_state = deployment_block.get("runtime")
    if isinstance(runtime_state, dict):
        for key in ("latest_deployment_id", "deployment_id"):
            value = _clean_text(runtime_state.get(key))
            if value:
                return value
        latest = runtime_state.get("latest_deployment")
        if isinstance(latest, dict):
            value = _clean_text(latest.get("deployment_id") or latest.get("id"))
            if value:
                return value
    profile = deployment_block.get("profile")
    if isinstance(profile, dict):
        value = _clean_text(profile.get("latest_deployment_id"))
        if value:
            return value
    return None


def _build_demo_crypto_dsl() -> dict[str, Any]:
    suffix = uuid4().hex[:8]
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": f"Demo Crypto Trend {suffix}",
            "description": "Video demo strategy from validated harness template.",
        },
        "universe": {"market": "crypto", "tickers": ["BTC/USD"]},
        "timeframe": "1m",
        "factors": {
            "ema_9": {"type": "ema", "params": {"period": 9, "source": "close"}},
            "ema_21": {"type": "ema", "params": {"period": 21, "source": "close"}},
        },
        "trade": {
            "long": {
                "position_sizing": {"mode": "fixed_qty", "qty": 1},
                "entry": {
                    "order": {"type": "market"},
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_9"},
                            "op": "cross_above",
                            "b": {"ref": "ema_21"},
                        }
                    },
                },
                "exits": [
                    {
                        "type": "signal_exit",
                        "name": "exit_on_cross_down",
                        "order": {"type": "market"},
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_9"},
                                "op": "cross_below",
                                "b": {"ref": "ema_21"},
                            }
                        },
                    },
                    {
                        "type": "bracket_rr",
                        "name": "protective_bracket",
                        "stop": {"kind": "pct", "value": 0.01},
                        "risk_reward": 2.0,
                    },
                ],
            }
        },
    }


def _build_pretty_backtest_result(
    *,
    strategy_id: str,
    start_time: datetime,
) -> dict[str, Any]:
    points = 540
    start_equity = 10000.0
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    wins = 0
    losses = 0
    total_pnl = 0.0
    commission_total = 0.0

    # Deterministic long-horizon profile:
    # - moderate positive drift over ~18 months
    # - visible drawdowns and recovery cycles
    for idx in range(points):
        ts = start_time + timedelta(days=idx)
        drift = 0.00028
        seasonal = (
            0.0010 * math.sin(idx / 18.5)
            + 0.0007 * math.sin(idx / 51.0)
            - 0.00055 * math.cos(idx / 33.0)
        )
        shock = ((idx * 9301 + 49297) % 233280) / 233280.0 - 0.5
        ret = drift + seasonal + shock * 0.0012
        if idx % 90 in {28, 29, 30}:
            ret -= 0.003
        if idx % 145 in {72}:
            ret -= 0.0042
        ret = max(min(ret, 0.011), -0.013)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)
        equity_curve.append({"timestamp": ts.replace(tzinfo=UTC).isoformat(), "equity": round(equity, 2)})

        if idx % 6 == 0 and idx > 20:
            signal = (
                0.0032 * math.sin(idx / 11.0)
                + 0.0016 * math.cos(idx / 29.0)
                + ((((idx * 177 + 29) % 1000) / 1000.0) - 0.5) * 0.0014
                + 0.00055
            )
            gross_pnl = equity * signal * 3.1
            commission = max(0.55, abs(gross_pnl) * 0.04)
            pnl = gross_pnl - commission
            quantity = round(0.8 + ((idx % 7) * 0.2), 2)
            entry_price = 28000 + idx * 19.5 + 340 * math.sin(idx / 14.0)
            exit_price = entry_price * (1.0 + (gross_pnl / max(equity, 1.0)))
            bars_held = 8 + (idx % 24)
            side = "long" if signal >= 0 else "short"
            exit_reason = "take_profit" if pnl >= 0 else "stop_loss"
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
            total_pnl += pnl
            commission_total += commission
            trades.append(
                {
                    "side": side,
                    "entry_time": (ts - timedelta(minutes=bars_held * 15)).replace(tzinfo=UTC).isoformat(),
                    "exit_time": ts.replace(tzinfo=UTC).isoformat(),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "quantity": quantity,
                    "bars_held": bars_held,
                    "exit_reason": exit_reason,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round((pnl / max(equity, 1.0)) * 100.0, 4),
                    "commission": round(commission, 4),
                }
            )

    returns: list[float] = []
    for left, right in zip(equity_curve, equity_curve[1:], strict=False):
        prev_eq = float(left["equity"])
        next_eq = float(right["equity"])
        if prev_eq == 0:
            returns.append(0.0)
        else:
            returns.append((next_eq / prev_eq) - 1.0)

    total_return_pct = (float(equity_curve[-1]["equity"]) / start_equity - 1.0) * 100.0
    trade_count = len(trades)
    win_rate = (wins / trade_count * 100.0) if trade_count else 0.0

    return {
        "strategy_id": strategy_id,
        "market": "crypto",
        "symbol": "BTCUSD",
        "timeframe": "1d",
        "started_at": start_time.replace(tzinfo=UTC).isoformat(),
        "finished_at": (start_time + timedelta(days=points - 1)).replace(tzinfo=UTC).isoformat(),
        "summary": {
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(abs(max_dd), 2),
            "trade_count": trade_count,
            "win_rate_pct": round(win_rate, 2),
            "net_pnl": round(total_pnl - commission_total, 2),
        },
        "performance": {
            "library": "quantstats",
            "metrics": {
                "sharpe": 0.96,
                "sortino": 1.28,
                "calmar": 0.74,
                "cagr": 0.12,
            },
            "meta": {"source": "demo_polish"},
        },
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "events": [
            {"timestamp": equity_curve[118]["timestamp"], "type": "regime_change", "detail": "trend_acceleration"},
            {"timestamp": equity_curve[262]["timestamp"], "type": "risk_event", "detail": "mid_cycle_drawdown"},
            {"timestamp": equity_curve[447]["timestamp"], "type": "stability", "detail": "recovery_confirmed"},
        ],
    }


async def _reset_kyc_profile(conn: asyncpg.Connection, *, user_id: UUID) -> None:
    row = await conn.fetchrow(
        "select user_id from user_profiles where user_id=$1 limit 1",
        user_id,
    )
    if row is None:
        await conn.execute(
            """
            insert into user_profiles (
                id, user_id, trading_years_bucket, risk_tolerance,
                return_expectation, kyc_status, created_at, updated_at
            ) values ($1, $2, null, null, null, 'incomplete', now(), now())
            """,
            uuid4(),
            user_id,
        )
        return
    await conn.execute(
        """
        update user_profiles
        set
            trading_years_bucket = null,
            risk_tolerance = null,
            return_expectation = null,
            kyc_status = 'incomplete',
            updated_at = now()
        where user_id = $1
        """,
        user_id,
    )


async def _apply_backtest_polish(
    conn: asyncpg.Connection,
    *,
    job_id: UUID,
    strategy_id: str,
    session_id: UUID,
) -> None:
    started_at = datetime.now(UTC) - timedelta(days=540)
    pretty_result = _build_pretty_backtest_result(strategy_id=strategy_id, start_time=started_at)
    await conn.execute(
        """
        update backtest_jobs
        set
            status = 'completed',
            progress = 100,
            current_step = 'completed',
            results = $2::jsonb,
            error_message = null,
            completed_at = now(),
            session_id = $3,
            updated_at = now()
        where id = $1
        """,
        job_id,
        json.dumps(pretty_result, ensure_ascii=False),
        session_id,
    )


async def _append_polish_turn(
    conn: asyncpg.Connection,
    *,
    session_id: UUID,
    strategy_id: str | None,
    backtest_job_id: str | None,
    deployment_id: str | None,
    alpaca_broker_id: str | None,
    language: str,
) -> None:
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    now = datetime.now(UTC)
    en = _is_english(language)
    user_text = (
        "Great, please summarize this end-to-end flow from KYC to deployment into a demo-ready recap and highlight the backtest."
        if en
        else "很好，请把这次从 KYC 到部署的流程整理成一个可录屏的演示版摘要，并突出回测表现。"
    )

    bt_charts_payload = None
    if backtest_job_id:
        bt_charts_payload = {
            "type": "backtest_charts",
            "job_id": backtest_job_id,
            "strategy_id": strategy_id,
            "charts": [
                "equity_curve",
                "underwater_curve",
                "monthly_return_table",
                "holding_period_pnl_bins",
            ],
            "sampling": "eod",
            "max_points": 365,
            "title": "BTCUSD Trend Strategy - Credible Demo Backtest",
            "source": "manual_demo_polish",
        }

    summary_lines = (
        [
            "The demo flow is fully closed-loop: KYC -> pre-strategy -> strategy -> backtest -> deployment.",
            "Strategy theme: BTCUSD trend-following with momentum filter and ATR-style risk control.",
            "Backtest recap (demo profile): total return about +18.6%, max drawdown about 9.3%, Sharpe about 0.96, with both winning and losing periods.",
        ]
        if en
        else [
            "演示流程已完整闭环：KYC -> pre-strategy -> strategy -> backtest -> deployment。",
            "策略主题：BTCUSD 趋势跟随 + 动量过滤 + ATR 风控。",
            "回测摘要（演示口径）：总收益约 +18.6%，最大回撤约 9.3%，Sharpe 约 0.96，且包含盈亏波动区间。",
        ]
    )
    if deployment_id:
        summary_lines.append(
            (
                f"Deployment status: paper deployment created and started (deployment_id={deployment_id})."
                if en
                else f"部署状态：paper deployment 已创建并启动（deployment_id={deployment_id}）。"
            )
        )
    if alpaca_broker_id:
        summary_lines.append(
            (
                f"Execution account: Alpaca (broker_account_id={alpaca_broker_id})."
                if en
                else f"执行账户：Alpaca（broker_account_id={alpaca_broker_id}）。"
            )
        )
    if bt_charts_payload is not None:
        summary_lines.append(
            f"<AGENT_UI_JSON>{json.dumps(bt_charts_payload, ensure_ascii=False)}</AGENT_UI_JSON>"
        )
    assistant_text = "\n".join(summary_lines)

    mcp_tool_calls: list[dict[str, Any]] = []
    if backtest_job_id:
        mcp_tool_calls.append(
            {
                "type": "mcp_call",
                "call_id": f"demo_bt_{uuid4().hex[:8]}",
                "name": "backtest_get_job",
                "status": "success",
                "arguments": {"job_id": backtest_job_id},
                "output": {
                    "ok": True,
                    "status": "done",
                    "job_id": backtest_job_id,
                    "strategy_id": strategy_id,
                    "summary": {
                        "total_return_pct": 18.6,
                        "max_drawdown_pct": 9.3,
                        "sharpe": 0.96,
                    },
                },
            }
        )
    if deployment_id:
        mcp_tool_calls.append(
            {
                "type": "mcp_call",
                "call_id": f"demo_deploy_{uuid4().hex[:8]}",
                "name": "trading_start_deployment",
                "status": "success",
                "arguments": {"deployment_id": deployment_id},
                "output": {
                    "ok": True,
                    "data": {
                        "deployment": {"deployment_id": deployment_id, "status": "active"},
                        "resolved_broker_account_id": alpaca_broker_id,
                    },
                },
            }
        )

    await conn.execute(
        """
        insert into messages (
            id, session_id, role, content, phase, response_id, tool_calls, token_usage,
            created_at, updated_at
        ) values ($1, $2, 'user', $3, 'deployment', null, null, null, $4, $4)
        """,
        user_message_id,
        session_id,
        user_text,
        now,
    )
    await conn.execute(
        """
        insert into messages (
            id, session_id, role, content, phase, response_id, tool_calls, token_usage,
            created_at, updated_at
        ) values ($1, $2, 'assistant', $3, 'deployment', null, $4::jsonb, null, $5, $5)
        """,
        assistant_message_id,
        session_id,
        assistant_text,
        json.dumps(mcp_tool_calls, ensure_ascii=False),
        now + timedelta(milliseconds=300),
    )
    await conn.execute(
        """
        update sessions
        set
            current_phase = 'deployment',
            status = 'active',
            last_activity_at = now(),
            updated_at = now()
        where id = $1
        """,
        session_id,
    )


async def _login(client: httpx.AsyncClient, *, email: str, password: str) -> tuple[str, str]:
    response = await client.post(
        f"{API_BASE}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["access_token"]), str(payload["user_id"])


async def _send_turn(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    message: str,
    session_id: str | None,
    language: str,
) -> TurnResult:
    headers = {"Authorization": f"Bearer {access_token}"}
    body: dict[str, Any] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    response = await client.post(
        f"{API_BASE}/chat/send-openai-stream",
        params={"language": language},
        headers=headers,
        json=body,
        timeout=300,
    )
    response.raise_for_status()
    payloads = _parse_sse_payloads(response.text)
    stream_start = next((item for item in payloads if item.get("type") == "stream_start"), None)
    done = next((item for item in payloads if item.get("type") == "done"), None)
    if not isinstance(done, dict):
        raise RuntimeError(f"Turn finished without done payload: message={message[:120]}")
    resolved_session_id = (
        session_id
        or _clean_text(stream_start.get("session_id") if isinstance(stream_start, dict) else "")
        or _clean_text(done.get("session_id"))
    )
    if not resolved_session_id:
        raise RuntimeError("Unable to resolve session_id from stream payload.")

    assistant_text = "".join(_clean_text(item.get("delta")) for item in payloads if item.get("type") == "text_delta")
    choice_prompt = _extract_choice_prompt(payloads, assistant_text)
    mcp_events = [item for item in payloads if item.get("type") == "mcp_event"]
    phase_changes = [item for item in payloads if item.get("type") == "phase_change"]
    return TurnResult(
        session_id=resolved_session_id,
        done_payload=done,
        choice_prompt=choice_prompt,
        assistant_text=assistant_text,
        mcp_events=mcp_events,
        phase_changes=phase_changes,
    )


async def _get_session_detail(client: httpx.AsyncClient, *, access_token: str, session_id: str) -> dict[str, Any]:
    response = await client.get(
        f"{API_BASE}/sessions/{session_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid session detail payload.")
    return payload


async def _list_brokers(client: httpx.AsyncClient, *, access_token: str) -> list[dict[str, Any]]:
    response = await client.get(
        f"{API_BASE}/broker-accounts",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


async def _confirm_strategy_with_auto_backtest(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    session_id: str,
    language: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{API_BASE}/strategies/confirm",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "session_id": session_id,
            "dsl_json": _build_demo_crypto_dsl(),
            "auto_start_backtest": True,
            "auto_message": (
                "Run the backtest now until the backtest job reaches done, then report conclusions and key metrics."
                if _is_english(language)
                else "请立即执行回测，直到 backtest job 进入 done，然后输出结论与关键指标。"
            ),
            "language": language,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid strategy confirm response.")
    return payload


async def _ensure_demo_backtest_job(client: httpx.AsyncClient, *, access_token: str) -> dict[str, Any]:
    response = await client.post(
        f"{API_BASE}/backtests/jobs/demo/ensure",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid backtest demo ensure response.")
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one full demo chat session for 2@test.com")
    parser.add_argument("--email", default="2@test.com")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--keep-kyc", action="store_true", help="Do not reset KYC profile.")
    parser.add_argument("--no-polish", action="store_true", help="Skip backtest/message polish step.")
    args = parser.parse_args()

    async with asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=2) as pool:
        async with pool.acquire() as conn:
            user_row = await conn.fetchrow(
                "select id from users where email=$1 limit 1",
                args.email,
            )
            if user_row is None:
                raise RuntimeError(f"User not found: {args.email}")
            user_id = UUID(str(user_row["id"]))
            if not args.keep_kyc:
                await _reset_kyc_profile(conn, user_id=user_id)
                print(f"[setup] reset KYC profile -> incomplete for {args.email}")

    async with httpx.AsyncClient() as client:
        access_token, api_user_id = await _login(client, email=args.email, password=args.password)
        print(f"[auth] login ok user_id={api_user_id}")

        brokers = await _list_brokers(client, access_token=access_token)
        alpaca_broker_id: str | None = None
        for broker in brokers:
            provider = _clean_text(broker.get("provider")).lower()
            status = _clean_text(broker.get("status")).lower()
            if provider == "alpaca" and status in {"active", "ready", "connected"}:
                alpaca_broker_id = _clean_text(broker.get("broker_account_id"))
                break
        print(f"[setup] alpaca_broker_id={alpaca_broker_id or 'none'}")

        session_id: str | None = None
        phase = "kyc"

        # Kickoff turn.
        kickoff = (
            "Hi, I want a demo-ready full flow from KYC to strategy, backtest, and final deployment on Alpaca."
            if _is_english(args.language)
            else "你好，我要准备一条可演示的完整流程：从 KYC 到策略、回测，再部署到 Alpaca。"
        )
        turn = await _send_turn(
            client,
            access_token=access_token,
            message=kickoff,
            session_id=session_id,
            language=args.language,
        )
        session_id = turn.session_id
        phase = _clean_text(turn.done_payload.get("phase")) or phase
        print(f"[turn] kickoff phase={phase} mcp_events={len(turn.mcp_events)}")

        # KYC + pre_strategy guided loop.
        for idx in range(1, 14):
            if phase == "strategy":
                break
            prompt = turn.choice_prompt
            if not isinstance(prompt, dict):
                follow_text = "Continue" if _is_english(args.language) else "继续"
                wire_text = follow_text
            else:
                selected = _pick_choice_option(prompt, alpaca_broker_id=alpaca_broker_id)
                if selected is None:
                    follow_text = "Continue" if _is_english(args.language) else "继续"
                    wire_text = follow_text
                else:
                    option_id, label = selected
                    choice_id = _clean_text(prompt.get("choice_id"))
                    if phase == "deployment" or choice_id in {
                        "selected_broker_account_id",
                        "deployment_confirmation_status",
                    }:
                        wire_text = _choice_msg(choice_id=choice_id, option_id=option_id, option_label=label)
                        follow_text = label
                    else:
                        wire_text = label
                        follow_text = label
            turn = await _send_turn(
                client,
                access_token=access_token,
                message=wire_text,
                session_id=session_id,
                language=args.language,
            )
            phase = _clean_text(turn.done_payload.get("phase")) or phase
            print(
                f"[turn] collect#{idx} msg={follow_text[:36]} phase={phase} "
                f"mcp_events={len(turn.mcp_events)}"
            )

        if phase != "strategy":
            raise RuntimeError(f"Expected strategy phase, got phase={phase}")

        # Force explicit save + backtest by API confirm (deterministic).
        confirm_payload = await _confirm_strategy_with_auto_backtest(
            client,
            access_token=access_token,
            session_id=session_id,
            language=args.language,
        )
        strategy_id = _clean_text(confirm_payload.get("strategy_id"))
        print(
            "[strategy] confirm ok "
            f"strategy_id={strategy_id} auto_started={confirm_payload.get('auto_started')}"
        )

        # Move from strategy to deployment using explicit confirmation turn.
        transition_turn = await _send_turn(
            client,
            access_token=access_token,
            message=(
                "I confirm the strategy is finalized and ready for deployment. Please enter the deployment phase now."
                if _is_english(args.language)
                else "我确认策略已经最终定稿并准备部署，请现在进入 deployment 阶段。"
            ),
            session_id=session_id,
            language=args.language,
        )
        phase = _clean_text(transition_turn.done_payload.get("phase")) or phase
        print(f"[turn] transition phase={phase} mcp_events={len(transition_turn.mcp_events)}")

        if phase != "deployment":
            transition_turn = await _send_turn(
                client,
                access_token=access_token,
                message=(
                    "Confirming again: please move to the deployment flow immediately."
                    if _is_english(args.language)
                    else "再次确认：请立即进入部署流程。"
                ),
                session_id=session_id,
                language=args.language,
            )
            phase = _clean_text(transition_turn.done_payload.get("phase")) or phase
            print(f"[turn] transition-2 phase={phase} mcp_events={len(transition_turn.mcp_events)}")

        if phase != "deployment":
            raise RuntimeError(f"Failed to enter deployment phase. current_phase={phase}")

        # Deployment readiness turn.
        deploy_turn = await _send_turn(
            client,
            access_token=access_token,
            message=(
                "Please check deployment readiness, prioritize the Alpaca account, and give me the next confirmation options."
                if _is_english(args.language)
                else "请检查部署就绪性，优先使用 Alpaca 账户，并给我下一步确认选项。"
            ),
            session_id=session_id,
            language=args.language,
        )
        phase = _clean_text(deploy_turn.done_payload.get("phase")) or phase
        print(f"[turn] deployment-preflight phase={phase} mcp_events={len(deploy_turn.mcp_events)}")

        # Broker selection if prompted.
        current_prompt = deploy_turn.choice_prompt
        if isinstance(current_prompt, dict):
            cid = _clean_text(current_prompt.get("choice_id"))
            if cid == "selected_broker_account_id":
                selected = _pick_choice_option(current_prompt, alpaca_broker_id=alpaca_broker_id)
                if selected is not None:
                    oid, label = selected
                    deploy_turn = await _send_turn(
                        client,
                        access_token=access_token,
                        message=_choice_msg(choice_id=cid, option_id=oid, option_label=label),
                        session_id=session_id,
                        language=args.language,
                    )
                    phase = _clean_text(deploy_turn.done_payload.get("phase")) or phase
                    print(f"[turn] broker-choice={oid} phase={phase} mcp_events={len(deploy_turn.mcp_events)}")

        # Deployment confirmation.
        current_prompt = deploy_turn.choice_prompt
        if isinstance(current_prompt, dict) and _clean_text(current_prompt.get("choice_id")) == "deployment_confirmation_status":
            deploy_turn = await _send_turn(
                client,
                access_token=access_token,
                message=_choice_msg(
                    choice_id="deployment_confirmation_status",
                    option_id="confirmed",
                    option_label="Confirm deployment",
                ),
                session_id=session_id,
                language=args.language,
            )
            phase = _clean_text(deploy_turn.done_payload.get("phase")) or phase
            print(f"[turn] deployment-confirm phase={phase} mcp_events={len(deploy_turn.mcp_events)}")
        else:
            deploy_turn = await _send_turn(
                client,
                access_token=access_token,
                message=(
                    "I confirm the deployment summary. Please create and start it directly."
                    if _is_english(args.language)
                    else "我确认部署摘要，请直接执行创建并启动。"
                ),
                session_id=session_id,
                language=args.language,
            )
            phase = _clean_text(deploy_turn.done_payload.get("phase")) or phase
            print(f"[turn] deployment-confirm-text phase={phase} mcp_events={len(deploy_turn.mcp_events)}")

        # Final execute/status turn.
        final_turn = await _send_turn(
            client,
            access_token=access_token,
            message=(
                "Please create and start the paper deployment, then provide the deployment id and status."
                if _is_english(args.language)
                else "请创建并启动 paper deployment，然后给我 deployment id 与状态。"
            ),
            session_id=session_id,
            language=args.language,
        )
        phase = _clean_text(final_turn.done_payload.get("phase")) or phase
        print(f"[turn] deployment-execute phase={phase} mcp_events={len(final_turn.mcp_events)}")

        detail = await _get_session_detail(client, access_token=access_token, session_id=session_id)
        artifacts = detail.get("artifacts") if isinstance(detail.get("artifacts"), dict) else {}
        strategy_id = _extract_strategy_id_from_artifacts(artifacts or {}) or strategy_id
        messages = detail.get("messages") if isinstance(detail.get("messages"), list) else []

        backtest_job_id = _extract_backtest_job_id_from_messages(messages)
        if backtest_job_id is None:
            demo_job = await _ensure_demo_backtest_job(client, access_token=access_token)
            backtest_job_id = _clean_text(demo_job.get("job_id")) or None
            print(f"[backtest] ensured demo job={backtest_job_id} source={_clean_text(demo_job.get('source'))}")
        else:
            print(f"[backtest] detected chat-linked job={backtest_job_id}")

        deployment_id = _extract_deployment_id_from_session(detail)
        print(f"[deployment] latest_deployment_id={deployment_id}")

        if not args.no_polish:
            async with asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=2) as pool:
                async with pool.acquire() as conn:
                    if backtest_job_id is not None and strategy_id:
                        await _apply_backtest_polish(
                            conn,
                            job_id=UUID(backtest_job_id),
                            strategy_id=strategy_id,
                            session_id=UUID(session_id),
                        )
                        print(f"[polish] updated backtest job analytics job_id={backtest_job_id}")
                    await _append_polish_turn(
                        conn,
                        session_id=UUID(session_id),
                        strategy_id=strategy_id or None,
                        backtest_job_id=backtest_job_id,
                        deployment_id=deployment_id,
                        alpaca_broker_id=alpaca_broker_id,
                        language=args.language,
                    )
                    print("[polish] appended final demo summary turn")

        refreshed = await _get_session_detail(client, access_token=access_token, session_id=session_id)
        refreshed_messages = refreshed.get("messages") if isinstance(refreshed.get("messages"), list) else []
        mcp_tool_names: list[str] = []
        for message in refreshed_messages:
            if _clean_text(message.get("role")) != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                if _clean_text(call.get("type")) != "mcp_call":
                    continue
                name = _clean_text(call.get("name"))
                if name:
                    mcp_tool_names.append(name)
        unique_tools = sorted(set(mcp_tool_names))

        print("\n=== RESULT ===")
        print(f"session_id={session_id}")
        print(f"current_phase={_clean_text(refreshed.get('current_phase'))}")
        print(f"strategy_id={strategy_id}")
        print(f"backtest_job_id={backtest_job_id}")
        print(f"deployment_id={deployment_id}")
        print(f"mcp_tools={','.join(unique_tools)}")
        print(f"assistant_message_count={sum(1 for item in refreshed_messages if _clean_text(item.get('role')) == 'assistant')}")
        print("done")


if __name__ == "__main__":
    asyncio.run(main())
