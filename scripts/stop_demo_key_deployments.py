#!/usr/bin/env python3
"""Stop active paper deployments that use placeholder/demo broker credentials."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.execution.credentials import CredentialCipher  # noqa: E402
from src.models import database as db_module  # noqa: E402
from src.models.broker_account import BrokerAccount  # noqa: E402
from src.models.deployment import Deployment  # noqa: E402
from src.models.deployment_run import DeploymentRun  # noqa: E402
from src.models.user import User  # noqa: E402

_PLACEHOLDER_VALUES = {
    "",
    "demo",
    "demo_key",
    "demo_secret",
    "test",
    "placeholder",
    "changeme",
    "your_api_key",
    "your_api_secret",
    "your_alpaca_api_key",
    "your_alpaca_api_secret",
    "api_key",
    "api_secret",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes. Without this flag, only print dry-run candidates.",
    )
    parser.add_argument(
        "--reason",
        default="bulk_stop_demo_credentials",
        help="runtime_reason written into deployment_runs.runtime_state when stopping.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON summary in addition to plain text logs.",
    )
    return parser.parse_args()


def _extract_credential_value(credentials: dict[str, Any], *keys: str) -> str:
    for key in keys:
        raw = credentials.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return ""


def _is_placeholder_credentials(credentials: dict[str, Any]) -> bool:
    key = _extract_credential_value(credentials, "APCA-API-KEY-ID", "api_key", "key").strip().lower()
    secret = _extract_credential_value(
        credentials,
        "APCA-API-SECRET-KEY",
        "api_secret",
        "secret",
    ).strip().lower()

    if key in _PLACEHOLDER_VALUES or secret in _PLACEHOLDER_VALUES:
        return True
    if key.startswith("demo") and secret.startswith("demo"):
        return True
    return False


def _latest_run(deployment: Deployment) -> DeploymentRun | None:
    runs = list(deployment.deployment_runs or [])
    if not runs:
        return None
    runs.sort(key=lambda row: (row.created_at, row.id))
    return runs[-1]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    await db_module.init_postgres(ensure_schema=False)
    assert db_module.AsyncSessionLocal is not None

    async with db_module.AsyncSessionLocal() as db:
        deployments = list(
            (
                await db.scalars(
                    select(Deployment)
                    .options(selectinload(Deployment.deployment_runs))
                    .where(Deployment.mode == "paper", Deployment.status == "active")
                    .order_by(Deployment.created_at.asc()),
                )
            ).all()
        )
        if not deployments:
            return {
                "dry_run": not args.execute,
                "active_paper_deployments": 0,
                "candidates": 0,
                "stopped": 0,
                "rows": [],
            }

        users = {row.id: row.email for row in (await db.scalars(select(User))).all()}
        latest_runs = {deployment.id: _latest_run(deployment) for deployment in deployments}
        broker_ids = sorted(
            {
                latest_run.broker_account_id
                for latest_run in latest_runs.values()
                if latest_run is not None
            },
        )
        accounts = (
            list((await db.scalars(select(BrokerAccount).where(BrokerAccount.id.in_(broker_ids)))).all())
            if broker_ids
            else []
        )
        account_by_id = {row.id: row for row in accounts}
        cipher = CredentialCipher()

        rows: list[dict[str, Any]] = []
        for deployment in deployments:
            run = latest_runs.get(deployment.id)
            if run is None:
                continue
            account = account_by_id.get(run.broker_account_id)
            if account is None:
                continue
            try:
                credentials = cipher.decrypt(account.encrypted_credentials)
            except Exception:  # noqa: BLE001
                continue
            if not _is_placeholder_credentials(credentials):
                continue
            rows.append(
                {
                    "deployment_id": str(deployment.id),
                    "run_id": str(run.id),
                    "broker_account_id": str(account.id),
                    "user_id": str(deployment.user_id),
                    "user_email": users.get(deployment.user_id, ""),
                    "account_validation_status": account.last_validated_status,
                }
            )

        if args.execute and rows:
            now = datetime.now(UTC)
            row_by_deployment_id = {row["deployment_id"]: row for row in rows}
            for deployment in deployments:
                row = row_by_deployment_id.get(str(deployment.id))
                if row is None:
                    continue
                run = latest_runs.get(deployment.id)
                deployment.status = "stopped"
                deployment.stopped_at = now
                if run is not None:
                    run.status = "stopped"
                    state = dict(run.runtime_state) if isinstance(run.runtime_state, dict) else {}
                    state["runtime_status"] = "stopped"
                    state["runtime_reason"] = args.reason
                    state["stopped_at"] = now.isoformat()
                    run.runtime_state = state
            await db.commit()

        return {
            "dry_run": not args.execute,
            "active_paper_deployments": len(deployments),
            "candidates": len(rows),
            "stopped": len(rows) if args.execute else 0,
            "rows": rows,
        }


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_run(args))

    mode_label = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"[{mode_label}] active_paper={summary['active_paper_deployments']} "
        f"candidates={summary['candidates']} stopped={summary['stopped']}",
    )
    for row in summary["rows"]:
        print(
            " - deployment={deployment_id} user={user_email} "
            "broker={broker_account_id} validation={account_validation_status}".format(**row),
        )
    if args.json:
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
