"""Kill-switch controls for paper-trading runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from src.config import settings


def _normalize(value: UUID | str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_csv(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True, slots=True)
class KillSwitchDecision:
    allowed: bool
    reason: str


class RuntimeKillSwitch:
    """Evaluates global/user/deployment kill-switch controls."""

    def evaluate(
        self,
        *,
        user_id: UUID | str | None,
        deployment_id: UUID | str | None,
    ) -> KillSwitchDecision:
        if settings.paper_trading_kill_switch_global:
            return KillSwitchDecision(allowed=False, reason="kill_switch_global")

        user = _normalize(user_id)
        blocked_users = _parse_csv(settings.paper_trading_kill_switch_users_csv)
        if user and user in blocked_users:
            return KillSwitchDecision(allowed=False, reason="kill_switch_user")

        deployment = _normalize(deployment_id)
        blocked_deployments = _parse_csv(
            settings.paper_trading_kill_switch_deployments_csv,
        )
        if deployment and deployment in blocked_deployments:
            return KillSwitchDecision(
                allowed=False,
                reason="kill_switch_deployment",
            )
        return KillSwitchDecision(allowed=True, reason="ok")

    def snapshot(self) -> dict[str, Any]:
        return {
            "global": bool(settings.paper_trading_kill_switch_global),
            "blocked_users": sorted(_parse_csv(settings.paper_trading_kill_switch_users_csv)),
            "blocked_deployments": sorted(
                _parse_csv(settings.paper_trading_kill_switch_deployments_csv),
            ),
        }
