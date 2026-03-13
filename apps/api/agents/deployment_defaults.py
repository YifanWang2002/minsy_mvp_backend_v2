"""Shared helpers for deployment default hydration and risk-limit mapping."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

_DEFAULT_CAPITAL_ALLOCATED = "10000"
_DEFAULT_AUTO_START = True


def normalize_decimal_text(value: Any, *, allow_zero: bool) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if parsed < 0 or (parsed == 0 and not allow_zero):
        return None
    return format(parsed.normalize(), "f")


def normalize_percentage(value: Any, *, allow_zero: bool = False) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0 or (parsed == 0 and not allow_zero) or parsed > 100:
        return None
    return round(parsed, 6)


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def normalize_deploy_defaults(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    output: dict[str, Any] = {}

    capital = normalize_decimal_text(raw.get("capital_allocated"), allow_zero=False)
    if capital is not None:
        output["capital_allocated"] = capital

    max_position_size_pct = normalize_percentage(raw.get("max_position_size_pct"))
    if max_position_size_pct is not None:
        output["max_position_size_pct"] = max_position_size_pct

    stop_loss_pct = normalize_percentage(raw.get("stop_loss_pct"), allow_zero=True)
    if stop_loss_pct is not None:
        output["stop_loss_pct"] = stop_loss_pct

    max_daily_drawdown_pct = normalize_percentage(
        raw.get("max_daily_drawdown_pct"),
        allow_zero=True,
    )
    if max_daily_drawdown_pct is not None:
        output["max_daily_drawdown_pct"] = max_daily_drawdown_pct

    auto_start = normalize_bool(raw.get("auto_start"))
    if auto_start is not None:
        output["auto_start"] = auto_start

    risk_limits = raw.get("risk_limits")
    if isinstance(risk_limits, dict):
        output["risk_limits"] = dict(risk_limits)

    return output


def _build_position_sizing_override(
    *,
    max_position_size_pct: float,
) -> dict[str, Any]:
    pct_equity = round(max_position_size_pct / 100.0, 6)
    return {
        "mode": "pct_equity",
        "pct": pct_equity,
    }


def merge_risk_limits_with_defaults(
    *,
    base_risk_limits: dict[str, Any] | None,
    deploy_defaults: dict[str, Any] | None,
) -> dict[str, Any]:
    risk_limits = (
        dict(base_risk_limits)
        if isinstance(base_risk_limits, dict)
        else {}
    )
    defaults = normalize_deploy_defaults(deploy_defaults)

    # Preserve direct deployment-scope values while ensuring runtime-facing
    # position sizing override is always populated when default pct is available.
    max_position_size_pct = normalize_percentage(
        risk_limits.get("max_position_size_pct")
    )
    if max_position_size_pct is None:
        max_position_size_pct = normalize_percentage(defaults.get("max_position_size_pct"))
    if max_position_size_pct is not None:
        risk_limits.setdefault("max_position_size_pct", max_position_size_pct)
        if not isinstance(risk_limits.get("position_sizing_override"), dict):
            risk_limits["position_sizing_override"] = _build_position_sizing_override(
                max_position_size_pct=max_position_size_pct,
            )

    stop_loss_pct = normalize_percentage(
        risk_limits.get("stop_loss_pct"),
        allow_zero=True,
    )
    if stop_loss_pct is None:
        stop_loss_pct = normalize_percentage(defaults.get("stop_loss_pct"), allow_zero=True)
    if stop_loss_pct is not None:
        risk_limits.setdefault("stop_loss_pct", stop_loss_pct)

    max_daily_drawdown_pct = normalize_percentage(
        risk_limits.get("max_daily_drawdown_pct"),
        allow_zero=True,
    )
    if max_daily_drawdown_pct is None:
        max_daily_drawdown_pct = normalize_percentage(
            defaults.get("max_daily_drawdown_pct"),
            allow_zero=True,
        )
    if max_daily_drawdown_pct is not None:
        risk_limits.setdefault("max_daily_drawdown_pct", max_daily_drawdown_pct)

    # Merge optional nested defaults only when caller did not already provide keys.
    nested_risk_limits = defaults.get("risk_limits")
    if isinstance(nested_risk_limits, dict):
        for key, value in nested_risk_limits.items():
            risk_limits.setdefault(key, value)

    return risk_limits


def hydrate_deployment_profile_defaults(
    *,
    profile: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
    deploy_defaults: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_profile = dict(profile or {})
    resolved_runtime = dict(runtime_state or {})
    defaults = normalize_deploy_defaults(deploy_defaults)

    capital = normalize_decimal_text(
        resolved_profile.get("planned_capital_allocated"),
        allow_zero=False,
    )
    if capital is None:
        capital = normalize_decimal_text(
            resolved_runtime.get("planned_capital_allocated"),
            allow_zero=False,
        )
    if capital is None:
        capital = normalize_decimal_text(defaults.get("capital_allocated"), allow_zero=False)
    if capital is None:
        capital = _DEFAULT_CAPITAL_ALLOCATED

    auto_start = normalize_bool(resolved_profile.get("planned_auto_start"))
    if auto_start is None:
        auto_start = normalize_bool(resolved_runtime.get("planned_auto_start"))
    if auto_start is None:
        auto_start = normalize_bool(defaults.get("auto_start"))
    if auto_start is None:
        auto_start = _DEFAULT_AUTO_START

    planned_risk_limits = merge_risk_limits_with_defaults(
        base_risk_limits=(
            resolved_profile.get("planned_risk_limits")
            if isinstance(resolved_profile.get("planned_risk_limits"), dict)
            else resolved_runtime.get("planned_risk_limits")
            if isinstance(resolved_runtime.get("planned_risk_limits"), dict)
            else {}
        ),
        deploy_defaults=defaults,
    )

    resolved_profile["planned_capital_allocated"] = capital
    resolved_profile["planned_auto_start"] = bool(auto_start)
    resolved_profile["planned_risk_limits"] = planned_risk_limits

    resolved_runtime["planned_capital_allocated"] = capital
    resolved_runtime["planned_auto_start"] = bool(auto_start)
    resolved_runtime["planned_risk_limits"] = planned_risk_limits
    if defaults:
        resolved_runtime["deploy_defaults"] = defaults

    return resolved_profile, resolved_runtime
