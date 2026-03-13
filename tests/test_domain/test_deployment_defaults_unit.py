"""Unit tests for deployment defaults hydration helpers."""

from __future__ import annotations

from apps.api.agents.deployment_defaults import (
    hydrate_deployment_profile_defaults,
    merge_risk_limits_with_defaults,
)


def test_hydrate_deployment_defaults_populates_capital_and_position_sizing() -> None:
    profile, runtime_state = hydrate_deployment_profile_defaults(
        profile={},
        runtime_state={},
        deploy_defaults={
            "capital_allocated": "25000",
            "max_position_size_pct": 12.5,
            "auto_start": True,
            "stop_loss_pct": 2.0,
        },
    )

    assert profile["planned_capital_allocated"] == "25000"
    assert profile["planned_auto_start"] is True
    assert profile["planned_risk_limits"]["position_sizing_override"] == {
        "mode": "pct_equity",
        "pct": 0.125,
    }
    assert runtime_state["planned_risk_limits"]["stop_loss_pct"] == 2.0


def test_hydrate_deployment_defaults_preserves_explicit_profile_values() -> None:
    profile, _runtime_state = hydrate_deployment_profile_defaults(
        profile={
            "planned_capital_allocated": "5000",
            "planned_auto_start": False,
            "planned_risk_limits": {
                "position_sizing_override": {"mode": "pct_equity", "pct": 0.3},
            },
        },
        runtime_state={},
        deploy_defaults={
            "capital_allocated": "15000",
            "max_position_size_pct": 10.0,
            "auto_start": True,
        },
    )

    assert profile["planned_capital_allocated"] == "5000"
    assert profile["planned_auto_start"] is False
    assert profile["planned_risk_limits"]["position_sizing_override"]["pct"] == 0.3


def test_merge_risk_limits_maps_max_position_pct_to_position_override() -> None:
    merged = merge_risk_limits_with_defaults(
        base_risk_limits={"max_position_size_pct": 20},
        deploy_defaults={},
    )

    assert merged["max_position_size_pct"] == 20
    assert merged["position_sizing_override"] == {
        "mode": "pct_equity",
        "pct": 0.2,
    }
