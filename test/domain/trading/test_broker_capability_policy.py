from __future__ import annotations

from packages.domain.trading.broker_capability_policy import (
    build_broker_capabilities,
    derive_supported_markets,
    evaluate_broker_compatibility,
)


def test_build_broker_capabilities_for_builtin_sandbox_supports_us_stocks_and_crypto() -> None:
    capabilities = build_broker_capabilities(
        provider="sandbox",
        exchange_id="sandbox",
        is_sandbox=True,
    )

    assert capabilities["asset_classes"] == ["us_equity", "crypto"]
    assert capabilities["supported_markets"] == ["us_stocks", "crypto"]
    assert capabilities["sandbox_supported"] is True


def test_derive_supported_markets_falls_back_to_asset_classes() -> None:
    supported_markets = derive_supported_markets(
        {"asset_classes": ["us_equity", "crypto"]}
    )

    assert supported_markets == ["us_stocks", "crypto"]


def test_evaluate_broker_compatibility_returns_needs_choice_when_multiple_match() -> None:
    result = evaluate_broker_compatibility(
        strategy_market="crypto",
        accounts=[
            {
                "broker_account_id": "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8",
                "status": "active",
                "is_default": True,
                "capabilities": {"supported_markets": ["crypto"]},
            },
            {
                "broker_account_id": "ca4084b8-d30c-44ff-b278-cba45fd01332",
                "status": "active",
                "is_default": False,
                "capabilities": {"supported_markets": ["crypto"]},
            },
        ],
    )

    assert result["status"] == "needs_choice"
    assert result["preferred_broker_account_id"] == "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    assert set(result["matched_broker_account_ids"]) == {
        "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8",
        "ca4084b8-d30c-44ff-b278-cba45fd01332",
    }


def test_evaluate_broker_compatibility_returns_ready_for_explicit_matching_selection() -> None:
    selected_broker_id = "6b3f4e8b-e0d4-4cb1-911f-0119985c31e8"
    result = evaluate_broker_compatibility(
        strategy_market="us_stocks",
        explicit_broker_account_id=selected_broker_id,
        accounts=[
            {
                "broker_account_id": selected_broker_id,
                "status": "active",
                "is_default": False,
                "capabilities": {"supported_markets": ["us_stocks"]},
            },
            {
                "broker_account_id": "ca4084b8-d30c-44ff-b278-cba45fd01332",
                "status": "inactive",
                "is_default": True,
                "capabilities": {"supported_markets": ["us_stocks"]},
            },
        ],
    )

    assert result["status"] == "ready"
    assert result["preferred_broker_account_id"] == selected_broker_id
    assert result["blockers"] == []
