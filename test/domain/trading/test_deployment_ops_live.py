from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from packages.domain.trading import deployment_ops


async def test_000_accessibility_resolve_deployment_capital_uses_account_metadata_starting_cash() -> None:
    account = SimpleNamespace(
        validation_metadata={},
        metadata_={"starting_cash": "25000"},
    )

    resolved = await deployment_ops._resolve_deployment_capital_allocated(
        requested_capital=Decimal("0"),
        account=account,
    )

    assert resolved == Decimal("25000.00")


async def test_010_accessibility_resolve_deployment_capital_prefers_explicit_request() -> None:
    account = SimpleNamespace(
        validation_metadata={"paper_equity": "8000"},
        metadata_={"starting_cash": "25000"},
    )

    resolved = await deployment_ops._resolve_deployment_capital_allocated(
        requested_capital=Decimal("1500"),
        account=account,
    )

    assert resolved == Decimal("1500.00")


async def test_015_accessibility_resolve_deployment_capital_reports_resolution_source() -> None:
    account = SimpleNamespace(
        validation_metadata={},
        metadata_={"starting_cash": "25000"},
    )

    resolution = await deployment_ops.resolve_deployment_capital(
        requested_capital=Decimal("0"),
        account=account,
    )

    assert resolution.amount == Decimal("25000.00")
    assert resolution.source == "account_metadata"


def test_020_accessibility_runtime_compatibility_blocks_multi_symbol() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD", "ETHUSD"]},
            "trade": {},
        }
    )

    assert compatibility.status == "blocked"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_MULTI_SYMBOL" in compatibility.blocker_codes


def test_030_accessibility_runtime_compatibility_allows_dsl_exit_rules() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD"]},
            "trade": {
                "long": {
                    "exits": [
                        {"type": "signal_exit"},
                        {"type": "stop_loss"},
                    ]
                }
            },
        }
    )

    assert compatibility.status == "ok"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_EXIT_RULE" not in compatibility.blocker_codes


def test_040_accessibility_runtime_compatibility_blocks_unsupported_timeframe() -> None:
    compatibility = deployment_ops.assess_strategy_runtime_compatibility(
        {
            "universe": {"tickers": ["BTCUSD"]},
            "timeframe": "7m",
            "trade": {},
        }
    )

    assert compatibility.status == "blocked"
    assert "DEPLOYMENT_RUNTIME_UNSUPPORTED_TIMEFRAME" in compatibility.blocker_codes


def test_045_accessibility_strategy_market_supported_by_account_detects_mismatch() -> None:
    account = SimpleNamespace(
        provider="sandbox",
        exchange_id="sandbox",
        is_sandbox=True,
        capabilities={"supported_markets": ["us_stocks", "crypto"]},
    )

    supported, market = deployment_ops._strategy_market_supported_by_account(
        strategy_payload={"universe": {"market": "forex"}},
        account=account,
    )

    assert supported is False
    assert market == "forex"


def test_047_accessibility_strategy_market_supported_by_account_uses_provider_fallback() -> None:
    account = SimpleNamespace(
        provider="alpaca",
        exchange_id="",
        is_sandbox=False,
        capabilities={},
    )

    supported, market = deployment_ops._strategy_market_supported_by_account(
        strategy_payload={"universe": {"market": "us_stocks"}},
        account=account,
    )

    assert supported is True
    assert market == "us_stocks"


async def test_050_accessibility_apply_status_transition_avoids_lazy_broker_account_load(
    monkeypatch,
) -> None:
    class _Run:
        def __init__(self) -> None:
            self.deployment_id = uuid4()
            self.strategy_id = uuid4()
            self.broker_account_id = uuid4()
            self.status = "stopped"
            self.last_bar_time = None
            self.runtime_state = {}
            self.created_at = datetime.now(UTC)
            self.updated_at = datetime.now(UTC)

        @property
        def broker_account(self):  # pragma: no cover - regression guard
            raise AssertionError("apply_status_transition should not lazy-load run.broker_account")

    class _DbStub:
        def __init__(self, active_account, reloaded_deployment) -> None:
            self._results = [active_account, reloaded_deployment]

        async def scalar(self, stmt):  # noqa: ANN001
            del stmt
            if not self._results:
                raise AssertionError("unexpected scalar() call")
            return self._results.pop(0)

        async def commit(self) -> None:
            return None

    run = _Run()
    deployment = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        strategy_id=uuid4(),
        mode="paper",
        status="pending",
        deployed_at=None,
        stopped_at=None,
        strategy=None,
        deployment_runs=[run],
    )
    active_account = SimpleNamespace(status="active")
    db = _DbStub(active_account=active_account, reloaded_deployment=deployment)

    async def _noop(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        return None

    async def _fake_provider(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        return "sandbox"

    monkeypatch.setattr(deployment_ops, "_assert_running_deployment_quota_for_activation", _noop)
    monkeypatch.setattr(deployment_ops, "_enqueue_deployment_started_notification", _noop)
    monkeypatch.setattr(deployment_ops, "_sync_session_deployment_phase_state", _noop)
    monkeypatch.setattr(deployment_ops, "_resolve_broker_provider_for_run", _fake_provider)
    monkeypatch.setattr(deployment_ops.runtime_state_store, "upsert", _noop)

    transitioned = await deployment_ops.apply_status_transition(
        db,
        deployment=deployment,
        target_status="active",
    )

    assert transitioned is deployment
    assert deployment.status == "active"
    assert run.status == "starting"
