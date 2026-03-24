"""Refactor verification checks for factor-engine single-source migration.

Usage:
  uv run python scripts/refactor_factor_checks.py --mode smoke
  uv run python scripts/refactor_factor_checks.py --mode step1
  uv run python scripts/refactor_factor_checks.py --mode step2
  uv run python scripts/refactor_factor_checks.py --mode step3
  uv run python scripts/refactor_factor_checks.py --mode step4
  uv run python scripts/refactor_factor_checks.py --mode p0
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.mcp.domains.market_data import tools as market_tools
from apps.mcp.domains.strategy import tools as strategy_tools
from packages.domain.backtest.factors import prepare_backtest_frame
from packages.domain.backtest.trade_snapshot import _resolve_factor_series
from packages.domain.market_data.runtime import RuntimeBar
from packages.domain.market_data.regime import feature_snapshot as regime_feature_snapshot
from packages.domain.strategy.feature.indicators import IndicatorRegistry
from packages.domain.strategy.feature.indicators import IndicatorWrapper
from packages.domain.strategy.parser import build_parsed_strategy
from packages.domain.strategy.semantic import validate_strategy_semantics
from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime


def _log(message: str) -> None:
    print(f"[CHECK] {message}")


def _load_example_strategy_payload() -> dict[str, Any]:
    path = Path("packages/domain/strategy/assets/example_strategy.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_ohlcv(rows: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0.0, 24.0, rows)
    base = 100.0 + np.sin(x) * 3.0 + np.linspace(0.0, 8.0, rows)
    noise = np.cos(x * 2.3) * 0.15
    close = base + noise
    open_ = close + np.sin(x * 1.7) * 0.12
    high = np.maximum(open_, close) + 0.5 + np.abs(np.sin(x * 0.7)) * 0.2
    low = np.minimum(open_, close) - 0.5 - np.abs(np.cos(x * 0.6)) * 0.2
    volume = 1000.0 + 120.0 * np.sin(x * 1.3) + np.linspace(0.0, 50.0, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.maximum(volume, 1.0),
        },
        index=idx,
    )


def _build_future_leak_probe_ohlcv(rows: int = 320) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0.0, 12.0, rows)
    close = 100.0 + np.linspace(0.0, 6.0, rows) + np.sin(x) * 0.15
    open_ = close + np.cos(x * 0.7) * 0.05
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = np.full(rows, 1200.0)

    breakout_idx = 200
    close[breakout_idx] = close[breakout_idx - 1] + 20.0
    open_[breakout_idx] = close[breakout_idx] - 0.3
    high[breakout_idx] = close[breakout_idx] + 0.9
    low[breakout_idx] = close[breakout_idx] - 1.0
    for offset in (1, 2, 3):
        i = breakout_idx + offset
        close[i] = close[breakout_idx - 1] + (0.05 * offset)
        open_[i] = close[i] + 0.02
        high[i] = close[i] + 0.8
        low[i] = close[i] - 0.8

    dry_up_idx = 180
    volume[dry_up_idx] = 10.0
    close[dry_up_idx - 2] = close[dry_up_idx - 2] - 0.3
    close[dry_up_idx - 1] = close[dry_up_idx - 1] + 0.5
    close[dry_up_idx] = close[dry_up_idx - 1] + 0.2
    close[dry_up_idx + 1] = close[dry_up_idx] - 0.9
    open_[dry_up_idx - 1 : dry_up_idx + 2] = close[dry_up_idx - 1 : dry_up_idx + 2] + 0.03
    high[dry_up_idx - 1 : dry_up_idx + 2] = close[dry_up_idx - 1 : dry_up_idx + 2] + 0.8
    low[dry_up_idx - 1 : dry_up_idx + 2] = close[dry_up_idx - 1 : dry_up_idx + 2] - 0.8

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _bars_from_frame(frame: pd.DataFrame) -> list[RuntimeBar]:
    bars: list[RuntimeBar] = []
    for ts, row in frame.iterrows():
        bars.append(
            RuntimeBar(
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return bars


def _check_a_backtest_factor_runtime() -> None:
    _log("a) backtest factor runtime")
    payload = _load_example_strategy_payload()
    parsed = build_parsed_strategy(payload)
    frame = _build_ohlcv(rows=700)

    enriched = prepare_backtest_frame(frame, strategy=parsed)
    required_cols = {
        "ema_9",
        "ema_21",
        "rsi_14",
        "atr_14",
        "macd_12_26_9.macd_line",
        "macd_12_26_9.signal",
        "macd_12_26_9.histogram",
    }
    missing = sorted(col for col in required_cols if col not in enriched.columns)
    assert not missing, f"Backtest enriched frame missing columns: {missing}"


def _check_b_live_runtime() -> None:
    _log("b) live signal runtime factor computation")
    payload = _load_example_strategy_payload()
    frame = _build_ohlcv(rows=380)
    bars = _bars_from_frame(frame)

    runtime = LiveSignalRuntime()
    decision = runtime.evaluate(strategy_payload=payload, bars=bars)
    assert decision.signal in {"OPEN_LONG", "OPEN_SHORT", "CLOSE", "NOOP"}
    assert decision.reason


def _check_d_registry_and_mcp_catalog() -> None:
    _log("d) registry + MCP indicator catalog/detail")
    catalog_raw = strategy_tools.get_indicator_catalog()
    catalog = json.loads(catalog_raw)
    assert catalog["ok"] is True
    assert int(catalog.get("count", 0)) > 0

    detail_raw = strategy_tools.get_indicator_detail(indicator_list=["ema", "macd", "rsi"])
    detail = json.loads(detail_raw)
    assert detail["ok"] is True
    names = {item.get("indicator") for item in detail.get("indicators", [])}
    assert {"ema", "macd", "rsi"}.issubset(names)

    all_names = set(IndicatorRegistry.list_all())
    assert "ema" in all_names and "macd" in all_names and "rsi" in all_names


async def _check_c_pre_strategy_regime_tool() -> None:
    _log("c) pre-strategy regime snapshot MCP tool")
    frame = _build_ohlcv(rows=820)

    original = market_tools._resolve_regime_frame

    async def _fake_resolve_regime_frame(**kwargs: Any) -> tuple[pd.DataFrame, str, str]:
        del kwargs
        return frame.copy(), "local_primary", "local_parquet"

    market_tools._resolve_regime_frame = _fake_resolve_regime_frame
    try:
        raw = await market_tools.pre_strategy_get_regime_snapshot(
            market="crypto",
            symbol="BTCUSD",
            opportunity_frequency_bucket="daily",
            holding_period_bucket="swing_days",
            lookback_bars=500,
        )
        payload = json.loads(raw)
        assert payload["ok"] is True
        assert isinstance(payload.get("primary"), dict)
        assert isinstance(payload.get("secondary"), dict)
        primary = payload["primary"]
        assert "family_scores" in primary and "features" in primary
        assert primary["family_scores"]["recommended_family"] in {
            "trend_continuation",
            "mean_reversion",
            "volatility_regime",
        }
    finally:
        market_tools._resolve_regime_frame = original


def _check_e_trade_snapshot_pane_resolution() -> None:
    _log("e) trade snapshot overlay/append pane resolution")
    payload = _load_example_strategy_payload()
    parsed = build_parsed_strategy(payload)
    frame = _build_ohlcv(rows=420)
    enriched = prepare_backtest_frame(frame, strategy=parsed)

    series = _resolve_factor_series(frame_columns=list(enriched.columns), strategy=parsed)
    assert series, "Expected factor series specs for trade snapshot"

    by_id: dict[str, list[Any]] = {}
    for item in series:
        by_id.setdefault(item.factor_id, []).append(item)

    ema_specs = by_id.get("ema_9", [])
    rsi_specs = by_id.get("rsi_14", [])
    assert ema_specs and rsi_specs
    assert all(spec.pane_id == "price" for spec in ema_specs)
    assert any(spec.pane_id != "price" for spec in rsi_specs)


def _check_f_semantic_contract_paths() -> None:
    _log("f) semantic validation and alias compatibility")
    payload = _load_example_strategy_payload()
    with_alias = copy.deepcopy(payload)
    macd_factor = with_alias.get("factors", {}).get("macd_12_26_9")
    if isinstance(macd_factor, dict):
        # Legacy alias compatibility currently applies when outputs are inferred.
        macd_factor.pop("outputs", None)

    # Ensure legacy alias path remains semantically accepted.
    long_entry = with_alias["trade"]["long"]["entry"]["condition"]["all"]
    long_entry.append(
        {
            "cmp": {
                "left": {"ref": "macd_12_26_9.MACDh"},
                "op": "gt",
                "right": 0,
            }
        }
    )

    errors = validate_strategy_semantics(with_alias)
    bad = [e for e in errors if e.code in {"INVALID_OUTPUT_NAME", "UNKNOWN_FACTOR_REF"}]
    assert not bad, f"Semantic alias compatibility broke: {[e.code for e in bad]}"


def _check_step1_removals() -> None:
    _log("step1 assertions: removed factors unavailable")
    removed = {
        "beta",
        "correl",
        "ht_dcperiod",
        "ht_dcphase",
        "ht_phasor",
        "ht_sine",
        "ht_trendmode",
        "cdl_doji",
        "cdl_engulfing",
        "ha",
    }
    all_names = set(IndicatorRegistry.list_all())
    still = sorted(name for name in removed if name in all_names)
    assert not still, f"Removed factors still registered: {still}"
    stale_candle = sorted(name for name in all_names if name.startswith("cdl_") or name == "ha")
    assert not stale_candle, f"Candle factors still registered: {stale_candle}"


def _check_step2_regime_atomic_factors() -> None:
    _log("step2 assertions: regime atomic factors registered")
    expected = {
        "efficiency_ratio",
        "directional_persistence",
        "sign_autocorrelation",
        "breakout_frequency",
        "false_breakout_frequency",
        "volatility_regime_ratio",
        "atr_regime_ratio",
        "squeeze_score",
        "parkinson_volatility",
        "garman_klass_volatility",
        "dry_up_reversal_hint",
    }
    all_names = set(IndicatorRegistry.list_all())
    missing = sorted(name for name in expected if name not in all_names)
    assert not missing, f"Expected regime factors missing: {missing}"


def _check_step3_unified_contract_layer() -> None:
    _log("step3 assertions: unified contract layer is source of truth")
    from packages.domain.strategy.feature.contracts import (
        get_contract,
        resolve_indicator_params_from_dsl,
        to_dsl_output_alias,
    )

    macd_contract = get_contract("macd")
    assert macd_contract is not None
    mapped, source = resolve_indicator_params_from_dsl(
        "macd", {"fast": 12, "slow": 26, "signal": 9, "source": "close"}
    )
    assert source == "close"
    assert mapped["fast"] == 12 and mapped["slow"] == 26 and mapped["signal"] == 9
    assert to_dsl_output_alias("macd", "MACDh") == "histogram"


def _check_step4_regime_composer_uses_engine() -> None:
    _log("step4 assertions: regime composer calls unified engine")
    from packages.domain.market_data.regime import feature_snapshot as fs

    assert hasattr(fs, "_compute_with_feature_engine"), "Expected feature-engine bridge helper not found"


def _catalog_indicator_names(catalog_payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for category in catalog_payload.get("categories", []):
        if not isinstance(category, dict):
            continue
        for item in category.get("indicators", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("indicator", "")).strip().lower()
            if name:
                names.add(name)
    return names


def _count_future_drifts(
    *,
    indicator: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    start_bar: int,
) -> int:
    wrapper = IndicatorWrapper()
    full = wrapper.calculate(indicator, frame, **params)
    if not isinstance(full, pd.Series):
        raise AssertionError(f"Expected Series for indicator={indicator}")

    drifts = 0
    for bar_idx in range(start_bar, len(frame)):
        partial = frame.iloc[: bar_idx + 1]
        partial_series = wrapper.calculate(indicator, partial, **params)
        if not isinstance(partial_series, pd.Series):
            raise AssertionError(f"Expected Series for indicator={indicator} on partial replay")
        full_value = full.iloc[bar_idx]
        partial_value = partial_series.iloc[-1]
        if pd.isna(full_value) and pd.isna(partial_value):
            continue
        if pd.isna(full_value) != pd.isna(partial_value):
            drifts += 1
            continue
        if not np.isclose(float(full_value), float(partial_value), atol=1e-9, rtol=1e-7):
            drifts += 1
    return drifts


def _check_p0_single_source_map_and_dry_run() -> None:
    _log("p0-1/2/3) pre-strategy atomic map dry-run and hard-switch checks")
    specs = regime_feature_snapshot._ATOMIC_REGIME_FACTOR_SPECS
    assert specs, "Atomic regime factor spec map is empty"
    for key, spec in specs.items():
        assert isinstance(spec, dict), f"Atomic spec must be dict: {key}"
        assert isinstance(spec.get("indicator"), str) and spec["indicator"], f"Atomic spec missing indicator: {key}"
        assert isinstance(spec.get("params"), dict), f"Atomic spec params must be dict: {key}"

    frame = _build_ohlcv(rows=360)
    factors: dict[str, dict[str, Any]] = {}
    for key, spec in specs.items():
        copied = dict(spec)
        copied["params"] = dict(spec.get("params", {}))
        factors[key] = copied
    result = regime_feature_snapshot._compute_with_feature_engine(
        frame,
        factors=factors,
    )
    assert set(result.keys()) == set(specs.keys()), "Atomic engine output keys diverged from spec map"
    for key, series in result.items():
        assert isinstance(series, pd.Series), f"Atomic factor did not return Series: {key}"
        assert not series.empty, f"Atomic factor returned empty Series: {key}"
        assert series.notna().sum() > 0, f"Atomic factor all-NaN: {key}"

    thresholds = {
        "volatility_relative_error_max": 1e-9,
        "direction_sign_match_min": 0.999,
        "probability_abs_error_max": 1e-9,
    }
    volatility_factors = {
        "volatility_regime_ratio",
        "atr_regime_ratio",
        "parkinson_volatility",
        "garman_klass_volatility",
    }
    direction_factors = {"sign_autocorrelation"}
    probability_factors = set(specs) - volatility_factors - direction_factors

    wrapper = IndicatorWrapper()
    for key, spec in factors.items():
        indicator = str(spec["indicator"]).strip().lower()
        params = dict(spec.get("params", {}))
        source = str(spec.get("source", "close")).strip().lower() or "close"
        direct = wrapper.calculate(indicator, frame, source=source, **params)
        assert isinstance(direct, pd.Series), f"Expected direct Series output for {indicator}"
        mapped = result[key]
        if key in volatility_factors:
            denom = direct.abs().replace(0.0, np.nan)
            rel_err = ((mapped - direct).abs() / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            assert float(rel_err.max()) <= thresholds["volatility_relative_error_max"], (
                f"volatility relative error too high for {key}: {float(rel_err.max())}"
            )
        elif key in direction_factors:
            sign_match = np.sign(mapped.fillna(0.0)).eq(np.sign(direct.fillna(0.0))).mean()
            assert float(sign_match) >= thresholds["direction_sign_match_min"], (
                f"direction sign match too low for {key}: {float(sign_match)}"
            )
        elif key in probability_factors:
            abs_err = (mapped - direct).abs().fillna(0.0)
            assert float(abs_err.max()) <= thresholds["probability_abs_error_max"], (
                f"probability abs error too high for {key}: {float(abs_err.max())}"
            )

    snapshot_path = Path("packages/domain/market_data/regime/feature_snapshot.py")
    source = snapshot_path.read_text(encoding="utf-8")
    assert "engine_factor_series.get(" not in source, "Found legacy fallback branch in feature snapshot"


def _check_p0_causal_metadata_and_strategy_visibility() -> None:
    _log("p0-5/6/7/9) no-lookahead factor availability + output layer")
    fb = IndicatorRegistry.get("false_breakout_frequency")
    dry = IndicatorRegistry.get("dry_up_reversal_hint")
    assert fb is not None and dry is not None

    catalog = json.loads(strategy_tools.get_indicator_catalog())
    names = _catalog_indicator_names(catalog)
    assert "false_breakout_frequency" in names
    assert "dry_up_reversal_hint" in names

    visible_detail = json.loads(
        strategy_tools.get_indicator_detail(
            indicator_list=["false_breakout_frequency", "dry_up_reversal_hint"]
        )
    )
    assert visible_detail.get("ok") is True

    snapshot = regime_feature_snapshot.build_regime_feature_snapshot(
        _build_ohlcv(rows=420),
        timeframe="1h",
        lookback_bars=360,
    )
    meta = snapshot.get("meta", {})
    assert isinstance(meta.get("atomic_factor_policies"), list) and meta["atomic_factor_policies"], "Missing atomic factor policy payload"
    layers = meta.get("field_usage_layer")
    assert isinstance(layers, dict) and layers, "Missing field usage layer map"
    assert layers.get("swing_structure.false_breakout_frequency") == "realtime_feature"
    assert layers.get("volume_participation.dry_up_reversal_hint") == "realtime_feature"


def _check_p0_future_leak_replay() -> None:
    _log("p0-10) expanding-window future-leak replay")
    frame = _build_future_leak_probe_ohlcv(rows=320)
    start_bar = 140

    causal_fb_drift = _count_future_drifts(
        indicator="false_breakout_frequency",
        params={"length": 20, "confirm_bars": 5, "window": 50},
        frame=frame,
        start_bar=start_bar,
    )
    causal_dry_drift = _count_future_drifts(
        indicator="dry_up_reversal_hint",
        params={"length": 120, "quantile": 0.2, "reversal_bars": 1},
        frame=frame,
        start_bar=start_bar,
    )

    assert causal_fb_drift == 0, f"false_breakout_frequency future drift detected: {causal_fb_drift}"
    assert causal_dry_drift == 0, f"dry_up_reversal_hint future drift detected: {causal_dry_drift}"


def _check_p1_decorator_registry_and_validation() -> None:
    _log("p1-1/2/3/4) decorator path + legacy warning + registry validation")
    from packages.domain.strategy.feature.indicators.registry import IndicatorRegistry as Registry

    metadata = Registry.get("ema")
    calculator = Registry.get_calculator("ema")
    assert metadata is not None and calculator is not None

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        Registry.register(
            metadata,
            calculator,
            registration_mode="legacy",
        )
    assert any(item.category is DeprecationWarning for item in captured), "legacy register should emit deprecation warning"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        Registry.register(
            metadata,
            calculator,
            registration_mode="decorator",
        )
    assert not captured, "decorator-mode register should not emit deprecation warning"

    ok, errors = Registry.validate_registry()
    assert ok, f"registry validation failed: {errors}"


def _check_p1_all_categories_migrated() -> None:
    _log("p1-5) all categories migrated off direct register blocks")
    category_paths = [
        path
        for path in Path("packages/domain/strategy/feature/indicators/categories").glob("*.py")
        if path.name != "__init__.py"
    ]
    assert category_paths
    for path in category_paths:
        content = path.read_text(encoding="utf-8")
        assert "IndicatorRegistry.register(" not in content, f"legacy register block still exists: {path}"
        assert "indicator(" in content, f"decorator registration not found in: {path}"


def _check_p1_contract_coverage() -> None:
    _log("p1-6/8) contracts cover multi-output indicators")
    from packages.domain.strategy.feature.contracts import (
        default_outputs_for_indicator,
        missing_multi_output_contracts,
    )
    from packages.domain.strategy.feature.indicators.registry import IndicatorRegistry as Registry

    missing = missing_multi_output_contracts()
    assert not missing, f"multi-output contracts missing for indicators: {missing}"
    for name in Registry.list_multi_output():
        outputs = default_outputs_for_indicator(name)
        assert len(outputs) >= 2, f"multi-output indicator missing default outputs: {name}"


def _check_p1_no_hardcoded_mapping_regression() -> None:
    _log("p1-7) no duplicated private mapping tables in call-site modules")
    targets = [
        Path("packages/domain/strategy/semantic.py"),
        Path("packages/domain/backtest/factors.py"),
        Path("packages/domain/backtest/trade_snapshot.py"),
        Path("apps/mcp/domains/strategy/tools.py"),
    ]
    forbidden = (
        "_DEFAULT_MULTI_OUTPUTS",
        "_DSL_OUTPUT_ALIAS_MAP",
        "_PERIOD_BASED_FACTORS",
    )
    for path in targets:
        content = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in content, f"forbidden mapping token '{token}' found in {path}"


def _check_p2_docs_generation_and_sync() -> None:
    _log("p2-1/2) registry docs generated and in sync")
    result = subprocess.run(
        [sys.executable, "scripts/generate_indicator_docs.py", "--check"],
        cwd=Path("."),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "generated indicator docs are out of date: "
        f"{result.stdout.strip()} {result.stderr.strip()}"
    )
    readme_path = Path("packages/domain/strategy/feature/README.md")
    json_path = Path("packages/domain/strategy/feature/indicator_catalog.generated.json")
    assert readme_path.exists() and json_path.exists()
    readme = readme_path.read_text(encoding="utf-8")
    assert "自动生成" in readme and "scripts/generate_indicator_docs.py" in readme


def _check_p2_skills_disabled_and_tooling_trimmed() -> None:
    _log("p2-3/4/5/6) skills disabled path + strategy tool trimming + prompt updates")
    catalog = json.loads(strategy_tools.get_indicator_catalog())
    assert catalog.get("ok") is True
    assert catalog.get("skills_enabled") is False
    for category in catalog.get("categories", []):
        if not isinstance(category, dict):
            continue
        for item in category.get("indicators", []):
            if not isinstance(item, dict):
                continue
            assert "skill_path" not in item
            assert "skill_summary" not in item

    detail = json.loads(strategy_tools.get_indicator_detail(indicator_list=["ema"]))
    assert detail.get("ok") is True
    assert detail.get("skills_enabled") is False
    indicators = detail.get("indicators", [])
    assert indicators and isinstance(indicators, list)
    assert all("content" not in item and "skill_path" not in item for item in indicators if isinstance(item, dict))

    from apps.api.orchestration import constants as orchestration_constants

    assert "get_indicator_detail" not in orchestration_constants._STRATEGY_SCHEMA_ONLY_TOOL_NAMES
    assert "get_indicator_detail" not in orchestration_constants._STRATEGY_ARTIFACT_OPS_TOOL_NAMES

    strategy_handler_text = Path("apps/api/agents/handlers/strategy_handler.py").read_text(encoding="utf-8")
    assert '"get_indicator_detail"' not in strategy_handler_text

    prompt_files = [
        Path("apps/api/agents/skills/strategy/stages/schema_only.md"),
        Path("apps/api/agents/skills/strategy/skills.md"),
        Path("packages/domain/strategy/assets/SKILL.md"),
    ]
    for path in prompt_files:
        content = path.read_text(encoding="utf-8")
        assert "get_indicator_detail" not in content, f"prompt still references get_indicator_detail: {path}"


def _check_p2_token_baseline_artifact() -> None:
    _log("p2-7) token baseline artifact")
    report_path = Path("dev_docs/indicator_token_cost_baseline_2026-03-17.md")
    assert report_path.exists(), "token baseline report missing"
    content = report_path.read_text(encoding="utf-8")
    assert "All Indicators (random cohort)" in content
    assert "Skill Indicators Only (focused cohort)" in content
    assert "p50 savings ratio" in content


def _capture_logs(logger_obj: logging.Logger) -> tuple[list[str], logging.Handler]:
    messages: list[str] = []

    class _MemoryHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            messages.append(self.format(record))

    handler = _MemoryHandler(level=logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger_obj.addHandler(handler)
    return messages, handler


def _check_p3_versioning_and_mcp_metadata() -> None:
    _log("p3-1/3/4) metadata versioning + MCP lifecycle exposure + governance doc")
    meta = IndicatorRegistry.get("ema")
    assert meta is not None
    signature = meta.get_signature()
    for key in ("version", "status", "deprecated_since", "replacement", "remove_after"):
        assert key in signature, f"missing metadata field in signature: {key}"

    catalog = json.loads(strategy_tools.get_indicator_catalog())
    assert catalog.get("ok") is True
    first_indicator: dict[str, Any] | None = None
    for category in catalog.get("categories", []):
        if not isinstance(category, dict):
            continue
        indicators = category.get("indicators", [])
        if isinstance(indicators, list) and indicators:
            if isinstance(indicators[0], dict):
                first_indicator = indicators[0]
                break
    assert isinstance(first_indicator, dict), "catalog missing indicator payload"
    for key in ("version", "status", "deprecated_since", "replacement", "remove_after"):
        assert key in first_indicator, f"catalog missing lifecycle field: {key}"

    detail = json.loads(strategy_tools.get_indicator_detail(indicator_list=["ema"]))
    registry_payload = detail.get("indicators", [{}])[0].get("registry", {})
    for key in ("version", "status", "deprecated_since", "replacement", "remove_after"):
        assert key in registry_payload, f"detail missing lifecycle field: {key}"

    policy_doc = Path("dev_docs/indicator_deprecation_policy.md")
    assert policy_doc.exists(), "missing deprecation governance document"
    policy_text = policy_doc.read_text(encoding="utf-8")
    assert "弃用窗口规则" in policy_text and "status" in policy_text


def _check_p3_alias_deprecation_warnings() -> None:
    _log("p3-2) alias deprecation warnings observable in semantic/backtest")
    from packages.infra.observability.logger import logger as app_logger

    payload = _load_example_strategy_payload()
    payload["factors"]["macd_12_26_9"]["outputs"] = ["MACD", "MACDs", "MACDh"]
    payload["trade"]["long"]["entry"]["condition"]["all"].append(
        {
            "cmp": {
                "left": {"ref": "macd_12_26_9.MACDh"},
                "op": "gt",
                "right": 0,
            }
        }
    )

    messages, handler = _capture_logs(app_logger)
    try:
        errors = validate_strategy_semantics(payload)
        assert not [e for e in errors if e.code == "INVALID_OUTPUT_NAME"]
        parsed = build_parsed_strategy(payload)
        frame = _build_ohlcv(rows=420)
        prepare_backtest_frame(frame, strategy=parsed)
    finally:
        app_logger.removeHandler(handler)

    joined = "\n".join(messages)
    assert "deprecated output alias" in joined, "expected deprecated alias warning was not emitted"


def _check_p3_cache_layer_and_invalidation() -> None:
    _log("p3-5/6) cache behavior and invalidation")
    regime_feature_snapshot.reset_feature_engine_cache()
    frame = _build_ohlcv(rows=760)

    snapshot_a = regime_feature_snapshot.build_regime_feature_snapshot(
        frame,
        timeframe="1h",
        lookback_bars=500,
        pivot_window=5,
    )
    stats_after_first = regime_feature_snapshot.get_feature_engine_cache_stats()
    assert int(stats_after_first.get("misses", 0)) > 0

    snapshot_b = regime_feature_snapshot.build_regime_feature_snapshot(
        frame,
        timeframe="1h",
        lookback_bars=500,
        pivot_window=5,
    )
    stats_after_second = regime_feature_snapshot.get_feature_engine_cache_stats()
    assert int(stats_after_second.get("hits", 0)) > int(stats_after_first.get("hits", 0))
    assert "feature_engine_cache" in snapshot_b.get("meta", {})

    frame_changed = frame.copy()
    frame_changed.iloc[-1, frame_changed.columns.get_loc("close")] += 1.234
    regime_feature_snapshot.build_regime_feature_snapshot(
        frame_changed,
        timeframe="1h",
        lookback_bars=500,
        pivot_window=5,
    )
    stats_after_changed = regime_feature_snapshot.get_feature_engine_cache_stats()
    assert int(stats_after_changed.get("misses", 0)) > int(stats_after_second.get("misses", 0))
    assert isinstance(snapshot_a, dict) and isinstance(snapshot_b, dict)


def _check_p3_benchmark_artifact() -> None:
    _log("p3-7) performance benchmark artifact")
    report_path = Path("dev_docs/pre_strategy_cache_benchmark_2026-03-17.md")
    assert report_path.exists(), "missing cache benchmark report"
    content = report_path.read_text(encoding="utf-8")
    assert "Cold run latency" in content
    assert "Hot run avg latency" in content
    assert "Cache Stats" in content


def run_checks(mode: str) -> None:
    _check_a_backtest_factor_runtime()
    _check_b_live_runtime()
    _check_d_registry_and_mcp_catalog()
    asyncio.run(_check_c_pre_strategy_regime_tool())
    _check_e_trade_snapshot_pane_resolution()
    _check_f_semantic_contract_paths()

    if mode == "step1":
        _check_step1_removals()
    elif mode == "step2":
        _check_step2_regime_atomic_factors()
    elif mode == "step3":
        _check_step3_unified_contract_layer()
    elif mode == "step4":
        _check_step4_regime_composer_uses_engine()
    elif mode == "p0":
        _check_p0_single_source_map_and_dry_run()
        _check_p0_causal_metadata_and_strategy_visibility()
        _check_p0_future_leak_replay()
    elif mode == "p1":
        _check_p1_decorator_registry_and_validation()
        _check_p1_all_categories_migrated()
        _check_p1_contract_coverage()
        _check_p1_no_hardcoded_mapping_regression()
    elif mode == "p2":
        _check_p2_docs_generation_and_sync()
        _check_p2_skills_disabled_and_tooling_trimmed()
        _check_p2_token_baseline_artifact()
    elif mode == "p3":
        _check_p3_versioning_and_mcp_metadata()
        _check_p3_alias_deprecation_warnings()
        _check_p3_cache_layer_and_invalidation()
        _check_p3_benchmark_artifact()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="smoke",
        choices=["smoke", "step1", "step2", "step3", "step4", "p0", "p1", "p2", "p3"],
    )
    args = parser.parse_args()
    run_checks(args.mode)
    _log(f"all checks passed (mode={args.mode})")


if __name__ == "__main__":
    main()
