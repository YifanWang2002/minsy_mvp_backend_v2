"""Trade-snapshot builders for completed backtest jobs."""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID

import mplfinance as mpf
import pandas as pd

from packages.domain.backtest.factors import prepare_backtest_frame
from packages.domain.backtest.decision_trace import build_trade_decision_trace
from packages.domain.market_data.data import DataLoader
from packages.domain.strategy.feature.contracts import (
    canonicalize_output_name as canonicalize_contract_output_name,
)
from packages.domain.strategy.feature.indicators import IndicatorRegistry
from packages.domain.strategy.models import ParsedStrategyDsl

DEFAULT_BACKTEST_TRADE_SNAPSHOT_RANDOM_SEED = 20260315
_DEFAULT_IMAGE_MAX_BARS = 240
_BACKTEST_TRADE_SNAPSHOT_TEMP_ROOT = (
    Path(__file__).resolve().parents[3] / "temp" / "backtest_trade_snapshots"
)
_TRADE_SNAPSHOT_VISUAL_SPEC_VERSION = "1.1"
_MAX_INDICATOR_WARMUP_BARS = 2000
_MAX_SNAPSHOT_WINDOW_EXPANSION_ATTEMPTS = 8
_LOOKBACK_PARAM_KEYWORDS: tuple[str, ...] = (
    "period",
    "length",
    "window",
    "lookback",
    "slow",
    "fast",
    "signal",
    "span",
    "timeperiod",
)


@dataclass(frozen=True, slots=True)
class _TradeRow:
    index: int
    raw: dict[str, Any]
    entry_time: datetime
    exit_time: datetime


@dataclass(frozen=True, slots=True)
class _FactorSeriesSpec:
    factor_id: str
    factor_type: str
    column: str
    output: str | None
    category: str
    pane_id: str
    pane_title: str
    pane_height_weight: float
    renderer: str


def build_trade_snapshots_from_result(
    *,
    result_payload: dict[str, Any],
    strategy: ParsedStrategyDsl,
    market: str,
    symbol: str,
    timeframe: str,
    selection_mode: str,
    selection_count: int | None,
    lookback_bars: int,
    lookforward_bars: int,
    render_images: bool,
    save_images_to_temp: bool = False,
    job_id: UUID | str | None = None,
    random_seed: int | None = None,
    trade_index: int | None = None,
    loader: DataLoader | None = None,
    include_decision_trace: bool = False,
) -> dict[str, Any]:
    trades = _normalize_trades(result_payload.get("trades"))
    indicator_warmup_bars = _estimate_indicator_warmup_bars(strategy)
    selected = _select_trades(
        trades=trades,
        mode=selection_mode,
        count=selection_count,
        trade_index=trade_index,
        random_seed=random_seed,
    )

    resolved_seed = (
        int(random_seed)
        if random_seed is not None
        else DEFAULT_BACKTEST_TRADE_SNAPSHOT_RANDOM_SEED
    )
    selection_summary: dict[str, Any] = {
        "mode": selection_mode,
        "requested_count": selection_count,
        "selected_count": len(selected),
        "available_count": len(trades),
    }
    if trade_index is not None:
        selection_summary["requested_trade_index"] = int(trade_index)
        selection_summary["selected_trade_index"] = (
            int(selected[0].index) if selected else None
        )
    if selection_mode == "random":
        selection_summary["random_seed"] = resolved_seed

    if not selected:
        return {
            "market": str(market).strip().lower(),
            "symbol": str(symbol).strip(),
            "timeframe": str(timeframe).strip().lower(),
            "timezone": "UTC",
            "visual_spec": {
                "version": _TRADE_SNAPSHOT_VISUAL_SPEC_VERSION,
                "panes": [
                    {
                        "pane_id": "price",
                        "title": "Price",
                        "height_weight": 3.0,
                    }
                ],
                "series_specs": [],
            },
            "selection": selection_summary,
            "window": {
                "lookback_bars": int(lookback_bars),
                "lookforward_bars": int(lookforward_bars),
                "indicator_warmup_bars": int(indicator_warmup_bars),
            },
            "snapshots": [],
            "warnings": [],
        }

    timeframe_minutes = DataLoader.TIMEFRAME_MINUTES.get(str(timeframe).strip().lower())
    if timeframe_minutes is None:
        raise ValueError(f"Unsupported timeframe for trade snapshots: {timeframe}")
    bar_delta = timedelta(minutes=int(timeframe_minutes))

    data_loader = loader or DataLoader()
    prepared, warnings = _load_prepared_snapshot_frame(
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        selected_trades=selected,
        lookback_bars=int(lookback_bars),
        lookforward_bars=int(lookforward_bars),
        indicator_warmup_bars=int(indicator_warmup_bars),
        bar_delta=bar_delta,
        data_loader=data_loader,
    )

    factor_series = _resolve_factor_series(
        frame_columns=list(prepared.columns),
        strategy=strategy,
    )
    factor_columns = [item.column for item in factor_series]
    visual_spec = _build_visual_spec(factor_series)
    snapshots: list[dict[str, Any]] = []
    for item in selected:
        snapshot = _build_single_trade_snapshot(
            frame=prepared,
            strategy=strategy,
            factor_columns=factor_columns,
            trade=item,
            lookback_bars=int(lookback_bars),
            lookforward_bars=int(lookforward_bars),
            include_decision_trace=bool(include_decision_trace),
        )

        if render_images:
            try:
                image_base64 = _render_slice_image_base64(
                    frame=_slice_frame_for_image(
                        frame=prepared,
                        start_index=int(snapshot["slice"]["start_index"]),
                        end_index=int(snapshot["slice"]["end_index"]),
                        factor_columns=factor_columns,
                    ),
                    title=f"{symbol} {timeframe} trade#{item.index}",
                    max_bars=_DEFAULT_IMAGE_MAX_BARS,
                )
                snapshot["image_png_base64"] = image_base64
                if save_images_to_temp:
                    snapshot["image_temp_path"] = str(
                        _persist_snapshot_image_to_temp(
                            image_base64=image_base64,
                            job_id=job_id,
                            trade_index=item.index,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                snapshot["image_png_base64"] = None
                if save_images_to_temp:
                    snapshot["image_temp_path"] = None
                warnings.append(
                    f"Failed to render image for trade_index={item.index}: {type(exc).__name__}"
                )

        snapshots.append(snapshot)

    return {
        "market": str(market).strip().lower(),
        "symbol": str(symbol).strip(),
        "timeframe": str(timeframe).strip().lower(),
        "timezone": "UTC",
        "visual_spec": visual_spec,
        "selection": selection_summary,
        "window": {
            "lookback_bars": int(lookback_bars),
            "lookforward_bars": int(lookforward_bars),
            "indicator_warmup_bars": int(indicator_warmup_bars),
        },
        "snapshots": snapshots,
        "warnings": warnings,
    }


def _normalize_trades(raw: Any) -> list[_TradeRow]:
    if not isinstance(raw, list):
        return []
    output: list[_TradeRow] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        entry_time = _parse_utc_datetime(item.get("entry_time"))
        exit_time = _parse_utc_datetime(item.get("exit_time"))
        if entry_time is None or exit_time is None:
            continue
        if exit_time < entry_time:
            exit_time = entry_time
        output.append(
            _TradeRow(
                index=idx,
                raw=dict(item),
                entry_time=entry_time,
                exit_time=exit_time,
            )
        )
    return output


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _select_trades(
    *,
    trades: list[_TradeRow],
    mode: str,
    count: int | None,
    trade_index: int | None,
    random_seed: int | None,
) -> list[_TradeRow]:
    if not trades:
        return []

    if trade_index is not None:
        index = int(trade_index)
        if index < 0 or index >= len(trades):
            return []
        return [trades[index]]

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "all":
        return list(trades)

    if count is None:
        requested = 20
    else:
        requested = max(1, int(count))
    requested = min(requested, len(trades))

    if normalized_mode == "latest":
        return list(trades[-requested:])
    if normalized_mode == "earliest":
        return list(trades[:requested])
    if normalized_mode == "random":
        seed = (
            int(random_seed)
            if random_seed is not None
            else DEFAULT_BACKTEST_TRADE_SNAPSHOT_RANDOM_SEED
        )
        generator = random.Random(seed)
        selected = generator.sample(trades, requested)
        return list(selected)
    raise ValueError(f"Unsupported selection_mode: {mode}")


def _normalize_frame_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
    normalized = normalized[index.notna()].copy()
    if normalized.empty:
        return normalized
    clean_index = index[index.notna()]
    normalized.index = pd.DatetimeIndex(clean_index)
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized


def _load_prepared_snapshot_frame(
    *,
    market: str,
    symbol: str,
    timeframe: str,
    strategy: ParsedStrategyDsl,
    selected_trades: list[_TradeRow],
    lookback_bars: int,
    lookforward_bars: int,
    indicator_warmup_bars: int,
    bar_delta: timedelta,
    data_loader: DataLoader,
) -> tuple[pd.DataFrame, list[str]]:
    if not selected_trades:
        raise ValueError("Cannot load snapshot frame without selected trades.")

    required_pre_bars = max(0, int(lookback_bars)) + max(0, int(indicator_warmup_bars))
    required_post_bars = max(0, int(lookforward_bars))
    min_entry = min(item.entry_time for item in selected_trades)
    max_exit = max(item.exit_time for item in selected_trades)

    load_start = min_entry - (bar_delta * required_pre_bars)
    load_end = max_exit + (bar_delta * required_post_bars)
    warnings: list[str] = []

    for attempt in range(_MAX_SNAPSHOT_WINDOW_EXPANSION_ATTEMPTS):
        raw_frame = data_loader.load(
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start_date=load_start,
            end_date=load_end,
        )
        prepared = prepare_backtest_frame(raw_frame, strategy=strategy)
        prepared = _normalize_frame_index(prepared)
        if prepared.empty:
            raise ValueError("No OHLCV rows available after loading snapshot window.")

        missing_pre_bars, missing_post_bars = _snapshot_context_shortfall_bars(
            frame_index=prepared.index,
            trades=selected_trades,
            required_pre_bars=required_pre_bars,
            required_post_bars=required_post_bars,
        )
        if missing_pre_bars <= 0 and missing_post_bars <= 0:
            return prepared, warnings

        if attempt >= (_MAX_SNAPSHOT_WINDOW_EXPANSION_ATTEMPTS - 1):
            warnings.append(
                "Trade snapshot context remains partial after retrying window expansion "
                f"(missing_pre_bars={missing_pre_bars}, missing_post_bars={missing_post_bars})."
            )
            return prepared, warnings

        # Expand by the measured shortfall plus a small buffer, then retry.
        pre_expand = max(0, int(missing_pre_bars))
        post_expand = max(0, int(missing_post_bars))
        buffer = max(1, int((pre_expand + post_expand) * 0.1))
        if pre_expand > 0:
            load_start = load_start - (bar_delta * (pre_expand + buffer))
        if post_expand > 0:
            load_end = load_end + (bar_delta * (post_expand + buffer))

    raise RuntimeError("Unexpected trade snapshot window expansion termination.")


def _snapshot_context_shortfall_bars(
    *,
    frame_index: pd.DatetimeIndex,
    trades: list[_TradeRow],
    required_pre_bars: int,
    required_post_bars: int,
) -> tuple[int, int]:
    if frame_index.empty:
        return max(0, int(required_pre_bars)), max(0, int(required_post_bars))

    max_missing_pre = 0
    max_missing_post = 0
    last_position = len(frame_index) - 1
    for trade in trades:
        entry_index = _find_bar_index(index=frame_index, timestamp=trade.entry_time)
        exit_index = _find_bar_index(index=frame_index, timestamp=trade.exit_time)
        if exit_index < entry_index:
            exit_index = entry_index

        available_pre = max(0, int(entry_index))
        available_post = max(0, int(last_position - exit_index))
        max_missing_pre = max(
            max_missing_pre,
            max(0, int(required_pre_bars) - available_pre),
        )
        max_missing_post = max(
            max_missing_post,
            max(0, int(required_post_bars) - available_post),
        )

    return max_missing_pre, max_missing_post


def _resolve_factor_series(
    *,
    frame_columns: list[str],
    strategy: ParsedStrategyDsl,
) -> list[_FactorSeriesSpec]:
    output: list[_FactorSeriesSpec] = []
    seen_series_keys: set[tuple[str, str]] = set()
    factors = (
        strategy.factors if isinstance(getattr(strategy, "factors", None), dict) else {}
    )
    for factor_id, factor_def in factors.items():
        factor_type = str(getattr(factor_def, "factor_type", "")).strip().lower()
        category = _factor_category(factor_type)
        factor_visual = _factor_visual_override(strategy=strategy, factor_id=factor_id)
        for column in frame_columns:
            if column == factor_id or column.startswith(f"{factor_id}."):
                output_name: str | None = None
                if column != factor_id:
                    output_name = column.split(".", 1)[1]
                canonical_output = _canonicalize_output_name(
                    factor_type=factor_type,
                    output_name=output_name,
                )
                dedupe_output = canonical_output or "__value__"
                series_key = (factor_id, dedupe_output)
                if series_key in seen_series_keys:
                    continue
                seen_series_keys.add(series_key)

                output_visual = _output_visual_override(
                    factor_visual=factor_visual,
                    output_name=canonical_output or output_name,
                )
                pane_id = _resolve_series_pane_id(
                    category=category,
                    factor_id=factor_id,
                    factor_visual=factor_visual,
                    output_visual=output_visual,
                )
                pane_title = _resolve_series_pane_title(
                    pane_id=pane_id,
                    factor_id=factor_id,
                    factor_visual=factor_visual,
                    output_visual=output_visual,
                )
                pane_height_weight = _resolve_pane_height_weight(
                    pane_id=pane_id,
                    factor_visual=factor_visual,
                    output_visual=output_visual,
                )
                renderer = _resolve_series_renderer(
                    category=category,
                    factor_type=factor_type,
                    output_name=canonical_output or output_name,
                    factor_visual=factor_visual,
                    output_visual=output_visual,
                )

                output.append(
                    _FactorSeriesSpec(
                        factor_id=factor_id,
                        factor_type=factor_type,
                        column=column,
                        output=canonical_output or output_name,
                        category=category,
                        pane_id=pane_id,
                        pane_title=pane_title,
                        pane_height_weight=pane_height_weight,
                        renderer=renderer,
                    )
                )
    return output


def _build_single_trade_snapshot(
    *,
    frame: pd.DataFrame,
    strategy: ParsedStrategyDsl,
    factor_columns: list[str],
    trade: _TradeRow,
    lookback_bars: int,
    lookforward_bars: int,
    include_decision_trace: bool,
) -> dict[str, Any]:
    index = frame.index
    assert isinstance(index, pd.DatetimeIndex)
    entry_index = _find_bar_index(index=index, timestamp=trade.entry_time)
    exit_index = _find_bar_index(index=index, timestamp=trade.exit_time)
    if exit_index < entry_index:
        exit_index = entry_index

    start_index = max(0, entry_index - int(lookback_bars))
    end_index = min(len(frame) - 1, exit_index + int(lookforward_bars))
    sliced = frame.iloc[start_index : end_index + 1]

    candles: list[dict[str, Any]] = []
    timestamps: list[str] = []
    for ts, row in sliced.iterrows():
        timestamp = _to_iso_utc(ts)
        timestamps.append(timestamp)
        candles.append(
            {
                "timestamp": timestamp,
                "open": _safe_float(row.get("open")),
                "high": _safe_float(row.get("high")),
                "low": _safe_float(row.get("low")),
                "close": _safe_float(row.get("close")),
                "volume": _safe_float(row.get("volume")),
            }
        )

    indicators: dict[str, list[float | None]] = {}
    for column in factor_columns:
        if column not in sliced.columns:
            continue
        indicators[column] = [_safe_float(value) for value in sliced[column].tolist()]

    trade_payload = dict(trade.raw)
    trade_payload["entry_time"] = trade.entry_time.isoformat()
    trade_payload["exit_time"] = trade.exit_time.isoformat()
    entry_offset = entry_index - start_index
    exit_offset = exit_index - start_index

    decision_trace = None
    if include_decision_trace:
        decision_trace = build_trade_decision_trace(
            frame=frame,
            strategy=strategy,
            trade=trade_payload,
            entry_index=entry_index,
            exit_index=exit_index,
            start_index=start_index,
        )

    payload = {
        "trade_uid": _trade_uid(trade=trade),
        "trade_index": trade.index,
        "trade": trade_payload,
        "trade_annotations": _build_trade_annotations(
            trade=trade_payload,
            timestamps=timestamps,
            entry_bar_offset=entry_offset,
            exit_bar_offset=exit_offset,
        ),
        "slice": {
            "start_time": _to_iso_utc(sliced.index[0]),
            "end_time": _to_iso_utc(sliced.index[-1]),
            "start_index": start_index,
            "end_index": end_index,
            "entry_bar_offset": entry_offset,
            "exit_bar_offset": exit_offset,
            "bar_count": len(sliced),
            "timestamps": timestamps,
            "candles": candles,
            "indicators": indicators,
        },
    }
    if decision_trace is not None:
        payload["decision_trace"] = decision_trace
    return payload


def _find_bar_index(
    *,
    index: pd.DatetimeIndex,
    timestamp: datetime,
) -> int:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    position = int(index.searchsorted(ts, side="right")) - 1
    if position < 0:
        return 0
    if position >= len(index):
        return len(index) - 1
    return position


def _to_iso_utc(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize("UTC").isoformat()
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _estimate_indicator_warmup_bars(strategy: ParsedStrategyDsl) -> int:
    factors = getattr(strategy, "factors", {})
    if not isinstance(factors, dict):
        return 0

    max_period = 0
    for factor in factors.values():
        params = getattr(factor, "params", {})
        if not isinstance(params, dict):
            continue
        for key, raw_value in params.items():
            if not _is_lookback_param_key(key):
                continue
            period = _coerce_positive_int(raw_value)
            if period is None:
                continue
            max_period = max(max_period, period)

    if max_period <= 1:
        return 0

    # Double the largest lookback to reduce warmup NaN lag for smoothed indicators.
    return min(int(max_period * 2), _MAX_INDICATOR_WARMUP_BARS)


def _is_lookback_param_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.strip().lower()
    if not lowered:
        return False
    return any(keyword in lowered for keyword in _LOOKBACK_PARAM_KEYWORDS)


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _trade_uid(*, trade: _TradeRow) -> str:
    return f"{trade.index}:{trade.entry_time.isoformat()}:{trade.exit_time.isoformat()}"


def _build_trade_annotations(
    *,
    trade: dict[str, Any],
    timestamps: list[str],
    entry_bar_offset: int,
    exit_bar_offset: int,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    side = str(trade.get("side", "")).strip().lower()
    side_label = "short" if side == "short" else "long"

    entry_time = (
        timestamps[entry_bar_offset]
        if 0 <= entry_bar_offset < len(timestamps)
        else str(trade.get("entry_time", ""))
    )
    exit_time = (
        timestamps[exit_bar_offset]
        if 0 <= exit_bar_offset < len(timestamps)
        else str(trade.get("exit_time", ""))
    )
    entry_price = _safe_float(trade.get("entry_price"))
    exit_price = _safe_float(trade.get("exit_price"))
    pnl = _safe_float(trade.get("pnl"))
    pnl_pct = _safe_float(trade.get("pnl_pct"))
    exit_reason = str(trade.get("exit_reason", "")).strip() or None

    annotations.append(
        {
            "annotation_id": "entry",
            "kind": "trade_entry",
            "time": entry_time,
            "bar_offset": int(entry_bar_offset),
            "price": entry_price,
            "side": side_label,
            "label": "BUY" if side_label == "long" else "SELL",
        }
    )
    annotations.append(
        {
            "annotation_id": "exit",
            "kind": "trade_exit",
            "time": exit_time,
            "bar_offset": int(exit_bar_offset),
            "price": exit_price,
            "side": side_label,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "label": "EXIT",
        }
    )
    annotations.append(
        {
            "annotation_id": "holding_range",
            "kind": "holding_range",
            "start_time": entry_time,
            "end_time": exit_time,
            "start_bar_offset": int(entry_bar_offset),
            "end_bar_offset": int(exit_bar_offset),
            "side": side_label,
        }
    )

    stop_price = _safe_float(
        trade.get("stop_price")
        if trade.get("stop_price") is not None
        else trade.get("stop_loss_price")
    )
    if stop_price is not None:
        annotations.append(
            {
                "annotation_id": "stop_loss",
                "kind": "stop_loss",
                "time": entry_time,
                "price": stop_price,
                "side": side_label,
            }
        )
    take_price = _safe_float(
        trade.get("take_price")
        if trade.get("take_price") is not None
        else trade.get("take_profit_price")
    )
    if take_price is not None:
        annotations.append(
            {
                "annotation_id": "take_profit",
                "kind": "take_profit",
                "time": entry_time,
                "price": take_price,
                "side": side_label,
            }
        )
    return annotations


def _build_visual_spec(factor_series: list[_FactorSeriesSpec]) -> dict[str, Any]:
    panes_map: dict[str, dict[str, Any]] = {
        "price": {
            "pane_id": "price",
            "title": "Price",
            "height_weight": 3.0,
            "order": 0,
        }
    }
    series_specs: list[dict[str, Any]] = []

    for index, series in enumerate(factor_series, start=1):
        if series.pane_id not in panes_map:
            panes_map[series.pane_id] = {
                "pane_id": series.pane_id,
                "title": series.pane_title,
                "height_weight": float(series.pane_height_weight),
                "order": index,
            }
        series_id = (
            f"{series.factor_id}.{series.output}" if series.output else series.factor_id
        )
        series_specs.append(
            {
                "series_id": series_id,
                "factor_id": series.factor_id,
                "factor_type": series.factor_type,
                "column": series.column,
                "output": series.output,
                "category": series.category,
                "pane_id": series.pane_id,
                "renderer": series.renderer,
                "style": {"line_width": 1},
            }
        )

    panes = [
        {
            "pane_id": pane["pane_id"],
            "title": pane["title"],
            "height_weight": pane["height_weight"],
        }
        for pane in sorted(
            panes_map.values(), key=lambda item: int(item.get("order", 0))
        )
    ]
    return {
        "version": _TRADE_SNAPSHOT_VISUAL_SPEC_VERSION,
        "panes": panes,
        "series_specs": series_specs,
    }


def _factor_visual_override(
    *,
    strategy: ParsedStrategyDsl,
    factor_id: str,
) -> dict[str, Any]:
    raw = getattr(strategy, "raw", {})
    if not isinstance(raw, dict):
        return {}
    raw_factors = raw.get("factors")
    if not isinstance(raw_factors, dict):
        return {}
    factor_def = raw_factors.get(factor_id)
    if not isinstance(factor_def, dict):
        return {}
    value = factor_def.get("x-visual")
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _output_visual_override(
    *,
    factor_visual: dict[str, Any],
    output_name: str | None,
) -> dict[str, Any]:
    if not output_name:
        return {}
    output_map = factor_visual.get("outputs")
    if not isinstance(output_map, dict):
        return {}
    direct = output_map.get(output_name)
    if isinstance(direct, dict):
        return dict(direct)
    lower_name = output_name.lower()
    for key, value in output_map.items():
        if (
            isinstance(key, str)
            and key.strip().lower() == lower_name
            and isinstance(value, dict)
        ):
            return dict(value)
    return {}


def _factor_category(factor_type: str) -> str:
    if not factor_type:
        return "unknown"
    metadata = IndicatorRegistry.get(factor_type)
    if metadata is None:
        return "unknown"
    return metadata.category.value


def _canonicalize_output_name(
    *,
    factor_type: str,
    output_name: str | None,
) -> str | None:
    return canonicalize_contract_output_name(factor_type, output_name)


def _resolve_series_pane_id(
    *,
    category: str,
    factor_id: str,
    factor_visual: dict[str, Any],
    output_visual: dict[str, Any],
) -> str:
    overlay_value = output_visual.get("overlay")
    if overlay_value is None:
        overlay_value = factor_visual.get("overlay")
    if overlay_value is True:
        return "price"

    pane_value = output_visual.get("pane")
    if pane_value is None:
        pane_value = factor_visual.get("pane")
    pane = str(pane_value).strip().lower() if pane_value is not None else ""
    if pane in {"price", "overlay", "main"}:
        return "price"
    if pane:
        return _safe_pane_token(pane)

    if category == "overlap" or category == "unknown":
        return "price"
    return _safe_pane_token(f"{category}_{factor_id}")


def _resolve_series_pane_title(
    *,
    pane_id: str,
    factor_id: str,
    factor_visual: dict[str, Any],
    output_visual: dict[str, Any],
) -> str:
    if pane_id == "price":
        return "Price"
    output_title = output_visual.get("pane_title")
    if isinstance(output_title, str) and output_title.strip():
        return output_title.strip()
    factor_title = factor_visual.get("pane_title")
    if isinstance(factor_title, str) and factor_title.strip():
        return factor_title.strip()
    return factor_id


def _resolve_pane_height_weight(
    *,
    pane_id: str,
    factor_visual: dict[str, Any],
    output_visual: dict[str, Any],
) -> float:
    if pane_id == "price":
        return 3.0
    output_weight = output_visual.get("height_weight")
    if isinstance(output_weight, int | float) and float(output_weight) > 0:
        return float(output_weight)
    factor_weight = factor_visual.get("height_weight")
    if isinstance(factor_weight, int | float) and float(factor_weight) > 0:
        return float(factor_weight)
    return 1.0


def _resolve_series_renderer(
    *,
    category: str,
    factor_type: str,
    output_name: str | None,
    factor_visual: dict[str, Any],
    output_visual: dict[str, Any],
) -> str:
    output_renderer = output_visual.get("renderer")
    if isinstance(output_renderer, str) and output_renderer.strip():
        return _normalize_renderer(output_renderer)
    factor_renderer = factor_visual.get("renderer")
    if isinstance(factor_renderer, str) and factor_renderer.strip():
        return _normalize_renderer(factor_renderer)

    normalized_output = (output_name or "").strip().lower()
    if factor_type == "macd" and normalized_output in {"hist", "histogram"}:
        return "histogram"
    if category == "volume" and normalized_output in {"volume", "hist", "histogram"}:
        return "histogram"
    return "line"


def _normalize_renderer(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"line", "histogram", "area", "band"}:
        return normalized
    return "line"


def _safe_pane_token(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact or "pane"


def _slice_frame_for_image(
    *,
    frame: pd.DataFrame,
    start_index: int,
    end_index: int,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    sliced = frame.iloc[start_index : end_index + 1].copy()
    columns = ["open", "high", "low", "close", "volume"]
    output = sliced[columns].copy()
    for column in factor_columns or []:
        if column not in sliced.columns:
            continue
        output[column] = pd.to_numeric(sliced[column], errors="coerce")
    return output


def _render_slice_image_base64(
    *,
    frame: pd.DataFrame,
    title: str,
    max_bars: int,
) -> str:
    if frame.empty:
        raise ValueError("Cannot render empty frame.")
    chart_frame = frame.copy().sort_index()
    if len(chart_frame) > int(max_bars):
        chart_frame = chart_frame.tail(int(max_bars))
    chart_frame.index = pd.to_datetime(chart_frame.index, utc=True)

    price_frame = chart_frame[["open", "high", "low", "close", "volume"]]
    indicator_columns = [
        column
        for column in chart_frame.columns
        if column not in {"open", "high", "low", "close", "volume"}
    ]
    addplots: list[Any] = []
    palette = [
        "#ff7f0e",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    for idx, column in enumerate(indicator_columns):
        series = pd.to_numeric(chart_frame[column], errors="coerce")
        if int(series.notna().sum()) == 0:
            continue
        addplots.append(
            mpf.make_addplot(
                series,
                panel=0,
                color=palette[idx % len(palette)],
                width=1.0,
            )
        )

    fig, _ = mpf.plot(
        price_frame,
        type="candle",
        style="charles",
        volume=True,
        title=title,
        addplot=addplots if addplots else None,
        returnfig=True,
        warn_too_much_data=max(int(max_bars), 200),
        figratio=(16, 9),
        figscale=1.1,
    )
    try:
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor="white",
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return encoded
    finally:
        fig.clf()


def _persist_snapshot_image_to_temp(
    *,
    image_base64: str,
    job_id: UUID | str | None,
    trade_index: int,
    symbol: str,
    timeframe: str,
) -> Path:
    job_segment = str(job_id) if job_id is not None else "unknown_job"
    safe_symbol = _sanitize_path_token(symbol)
    safe_timeframe = _sanitize_path_token(timeframe)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = _BACKTEST_TRADE_SNAPSHOT_TEMP_ROOT / job_segment
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"trade_{int(trade_index):05d}_{safe_symbol}_{safe_timeframe}_{timestamp}.png"
    )
    output_path.write_bytes(base64.b64decode(image_base64.encode("ascii")))
    return output_path


def _sanitize_path_token(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_" for char in str(value).strip()
    )
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact or "unknown"
