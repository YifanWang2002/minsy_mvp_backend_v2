"""OpenAI usage normalization and session-level cost aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

_OPENAI_COST_VERSION = 1


def _to_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value.strip())), 0)
        except ValueError:
            return 0
    return 0


def _to_non_negative_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return max(float(value), 0.0)
    if isinstance(value, str):
        try:
            return max(float(value.strip()), 0.0)
        except ValueError:
            return 0.0
    return 0.0


def _build_counter_block(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    return {
        "turn_count": _to_non_negative_int(raw.get("turn_count")),
        "input_tokens": _to_non_negative_int(raw.get("input_tokens")),
        "output_tokens": _to_non_negative_int(raw.get("output_tokens")),
        "total_tokens": _to_non_negative_int(raw.get("total_tokens")),
        "cost_usd": round(_to_non_negative_float(raw.get("cost_usd")), 6),
    }


def _to_at_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def normalize_openai_usage(raw_usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    """Normalize usage payloads from different OpenAI response formats."""
    if not isinstance(raw_usage, Mapping):
        return None

    input_tokens = _to_non_negative_int(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))
    )
    output_tokens = _to_non_negative_int(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
    )
    total_tokens = _to_non_negative_int(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_model_pricing(
    *,
    pricing: Mapping[str, Any] | None,
    model: str,
) -> tuple[float, float]:
    if not isinstance(pricing, Mapping):
        return (0.0, 0.0)

    raw_model_entry = pricing.get(model)
    if not isinstance(raw_model_entry, Mapping):
        raw_model_entry = pricing.get("default")
    if not isinstance(raw_model_entry, Mapping):
        return (0.0, 0.0)

    input_per_token = _to_non_negative_float(raw_model_entry.get("input_per_token"))
    output_per_token = _to_non_negative_float(raw_model_entry.get("output_per_token"))
    if input_per_token <= 0:
        input_per_token = _to_non_negative_float(raw_model_entry.get("input_per_1k_tokens")) / 1000.0
    if output_per_token <= 0:
        output_per_token = (
            _to_non_negative_float(raw_model_entry.get("output_per_1k_tokens")) / 1000.0
        )
    return (input_per_token, output_per_token)


def build_turn_usage_snapshot(
    *,
    raw_usage: Mapping[str, Any] | None,
    model: str | None,
    response_id: str | None,
    at: datetime | None = None,
    pricing: Mapping[str, Any] | None = None,
    cost_tracking_enabled: bool = True,
) -> dict[str, Any] | None:
    """Build one normalized turn usage payload to persist on message token_usage."""
    normalized = normalize_openai_usage(raw_usage)
    if normalized is None:
        return None

    resolved_model = (model or "").strip() or "unknown"
    snapshot: dict[str, Any] = {
        "model": resolved_model,
        "input_tokens": normalized["input_tokens"],
        "output_tokens": normalized["output_tokens"],
        "total_tokens": normalized["total_tokens"],
    }
    if isinstance(response_id, str) and response_id.strip():
        snapshot["response_id"] = response_id.strip()
    snapshot["at"] = _to_at_timestamp(at)

    if cost_tracking_enabled:
        input_per_token, output_per_token = _extract_model_pricing(
            pricing=pricing,
            model=resolved_model,
        )
        cost_usd = (
            snapshot["input_tokens"] * input_per_token
            + snapshot["output_tokens"] * output_per_token
        )
        snapshot["cost_usd"] = round(cost_usd, 6)

    return snapshot


def merge_session_openai_cost_metadata(
    metadata: Mapping[str, Any] | None,
    turn_usage: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Merge one turn usage snapshot into `Session.metadata.openai_cost`."""
    next_metadata = dict(metadata or {})
    if not isinstance(turn_usage, Mapping):
        return next_metadata, read_session_openai_cost_totals(next_metadata)

    openai_cost_raw = next_metadata.get("openai_cost")
    openai_cost = dict(openai_cost_raw) if isinstance(openai_cost_raw, Mapping) else {}

    totals = _build_counter_block(
        openai_cost.get("totals") if isinstance(openai_cost.get("totals"), Mapping) else None
    )
    model = str(turn_usage.get("model") or "unknown").strip() or "unknown"
    by_model_raw = openai_cost.get("by_model")
    by_model = dict(by_model_raw) if isinstance(by_model_raw, Mapping) else {}
    model_totals = _build_counter_block(
        by_model.get(model) if isinstance(by_model.get(model), Mapping) else None
    )

    turn_input = _to_non_negative_int(turn_usage.get("input_tokens"))
    turn_output = _to_non_negative_int(turn_usage.get("output_tokens"))
    turn_total = _to_non_negative_int(turn_usage.get("total_tokens"))
    if turn_total == 0:
        turn_total = turn_input + turn_output
    turn_cost = round(_to_non_negative_float(turn_usage.get("cost_usd")), 6)

    for block in (totals, model_totals):
        block["turn_count"] += 1
        block["input_tokens"] += turn_input
        block["output_tokens"] += turn_output
        block["total_tokens"] += turn_total
        block["cost_usd"] = round(block["cost_usd"] + turn_cost, 6)

    by_model[model] = model_totals
    openai_cost["version"] = _OPENAI_COST_VERSION
    openai_cost["totals"] = totals
    openai_cost["by_model"] = by_model
    openai_cost["last_turn"] = {
        "model": model,
        "response_id": str(turn_usage.get("response_id") or ""),
        "input_tokens": turn_input,
        "output_tokens": turn_output,
        "total_tokens": turn_total,
        "cost_usd": turn_cost,
        "at": _to_at_timestamp(turn_usage.get("at")),
    }
    next_metadata["openai_cost"] = openai_cost
    return next_metadata, totals


def read_session_openai_cost_totals(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read existing session cost totals from metadata if present."""
    if not isinstance(metadata, Mapping):
        return None
    openai_cost_raw = metadata.get("openai_cost")
    if not isinstance(openai_cost_raw, Mapping):
        return None
    totals_raw = openai_cost_raw.get("totals")
    if not isinstance(totals_raw, Mapping):
        return None
    return _build_counter_block(totals_raw)

