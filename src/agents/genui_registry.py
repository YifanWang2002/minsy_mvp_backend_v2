"""GenUI payload normalizer registry.

The orchestrator should not hardcode payload types. New GenUI payloads can be
added by registering a normalizer for the payload type.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

GenUiNormalizer = Callable[[dict[str, Any]], dict[str, Any] | None]

_GENUI_NORMALIZERS: dict[str, GenUiNormalizer] = {}


def register_genui_normalizer(payload_type: str, normalizer: GenUiNormalizer) -> None:
    key = payload_type.strip()
    if not key:
        return
    _GENUI_NORMALIZERS[key] = normalizer


def get_genui_normalizer(payload_type: str) -> GenUiNormalizer | None:
    return _GENUI_NORMALIZERS.get(payload_type)


def normalize_genui_payloads(
    payloads: list[dict[str, Any]],
    *,
    allow_passthrough_unregistered: bool = True,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for payload in payloads:
        payload_type = payload.get("type")
        if not isinstance(payload_type, str) or not payload_type.strip():
            continue
        payload_type = payload_type.strip()

        normalizer = get_genui_normalizer(payload_type)
        if normalizer is not None:
            candidate = normalizer(payload)
        elif allow_passthrough_unregistered:
            candidate = _normalize_passthrough(payload)
        else:
            candidate = None

        if candidate is None:
            continue

        dedupe_key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(candidate)

    return normalized


def _normalize_passthrough(payload: dict[str, Any]) -> dict[str, Any] | None:
    payload_type = payload.get("type")
    if not isinstance(payload_type, str) or not payload_type.strip():
        return None
    return dict(payload)


def _normalize_choice_prompt(payload: dict[str, Any]) -> dict[str, Any] | None:
    options_in = payload.get("options")
    if not isinstance(options_in, list):
        return None

    options_out: list[dict[str, str]] = []
    for option in options_in:
        if not isinstance(option, dict):
            continue
        option_id = option.get("id")
        if not isinstance(option_id, str) or not option_id.strip():
            continue

        label_value = option.get("label")
        if not isinstance(label_value, str):
            label_value = option.get("label_zh") or option.get("label_en")
        if not isinstance(label_value, str) or not label_value.strip():
            continue

        normalized = {"id": option_id.strip(), "label": label_value.strip()}
        subtitle = option.get("subtitle")
        if isinstance(subtitle, str) and subtitle.strip():
            normalized["subtitle"] = subtitle.strip()
        options_out.append(normalized)

    if len(options_out) < 2:
        return None

    choice_id = payload.get("choice_id")
    if not isinstance(choice_id, str) or not choice_id.strip():
        return None

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None

    result: dict[str, Any] = {
        "type": "choice_prompt",
        "choice_id": choice_id.strip(),
        "question": question.strip(),
        "options": options_out,
    }
    subtitle = payload.get("subtitle")
    if isinstance(subtitle, str) and subtitle.strip():
        result["subtitle"] = subtitle.strip()
    return result


def _normalize_tradingview_chart(payload: dict[str, Any]) -> dict[str, Any] | None:
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None

    interval = payload.get("interval")
    if not isinstance(interval, str) or not interval.strip():
        interval = "D"

    return {
        "type": "tradingview_chart",
        "symbol": symbol.strip(),
        "interval": interval.strip(),
    }


register_genui_normalizer("choice_prompt", _normalize_choice_prompt)
register_genui_normalizer("tradingview_chart", _normalize_tradingview_chart)
