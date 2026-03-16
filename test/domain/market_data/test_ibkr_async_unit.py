from __future__ import annotations

from datetime import date

from packages.infra.providers.market_data.ibkr_async import (
    _normalize_futures_exchange,
    _parse_contract_expiry_date,
)


def test_normalize_futures_exchange_aliases_to_supported_names() -> None:
    assert _normalize_futures_exchange("globex") == "CME"
    assert _normalize_futures_exchange("ECBOT") == "CBOT"
    assert _normalize_futures_exchange("nymex") == "NYMEX"


def test_parse_contract_expiry_date_handles_yyyymmdd() -> None:
    assert _parse_contract_expiry_date("20251219") == date(2025, 12, 19)


def test_parse_contract_expiry_date_handles_yyyymm_as_month_end() -> None:
    assert _parse_contract_expiry_date("202512") == date(2025, 12, 31)


def test_parse_contract_expiry_date_returns_none_for_invalid() -> None:
    assert _parse_contract_expiry_date("bad-value") is None
