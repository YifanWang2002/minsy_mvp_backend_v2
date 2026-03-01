from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.api.schemas.requests import BrokerAccountCreateRequest


def test_000_accessibility_sandbox_allows_empty_credentials() -> None:
    payload = BrokerAccountCreateRequest(
        provider="sandbox",
        mode="paper",
        credentials={},
        metadata={"source": "pytest-live"},
    )
    assert payload.provider == "sandbox"
    assert payload.credentials == {}


def test_010_accessibility_alpaca_rejects_empty_credentials() -> None:
    with pytest.raises(ValidationError):
        BrokerAccountCreateRequest(
            provider="alpaca",
            mode="paper",
            credentials={},
            metadata={},
        )
