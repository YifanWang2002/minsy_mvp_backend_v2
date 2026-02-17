from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest

from src.agents.handler_registry import init_all_artifacts
from src.engine.strategy import EXAMPLE_PATH, load_strategy_payload, upsert_strategy_dsl
from src.models.session import Session
from src.models.user import User
from src.services.session_title_service import (
    SESSION_TITLE_META_KEY,
    SESSION_TITLE_RECORD_META_KEY,
    read_session_title_from_metadata,
    refresh_session_title,
)


@pytest.mark.asyncio
async def test_refresh_session_title_tracks_kyc_pre_strategy_and_strategy(db_session) -> None:
    user = User(
        email=f"title_service_{uuid4().hex}@example.com",
        password_hash="hash",
        name="title",
    )
    db_session.add(user)
    await db_session.flush()

    session = Session(
        user_id=user.id,
        current_phase="kyc",
        status="active",
        artifacts=init_all_artifacts(),
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    kyc_result = await refresh_session_title(db=db_session, session=session)
    assert kyc_result.title == "KYC-In Progress"
    assert isinstance(kyc_result.record, dict)
    assert kyc_result.record["kind"] == "kyc_in_progress"
    assert session.metadata_[SESSION_TITLE_META_KEY] == "KYC-In Progress"

    artifacts = dict(session.artifacts or {})
    pre_block = dict(artifacts.get("pre_strategy", {}))
    pre_profile = dict(pre_block.get("profile", {}))
    pre_profile.update(
        {
            "target_market": "us_stocks",
            "target_instrument": "SPY",
            "holding_period_bucket": "intraday",
        }
    )
    pre_block["profile"] = pre_profile
    pre_block["missing_fields"] = []
    artifacts["pre_strategy"] = pre_block
    session.artifacts = artifacts
    session.current_phase = "pre_strategy"

    pre_strategy_result = await refresh_session_title(db=db_session, session=session)
    assert pre_strategy_result.title == "US Stocks SPY Intraday Strategy Development"
    assert isinstance(pre_strategy_result.record, dict)
    assert pre_strategy_result.record["kind"] == "pre_strategy_scope"
    assert pre_strategy_result.record["market"] == "us_stocks"
    assert pre_strategy_result.record["instrument"] == "SPY"
    assert pre_strategy_result.record["holding_period_bucket"] == "intraday"

    artifacts = dict(session.artifacts or {})
    strategy_block = dict(artifacts.get("strategy", {}))
    strategy_profile = dict(strategy_block.get("profile", {}))
    strategy_profile.update(
        {
            "strategy_market": "us_stocks",
            "strategy_primary_symbol": "SPY",
            "strategy_name": "Opening Range Breakout",
        }
    )
    strategy_block["profile"] = strategy_profile
    strategy_block["missing_fields"] = []
    artifacts["strategy"] = strategy_block
    session.artifacts = artifacts
    session.current_phase = "strategy"

    strategy_result = await refresh_session_title(db=db_session, session=session)
    assert strategy_result.title == "US Stocks · Opening Range Breakout"
    assert isinstance(strategy_result.record, dict)
    assert strategy_result.record["kind"] == "strategy_named"
    assert strategy_result.record["strategy_name"] == "Opening Range Breakout"
    assert strategy_result.record["market"] == "us_stocks"
    assert session.metadata_[SESSION_TITLE_META_KEY] == "US Stocks · Opening Range Breakout"


@pytest.mark.asyncio
async def test_refresh_session_title_resolves_strategy_name_from_strategy_record(
    db_session,
) -> None:
    user = User(
        email=f"title_service_lookup_{uuid4().hex}@example.com",
        password_hash="hash",
        name="title_lookup",
    )
    db_session.add(user)
    await db_session.flush()

    artifacts = init_all_artifacts()
    artifacts["pre_strategy"]["profile"] = {
        "target_market": "us_stocks",
        "target_instrument": "QQQ",
    }

    session = Session(
        user_id=user.id,
        current_phase="strategy",
        status="active",
        artifacts=artifacts,
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    dsl_payload = deepcopy(load_strategy_payload(EXAMPLE_PATH))
    strategy_block = dsl_payload.get("strategy")
    if isinstance(strategy_block, dict):
        strategy_block["name"] = "QQQ Momentum Breakout"

    persistence = await upsert_strategy_dsl(
        db_session,
        session_id=session.id,
        dsl_payload=dsl_payload,
        auto_commit=False,
    )
    strategy_id = str(persistence.receipt.strategy_id)

    strategy_phase_block = dict((session.artifacts or {}).get("strategy", {}))
    strategy_profile = dict(strategy_phase_block.get("profile", {}))
    strategy_profile.update(
        {
            "strategy_id": strategy_id,
            "strategy_market": "us_stocks",
        }
    )
    strategy_phase_block["profile"] = strategy_profile
    strategy_phase_block["missing_fields"] = []
    artifacts = dict(session.artifacts or {})
    artifacts["strategy"] = strategy_phase_block
    session.artifacts = artifacts
    session.metadata_ = {"strategy_id": strategy_id}

    result = await refresh_session_title(db=db_session, session=session)
    assert result.title == "US Stocks · QQQ Momentum Breakout"
    assert isinstance(result.record, dict)
    assert result.record["kind"] == "strategy_named"
    assert result.record["strategy_name"] == "QQQ Momentum Breakout"
    assert result.record["strategy_id"] == strategy_id
    assert session.metadata_["strategy_name"] == "QQQ Momentum Breakout"


def test_read_session_title_from_metadata_handles_missing_data() -> None:
    empty = read_session_title_from_metadata(None)
    assert empty.title is None
    assert empty.record is None

    payload = {
        SESSION_TITLE_META_KEY: "US Stocks · Sample",
        SESSION_TITLE_RECORD_META_KEY: {
            "kind": "strategy_named",
            "strategy_name": "Sample",
        },
    }
    parsed = read_session_title_from_metadata(payload)
    assert parsed.title == "US Stocks · Sample"
    assert isinstance(parsed.record, dict)
    assert parsed.record["kind"] == "strategy_named"
