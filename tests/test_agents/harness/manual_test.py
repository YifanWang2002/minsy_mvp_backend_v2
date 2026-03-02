#!/usr/bin/env python
"""Manual test script for the orchestrator test harness.

Run with: uv run python tests/test_agents/harness/manual_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

# Setup path
BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Setup environment
os.environ["MINSY_SERVICE"] = "api"
os.environ["MINSY_ENV_FILES"] = ",".join([
    "env/.env.secrets",
    "env/.env.common",
    "env/.env.dev",
    "env/.env.dev.api",
    "env/.env.dev.localtest",
])

# Note: Proxy vars are NOT removed to allow testing with proxies
# If you need to remove them, uncomment the following:
# for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
#     os.environ.pop(key, None)


async def run_full_workflow_test():
    """Test the full workflow from KYC to strategy phase."""
    # Import after env setup
    from packages.infra.db.session import init_postgres
    from packages.infra.db.models.user import User
    from packages.domain.session.services.openai_stream_service import OpenAIResponsesEventStreamer

    from tests.test_agents.harness import (
        ScriptedReply,
        ScriptedUser,
        TestReporter,
    )
    from tests.test_agents.harness.test_runner import OrchestratorTestRunner
    from tests.test_agents.harness.factories import scenario

    print("Initializing database...")
    await init_postgres(ensure_schema=True)

    # Get the session factory after init
    from packages.infra.db import session as db_session_module
    session_maker = db_session_module.AsyncSessionLocal

    if session_maker is None:
        raise RuntimeError("AsyncSessionLocal not initialized after init_postgres")

    async with session_maker() as db:
        # Create test user
        user = User(
            id=uuid4(),
            email=f"test_{uuid4().hex[:8]}@test.com",
            password_hash="test_hash",
            name="Test User",
            is_active=True,
        )
        db.add(user)
        await db.flush()
        print(f"Created test user: {user.id}")

        # Create streamer
        streamer = OpenAIResponsesEventStreamer()

        # Build a comprehensive script to go through KYC -> pre_strategy -> strategy
        script = (
            scenario()
            # KYC phase
            .add_message("我想创建一个量化交易策略")
            .add_message("我有3-5年的交易经验")
            .add_message("我的风险承受能力是中等")
            .add_message("我期望15-25%的年化收益")
            # Pre-strategy phase
            .add_message("我想交易美股")
            .add_message("我要交易苹果股票 AAPL")
            .add_message("每天交易一次")
            .add_message("持仓几天")
            # Strategy phase - request strategy creation
            .add_message("好的，请帮我创建一个基于均线交叉的策略")
            .build()
        )

        print(f"Script has {script.total_count} messages")
        print("Starting conversation...")
        print("=" * 60)

        runner = OrchestratorTestRunner(
            db=db,
            streamer=streamer,
            user=user,
            language="zh",
        )

        observation = await runner.run_conversation(
            script,
            max_turns=15,
            stop_on_phase="deployment",  # Stop if we reach deployment
        )

        print("=" * 60)
        print("Conversation completed!")
        print()

        # Print detailed report
        reporter = TestReporter(observation)
        reporter.print_summary()

        # Print token breakdown
        breakdown = reporter.get_token_breakdown()
        print("\nToken Breakdown by Phase:")
        for phase, stats in breakdown["by_phase"].items():
            print(f"  {phase}: {stats['total']:,} tokens ({stats['turns']} turns)")

        # Print latency breakdown
        latency = reporter.get_latency_breakdown()
        print(f"\nLatency: avg={latency['average_ms']:.0f}ms, min={latency['min_ms']:.0f}ms, max={latency['max_ms']:.0f}ms")

        # Rollback to not persist test data
        await db.rollback()

        return observation


async def run_single_turn_test():
    """Quick single turn test."""
    from packages.infra.db.session import init_postgres, AsyncSessionLocal, session_factory
    from packages.infra.db.models.user import User
    from packages.domain.session.services.openai_stream_service import OpenAIResponsesEventStreamer
    from tests.test_agents.harness.test_runner import QuickTestRunner

    print("Initializing database...")
    await init_postgres(ensure_schema=True)

    # Get the session factory after init
    from packages.infra.db import session as db_session_module
    session_maker = db_session_module.AsyncSessionLocal

    if session_maker is None:
        raise RuntimeError("AsyncSessionLocal not initialized after init_postgres")

    async with session_maker() as db:
        user = User(
            id=uuid4(),
            email=f"test_{uuid4().hex[:8]}@test.com",
            password_hash="test_hash",
            name="Test User",
            is_active=True,
        )
        db.add(user)
        await db.flush()

        streamer = OpenAIResponsesEventStreamer()
        runner = QuickTestRunner(db, streamer, user, language="zh")

        print("Sending single message...")
        obs = await runner.send("我想创建一个交易策略")

        print(f"\nPhase: {obs.phase}")
        print(f"Tokens: {obs.total_tokens} (in: {obs.input_tokens}, out: {obs.output_tokens})")
        print(f"Latency: {obs.latency_ms:.0f}ms")
        print(f"Instructions sent: {obs.instructions_sent}")
        print(f"Tools available: {len(obs.tools)}")
        print(f"\nResponse:\n{obs.cleaned_text[:500]}...")

        await db.rollback()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the orchestrator harness")
    parser.add_argument(
        "--mode",
        choices=["single", "full"],
        default="single",
        help="Test mode: single turn or full workflow",
    )
    args = parser.parse_args()

    if args.mode == "single":
        asyncio.run(run_single_turn_test())
    else:
        asyncio.run(run_full_workflow_test())

    print("\nTest completed successfully!")
