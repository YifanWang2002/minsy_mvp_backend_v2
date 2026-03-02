"""Pytest fixtures for orchestrator testing.

Provides fixtures for database sessions, test users, and orchestrator runners.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure backend is in path
BACKEND_DIR = Path(__file__).resolve().parents[3]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _setup_test_env() -> None:
    """Configure environment for testing."""
    os.environ.setdefault("MINSY_SERVICE", "api")
    os.environ.setdefault(
        "MINSY_ENV_FILES",
        ",".join([
            "env/.env.secrets",
            "env/.env.common",
            "env/.env.dev",
            "env/.env.dev.api",
            "env/.env.dev.localtest",
        ])
    )
    # Remove proxy variables that can interfere with OpenAI calls
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(key, None)


_setup_test_env()


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="function")
async def test_db() -> AsyncIterator[AsyncSession]:
    """Provide a test database session.

    This fixture initializes the database connection and yields a session.
    The session is rolled back after each test to maintain isolation.
    """
    from packages.infra.db.session import init_postgres, AsyncSessionLocal, close_postgres

    await init_postgres(ensure_schema=True)
    assert AsyncSessionLocal is not None

    async with AsyncSessionLocal() as session:
        yield session
        # Rollback any uncommitted changes
        await session.rollback()


@pytest.fixture(scope="function")
async def test_user(test_db: AsyncSession):
    """Create a test user for orchestrator testing.

    Returns a User model instance that can be used with the orchestrator.
    """
    from packages.infra.db.models.user import User

    user = User(
        id=uuid4(),
        email=f"test_{uuid4().hex[:8]}@test.com",
        password_hash="test_hash",
        name="Test User",
        is_active=True,
    )
    test_db.add(user)
    await test_db.flush()
    return user


@pytest.fixture(scope="function")
async def test_user_with_profile(test_db: AsyncSession, test_user):
    """Create a test user with a completed KYC profile.

    Returns a tuple of (User, UserProfile).
    """
    from packages.infra.db.models.user import UserProfile

    profile = UserProfile(
        user_id=test_user.id,
        trading_years_bucket="years_3_5",
        risk_tolerance="moderate",
        return_expectation="return_15_25",
        kyc_status="complete",
    )
    test_db.add(profile)
    await test_db.flush()
    return test_user, profile


@pytest.fixture(scope="function")
def openai_streamer():
    """Provide a real OpenAI responses event streamer.

    This fixture creates a streamer that makes real API calls.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    from apps.api.orchestration.openai_stream_service import OpenAIResponsesEventStreamer

    return OpenAIResponsesEventStreamer()


@pytest.fixture(scope="function")
def mock_streamer():
    """Provide a mock streamer for unit testing.

    Returns a MockResponsesEventStreamer that doesn't make real API calls.
    """
    from .observable_orchestrator import MockResponsesEventStreamer

    return MockResponsesEventStreamer()


@pytest.fixture(scope="function")
async def orchestrator_runner(test_db: AsyncSession, test_user, openai_streamer):
    """Provide a configured OrchestratorTestRunner.

    This fixture creates a runner with a real database session,
    test user, and OpenAI streamer for end-to-end testing.
    """
    from .test_runner import OrchestratorTestRunner

    return OrchestratorTestRunner(
        db=test_db,
        streamer=openai_streamer,
        user=test_user,
        language="en",
    )


@pytest.fixture(scope="function")
async def quick_runner(test_db: AsyncSession, test_user, openai_streamer):
    """Provide a QuickTestRunner for simple single-turn tests."""
    from .test_runner import QuickTestRunner

    return QuickTestRunner(
        db=test_db,
        streamer=openai_streamer,
        user=test_user,
        language="en",
    )


@pytest.fixture(scope="function")
async def orchestrator_runner_zh(test_db: AsyncSession, test_user, openai_streamer):
    """Provide a runner configured for Chinese language."""
    from .test_runner import OrchestratorTestRunner

    return OrchestratorTestRunner(
        db=test_db,
        streamer=openai_streamer,
        user=test_user,
        language="zh",
    )
