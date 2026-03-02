"""Pytest configuration for orchestrator tests.

This file registers the harness fixtures for use in test files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend is in path
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Re-export all fixtures from the harness
from tests.test_agents.harness.fixtures import (
    test_db,
    test_user,
    test_user_with_profile,
    openai_streamer,
    mock_streamer,
    orchestrator_runner,
    quick_runner,
    orchestrator_runner_zh,
)

# Make fixtures available to pytest
__all__ = [
    "test_db",
    "test_user",
    "test_user_with_profile",
    "openai_streamer",
    "mock_streamer",
    "orchestrator_runner",
    "quick_runner",
    "orchestrator_runner_zh",
]
