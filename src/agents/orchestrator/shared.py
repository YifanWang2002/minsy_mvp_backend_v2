"""Shared imports for orchestrator mixin modules.

Using a central import module keeps each mixin file focused on behavior.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agents.genui_registry import normalize_genui_payloads
from src.agents.handler_protocol import PhaseContext
from src.agents.handler_protocol import RuntimePolicy as HandlerRuntimePolicy
from src.agents.handler_registry import WORKFLOW_PHASES, get_handler, init_all_artifacts
from src.agents.phases import Phase, SessionStatus, can_transition
from src.agents.skills.pre_strategy_skills import (
    get_tradingview_symbol_for_market_instrument,
)
from src.api.schemas.requests import ChatSendRequest
from src.config import settings
from src.engine.strategy import create_strategy_draft, validate_strategy_payload
from src.mcp.context_auth import MCP_CONTEXT_HEADER, create_mcp_context_token
from src.models.phase_transition import PhaseTransition
from src.models.session import Message, Session
from src.models.strategy import Strategy
from src.models.user import User, UserProfile
from src.observability.openai_cost import (
    build_turn_usage_snapshot,
    merge_session_openai_cost_metadata,
)
from src.services.openai_stream_service import ResponsesEventStreamer
from src.services.session_title_service import refresh_session_title
from src.util.chat_debug_trace import (
    CHAT_TRACE_MODE_COMPACT,
    get_chat_debug_trace,
    record_chat_debug_trace,
)
from src.util.logger import log_agent

from .constants import *  # noqa: F403
from .types import _TurnPostProcessResult, _TurnPreparation, _TurnStreamState

# Re-export every imported symbol (including single-underscore constants/types)
# so mixin modules can rely on `from .shared import *` safely.
__all__ = [
    name
    for name in globals()
    if not (name.startswith("__") and name.endswith("__"))
]
