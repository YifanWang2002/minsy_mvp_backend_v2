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

from apps.api.agents.genui_registry import normalize_genui_payloads
from apps.api.agents.handler_protocol import PhaseContext
from apps.api.agents.handler_protocol import RuntimePolicy as HandlerRuntimePolicy
from apps.api.agents.handler_registry import WORKFLOW_PHASES, get_handler, init_all_artifacts
from apps.api.agents.phases import Phase, SessionStatus, can_transition
from apps.api.agents.skills.pre_strategy_skills import (
    get_tradingview_symbol_for_market_instrument,
)
from apps.api.schemas.requests import ChatSendRequest
from packages.shared_settings.schema.settings import settings
from packages.domain.strategy import create_strategy_draft, validate_strategy_payload
from packages.infra.auth.mcp_context import MCP_CONTEXT_HEADER, create_mcp_context_token
from packages.infra.db.models.phase_transition import PhaseTransition
from packages.infra.db.models.session import Message, Session
from packages.infra.db.models.strategy import Strategy
from packages.infra.db.models.user import User, UserProfile
from packages.infra.observability.openai_cost import (
    build_turn_usage_snapshot,
    merge_session_openai_cost_metadata,
)
from apps.api.orchestration.openai_stream_service import ResponsesEventStreamer
from packages.domain.session.services.session_title_service import refresh_session_title
from apps.api.orchestration.chat_debug_trace import (
    CHAT_TRACE_MODE_COMPACT,
    get_chat_debug_trace,
    record_chat_debug_trace,
)
from packages.infra.observability.logger import log_agent

from .constants import *  # noqa: F403
from .types import _TurnPostProcessResult, _TurnPreparation, _TurnStreamState

# Re-export every imported symbol (including single-underscore constants/types)
# so mixin modules can rely on `from .shared import *` safely.
__all__ = [
    name
    for name in globals()
    if not (name.startswith("__") and name.endswith("__"))
]
