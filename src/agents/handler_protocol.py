"""PhaseHandler protocol – the contract every phase agent must satisfy.

The orchestrator dispatches to handlers via a registry keyed by Phase.
Each handler encapsulates:
  - prompt construction (static instructions + dynamic state)
  - field validation rules
  - patch application and artifact mutation
  - optional side-effects (e.g. KYC persists to UserProfile)
  - optional genui filtering
  - completion detection and next-phase determination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class RuntimePolicy:
    """Per-turn runtime controls for prompt/tool exposure."""

    phase_stage: str | None = None
    tool_mode: str = "append"
    allowed_tools: list[dict[str, Any]] | None = None


@dataclass
class PhaseContext:
    """Read-only snapshot of data a handler needs to do its work.

    The orchestrator populates this before calling the handler.
    """

    user_id: UUID
    session_artifacts: dict[str, Any]
    language: str = "en"
    runtime_policy: RuntimePolicy = field(default_factory=RuntimePolicy)


@dataclass
class PromptPieces:
    """What the handler returns for the AI call."""

    instructions: str
    enriched_input: str
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    # Per-phase model override. None = use the global default from settings.
    model: str | None = None
    # Reasoning configuration for o-series / gpt-5+ models.
    # Example: {"effort": "none"} or {"effort": "low", "summary": "concise"}
    reasoning: dict[str, Any] | None = None


@dataclass
class PostProcessResult:
    """What the handler returns after processing the AI's output."""

    # Updated artifacts dict (orchestrator will write this to session)
    artifacts: dict[str, Any]
    # Fields still missing in this phase
    missing_fields: list[str]
    # Whether this phase is complete
    completed: bool
    # If completed, which phase to transition to
    next_phase: str | None = None
    # Reason string for the PhaseTransition audit record
    transition_reason: str | None = None
    # Optional per-phase status to include in the done event
    phase_status: dict[str, Any] = field(default_factory=dict)


class PhaseHandler(Protocol):
    """Contract that each phase handler must implement."""

    @property
    def phase_name(self) -> str:
        """The Phase enum value this handler is responsible for."""
        ...

    @property
    def required_fields(self) -> list[str]:
        """Canonical ordered list of fields this phase collects."""
        ...

    @property
    def valid_values(self) -> dict[str, set[str]]:
        """Allowed enum values for each field."""
        ...

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        """Build static instructions + enriched input for the AI call."""
        ...

    async def post_process(
        self,
        ctx: PhaseContext,
        raw_patches: list[dict[str, Any]],
        db: AsyncSession,
    ) -> PostProcessResult:
        """Validate patches, mutate artifacts, persist side-effects.

        The orchestrator passes ``ctx.session_artifacts`` which the handler
        should read/update *in place* (the orchestrator will write the
        updated dict back to the session).
        """
        ...

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        """Optional genui filtering (e.g. restrict instrument options).

        Return the payload (possibly modified), or None to suppress it.
        Default implementations should return the payload unchanged.
        """
        ...

    def init_artifacts(self) -> dict[str, Any]:
        """Return the initial artifacts block for this phase.

        Called by ``create_session`` to seed the session artifacts.
        Example: ``{"profile": {}, "missing_fields": [...]}``
        """
        ...

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        """Return a short user-facing handoff sentence when entering this phase.

        The orchestrator appends this after transitioning from the previous phase.
        Return ``None`` to skip handoff text.
        """
        ...
