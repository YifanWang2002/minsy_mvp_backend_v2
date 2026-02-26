"""Stub phase handler for phases not yet implemented (strategy, stress_test, deployment).

Returns a placeholder response so the orchestrator has something to work with
without crashing. When a real implementation is ready, swap the registration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.agents.handler_protocol import (
    PhaseContext,
    PostProcessResult,
    PromptPieces,
)


class StubHandler:
    """Generic placeholder handler for any unimplemented phase."""

    def __init__(self, phase_name: str, next_phase: str | None = None) -> None:
        self._phase_name = phase_name
        self._next_phase = next_phase

    # -- protocol properties -------------------------------------------

    @property
    def phase_name(self) -> str:
        return self._phase_name

    @property
    def required_fields(self) -> list[str]:
        return []

    @property
    def valid_values(self) -> dict[str, set[str]]:
        return {}

    # -- prompt --------------------------------------------------------

    def build_prompt(
        self,
        ctx: PhaseContext,
        user_message: str,
    ) -> PromptPieces:
        instructions = _build_stub_instructions(
            phase_name=self._phase_name,
            language=ctx.language,
            phase_stage=ctx.runtime_policy.phase_stage,
        )
        return PromptPieces(
            instructions=instructions,
            enriched_input=user_message,
        )

    # -- post-process --------------------------------------------------

    async def post_process(
        self,
        ctx: PhaseContext,
        raw_patches: list[dict[str, Any]],
        db: AsyncSession,
    ) -> PostProcessResult:
        artifacts = ctx.session_artifacts
        phase_data = artifacts.setdefault(self._phase_name, {"profile": {}, "missing_fields": []})
        return PostProcessResult(
            artifacts=artifacts,
            missing_fields=list(phase_data.get("missing_fields", [])),
            completed=False,
        )

    # -- genui ---------------------------------------------------------

    def filter_genui(
        self,
        payload: dict[str, Any],
        ctx: PhaseContext,
    ) -> dict[str, Any] | None:
        return payload

    # -- artifacts init ------------------------------------------------

    def init_artifacts(self) -> dict[str, Any]:
        return {"profile": {}, "missing_fields": []}

    def build_phase_entry_guidance(self, ctx: PhaseContext) -> str | None:
        phase_label = self._phase_name.replace("_", " ").title()
        if ctx.language == "zh":
            return f"接下来进入 {phase_label} 阶段。请继续告诉我你的目标，我会逐步完成该阶段。"
        return (
            f"Next, we are entering the {phase_label} phase. "
            "Share your goal for this stage and we will continue step by step."
        )


_SKILLS_ROOT_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "skills",
)
_SKILLS_ROOT = next((path for path in _SKILLS_ROOT_CANDIDATES if path.exists()), _SKILLS_ROOT_CANDIDATES[0])
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "es": "Español",
    "fr": "Français",
}


def _language_display(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


@lru_cache(maxsize=64)
def _load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_optional_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return _load_text(str(path))


def _build_stub_instructions(
    *,
    phase_name: str,
    language: str,
    phase_stage: str | None,
) -> str:
    phase_dir = _SKILLS_ROOT / phase_name
    phase_template = _load_optional_text(phase_dir / "skills.md")
    if phase_template:
        rendered = phase_template.replace("{{LANG_NAME}}", _language_display(language))
        if isinstance(phase_stage, str) and phase_stage.strip():
            stage_addendum = _load_optional_text(phase_dir / "stages" / f"{phase_stage.strip()}.md")
            if stage_addendum:
                rendered = rendered.rstrip() + "\n\n" + stage_addendum.strip() + "\n"
        return rendered

    phase_label = phase_name.replace("_", " ").title()
    return (
        f"You are the Minsy {phase_label} Agent.\n"
        f"This phase is under development. Acknowledge the user's message and "
        f"let them know the {phase_name} phase will be available soon.\n"
    )
