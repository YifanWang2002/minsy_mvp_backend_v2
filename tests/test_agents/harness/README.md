# Orchestrator Test Harness

A comprehensive testing infrastructure for the Minsy ChatOrchestrator, providing full observability into AI interactions.

## Features

- **Full Observability**: Capture all data sent to and received from the AI
  - Instructions, tools, enriched input
  - Raw responses, patches, GenUI payloads
  - Token usage and latency metrics
  - Phase transitions and artifact mutations

- **Scripted Users**: Simulate user interactions with pre-defined replies
  - Sequential scripts for deterministic testing
  - Conditional logic for adaptive responses
  - Composite users combining both approaches

- **Real Architecture**: Tests use the actual orchestrator, handlers, and skills
  - No mocking of core business logic
  - Real OpenAI API calls (external integration tests)
  - Full phase transition and artifact handling

- **Rich Reporting**: Generate detailed test reports
  - Console summaries
  - Markdown reports with full turn details
  - JSON export for programmatic analysis

## Quick Start

### Single Turn Test

```python
from tests.test_agents.harness import QuickTestRunner

async def test_single_turn(test_db, test_user, openai_streamer):
    runner = QuickTestRunner(test_db, openai_streamer, test_user)
    obs = await runner.send("I want to create a trading strategy")

    print(f"Phase: {obs.phase}")
    print(f"Tokens: {obs.total_tokens}")
    print(f"Response: {obs.cleaned_text}")
```

### Multi-Turn Conversation

```python
from tests.test_agents.harness import (
    OrchestratorTestRunner,
    ScriptedUser,
    ScriptedReply,
    TestReporter,
)

async def test_kyc_flow(test_db, test_user, openai_streamer):
    user = ScriptedUser([
        ScriptedReply("I want to create a strategy"),
        ScriptedReply("3-5 years of experience"),
        ScriptedReply("Moderate risk tolerance"),
        ScriptedReply("15-25% annual returns"),
    ])

    runner = OrchestratorTestRunner(test_db, openai_streamer, test_user)
    observation = await runner.run_conversation(
        user,
        max_turns=10,
        stop_on_phase="pre_strategy",
    )

    # Print summary
    TestReporter(observation).print_summary()

    # Verify results
    assert "kyc" in observation.phases_visited
    assert observation.final_phase == "pre_strategy"
```

### Using Factories

```python
from tests.test_agents.harness.factories import (
    create_kyc_user,
    quick_full_workflow_user,
    scenario,
)

# Pre-built KYC completion user
user = create_kyc_user("en")

# Full workflow user (KYC + pre-strategy)
user = quick_full_workflow_user("zh")

# Custom scenario builder
user = (
    scenario()
    .add_message("Hello")
    .add_kyc_completion("en")
    .add_pre_strategy_us_stocks()
    .build()
)
```

### Conditional Users

```python
from tests.test_agents.harness import ConditionalUser, ScriptedReply

def decide(turn):
    if "risk" in turn.cleaned_text.lower():
        return ScriptedReply("Moderate risk")
    if turn.phase != "kyc":
        return None  # Stop
    return ScriptedReply("Continue")

user = ConditionalUser(decide, max_turns=10)
```

## Components

### Types (`types.py`)

- `TurnObservation`: Complete data for a single turn
- `ConversationObservation`: Aggregated data for entire conversation

### Observer (`observer.py`)

- `TurnObserver`: Captures data at each orchestrator stage

### Observable Orchestrator (`observable_orchestrator.py`)

- `ObservableChatOrchestrator`: Instrumented orchestrator wrapper
- `MockResponsesEventStreamer`: Mock streamer for unit tests

### Scripted User (`scripted_user.py`)

- `ScriptedReply`: Single pre-defined reply
- `ScriptedUser`: Sequential reply playback
- `ConditionalUser`: Dynamic reply selection
- `CompositeUser`: Combines scripted + conditional

### Test Runner (`test_runner.py`)

- `OrchestratorTestRunner`: Full conversation runner
- `QuickTestRunner`: Simplified single-turn runner

### Reporter (`reporter.py`)

- `TestReporter`: Report generation (markdown, JSON, console)

### Fixtures (`fixtures.py`)

- `test_db`: Database session
- `test_user`: Test user model
- `openai_streamer`: Real OpenAI streamer
- `orchestrator_runner`: Configured test runner

### Factories (`factories.py`)

- Pre-built scripts for common scenarios
- `ScenarioBuilder` for custom test scenarios

## Running Tests

```bash
# Run all orchestrator tests
uv run pytest tests/test_agents/ -v

# Run only external (real API) tests
uv run pytest tests/test_agents/ -v -m external

# Run with detailed output
uv run pytest tests/test_agents/test_orchestrator_e2e.py -v -s
```

## Observation Data

Each `TurnObservation` captures:

```python
@dataclass
class TurnObservation:
    # Input
    user_message: str
    phase: str
    phase_stage: str | None
    session_state_snapshot: dict

    # Sent to AI
    instructions: str
    instructions_sent: bool
    enriched_input: str
    tools: list[dict]
    model: str
    max_output_tokens: int | None

    # AI Response
    raw_response_text: str
    cleaned_text: str
    extracted_genui: list[dict]
    mcp_tool_calls: list[dict]

    # Results
    artifacts_before: dict
    artifacts_after: dict
    missing_fields: list[str]
    phase_transition: tuple[str, str] | None

    # Metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
```

## Report Example

```
============================================================
ORCHESTRATOR TEST SUMMARY
============================================================
Session ID: abc-123-def
Total Turns: 4
Total Tokens: 8,450 (in: 6,200, out: 2,250)
Total Duration: 12.34s
Phases Visited: kyc -> pre_strategy
Final Phase: pre_strategy

Phase Transitions: 1
  kyc -> pre_strategy

Turn Summary:
  #1: [kyc] | 2,100 tokens | 3,200ms
       User: I want to create a trading strategy...
       AI: Welcome! Let me help you get started...
  #2: [kyc] | 1,800 tokens | 2,800ms
       User: 3-5 years of experience...
       AI: Great, you have solid experience...
  ...
============================================================
```
