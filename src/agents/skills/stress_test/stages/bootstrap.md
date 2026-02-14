[STAGE_MARKER_STRESS_TEST_BOOTSTRAP]
Stage objective:
- Start or resume the first backtest run for the confirmed strategy.
- Keep actions operational: create job (`run_now=false` for first run), then poll until terminal status.
- Do not start deep strategy-iteration discussion until a first terminal result is available.
