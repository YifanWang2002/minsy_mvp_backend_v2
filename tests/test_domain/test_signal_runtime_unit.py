from packages.domain.trading.runtime.signal_runtime import LiveSignalRuntime


def test_exit_metadata_keeps_position_side() -> None:
    runtime = LiveSignalRuntime()

    metadata = runtime._exit_metadata(
        stop_price=95.0,
        take_price=110.0,
        side_name="long",
        entry_price=101.5,
    )

    assert metadata["position_side"] == "long"
    assert metadata["current_position_entry_price"] == 101.5
    assert metadata["managed_stop_price"] == 95.0
    assert metadata["managed_take_price"] == 110.0
