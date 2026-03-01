"""Multiple baseline trend-following strategies for price action research.

All strategies follow similar principles (trend-following, momentum-based)
to test whether price action factors can enhance them consistently.
"""

from __future__ import annotations


def create_ema_crossover_strategy() -> dict:
    """EMA crossover - classic trend following."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "EMA Crossover",
            "description": "20/50 EMA crossover trend following",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "ema_fast": {
                "type": "ema",
                "params": {"source": "close", "period": 20},
            },
            "ema_slow": {
                "type": "ema",
                "params": {"source": "close", "period": 50},
            },
            "atr_14": {
                "type": "atr",
                "params": {"period": 14},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "cross": {
                            "a": {"ref": "ema_fast"},
                            "op": "cross_above",
                            "b": {"ref": "ema_slow"},
                        }
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop_loss",
                        "stop": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 2.0,
                        },
                    },
                    {
                        "type": "take_profit",
                        "name": "take_profit",
                        "take": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 4.0,
                        },
                    },
                    {
                        "type": "signal_exit",
                        "name": "ema_cross_down",
                        "condition": {
                            "cross": {
                                "a": {"ref": "ema_fast"},
                                "op": "cross_below",
                                "b": {"ref": "ema_slow"},
                            }
                        },
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }


def create_macd_strategy() -> dict:
    """MACD trend following strategy."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "MACD Trend",
            "description": "MACD crossover with trend filter",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "macd": {
                "type": "macd",
                "params": {"fast": 12, "slow": 26, "signal": 9},
            },
            "ema_200": {
                "type": "ema",
                "params": {"source": "close", "period": 200},
            },
            "atr_14": {
                "type": "atr",
                "params": {"period": 14},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "and": [
                            {
                                "cross": {
                                    "a": {"ref": "macd.macd"},
                                    "op": "cross_above",
                                    "b": {"ref": "macd.signal"},
                                }
                            },
                            {
                                "compare": {
                                    "a": {"ref": "close"},
                                    "op": ">",
                                    "b": {"ref": "ema_200"},
                                }
                            },
                        ]
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop_loss",
                        "stop": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 2.0,
                        },
                    },
                    {
                        "type": "take_profit",
                        "name": "take_profit",
                        "take": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 4.0,
                        },
                    },
                    {
                        "type": "signal_exit",
                        "name": "macd_cross_down",
                        "condition": {
                            "cross": {
                                "a": {"ref": "macd.macd"},
                                "op": "cross_below",
                                "b": {"ref": "macd.signal"},
                            }
                        },
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }


def create_breakout_strategy() -> dict:
    """Donchian channel breakout strategy."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "Donchian Breakout",
            "description": "20-period Donchian breakout with trend filter",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "donchian": {
                "type": "donchian",
                "params": {"period": 20},
            },
            "ema_100": {
                "type": "ema",
                "params": {"source": "close", "period": 100},
            },
            "atr_14": {
                "type": "atr",
                "params": {"period": 14},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "and": [
                            {
                                "compare": {
                                    "a": {"ref": "close"},
                                    "op": ">",
                                    "b": {"ref": "donchian.upper", "offset": 1},
                                }
                            },
                            {
                                "compare": {
                                    "a": {"ref": "close"},
                                    "op": ">",
                                    "b": {"ref": "ema_100"},
                                }
                            },
                        ]
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop_loss",
                        "stop": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 2.5,
                        },
                    },
                    {
                        "type": "take_profit",
                        "name": "take_profit",
                        "take": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 5.0,
                        },
                    },
                    {
                        "type": "signal_exit",
                        "name": "channel_exit",
                        "condition": {
                            "compare": {
                                "a": {"ref": "close"},
                                "op": "<",
                                "b": {"ref": "donchian.lower", "offset": 1},
                            }
                        },
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }


def create_adx_trend_strategy() -> dict:
    """ADX-based strong trend following strategy."""
    return {
        "dsl_version": "1.0.0",
        "strategy": {
            "name": "ADX Strong Trend",
            "description": "Enter on strong ADX with directional movement",
        },
        "universe": {
            "market": "crypto",
            "tickers": ["BTCUSD"],
        },
        "timeframe": "5m",
        "factors": {
            "adx": {
                "type": "adx",
                "params": {"period": 14},
            },
            "ema_50": {
                "type": "ema",
                "params": {"source": "close", "period": 50},
            },
            "atr_14": {
                "type": "atr",
                "params": {"period": 14},
            },
        },
        "trade": {
            "long": {
                "entry": {
                    "condition": {
                        "and": [
                            {
                                "compare": {
                                    "a": {"ref": "adx.adx"},
                                    "op": ">",
                                    "b": {"value": 25},
                                }
                            },
                            {
                                "compare": {
                                    "a": {"ref": "adx.plus_di"},
                                    "op": ">",
                                    "b": {"ref": "adx.minus_di"},
                                }
                            },
                            {
                                "compare": {
                                    "a": {"ref": "close"},
                                    "op": ">",
                                    "b": {"ref": "ema_50"},
                                }
                            },
                        ]
                    }
                },
                "exits": [
                    {
                        "type": "stop_loss",
                        "name": "stop_loss",
                        "stop": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 2.0,
                        },
                    },
                    {
                        "type": "take_profit",
                        "name": "take_profit",
                        "take": {
                            "kind": "atr_multiple",
                            "atr_ref": "atr_14",
                            "multiple": 4.0,
                        },
                    },
                    {
                        "type": "signal_exit",
                        "name": "trend_weakening",
                        "condition": {
                            "or": [
                                {
                                    "compare": {
                                        "a": {"ref": "adx.adx"},
                                        "op": "<",
                                        "b": {"value": 20},
                                    }
                                },
                                {
                                    "compare": {
                                        "a": {"ref": "adx.plus_di"},
                                        "op": "<",
                                        "b": {"ref": "adx.minus_di"},
                                    }
                                },
                            ]
                        },
                    },
                ],
                "position_sizing": {
                    "mode": "pct_equity",
                    "pct": 0.95,
                },
            },
        },
    }


def get_all_baseline_strategies() -> dict[str, dict]:
    """Get all baseline strategies for testing.

    Returns:
        Dictionary mapping strategy names to their DSL definitions
    """
    return {
        "ema_crossover": create_ema_crossover_strategy(),
        "macd_trend": create_macd_strategy(),
        "breakout": create_breakout_strategy(),
        "adx_trend": create_adx_trend_strategy(),
    }
