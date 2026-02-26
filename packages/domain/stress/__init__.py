"""Stress job orchestration public exports."""

_SERVICE_EXPORTS: frozenset[str] = frozenset(
    {
        "StressInputError",
        "StressJobNotFoundError",
        "StressJobReceipt",
        "StressJobView",
        "StressStrategyNotFoundError",
        "create_stress_job",
        "execute_stress_job",
        "execute_stress_job_with_fresh_session",
        "get_optimization_pareto_points",
        "get_stress_job_view",
        "list_black_swan_windows",
        "schedule_stress_job",
    }
)


def __getattr__(name: str):
    if name in _SERVICE_EXPORTS:
        from packages.domain.stress import service as _service

        return getattr(_service, name)
    raise AttributeError(name)

__all__ = [
    "StressInputError",
    "StressJobNotFoundError",
    "StressJobReceipt",
    "StressJobView",
    "StressStrategyNotFoundError",
    "create_stress_job",
    "execute_stress_job",
    "execute_stress_job_with_fresh_session",
    "get_stress_job_view",
    "get_optimization_pareto_points",
    "list_black_swan_windows",
    "schedule_stress_job",
]
