"""Optimization helper package."""

from src.engine.optimization.objectives import compute_objective_score, constraints_satisfied
from src.engine.optimization.pareto import build_metric_points, pareto_front_indices
from src.engine.optimization.search_algorithms import (
    generate_bayes_like_candidates,
    generate_grid_candidates,
    generate_random_candidates,
)
from src.engine.optimization.search_space import SearchDimension, build_search_space

__all__ = [
    "SearchDimension",
    "build_search_space",
    "generate_grid_candidates",
    "generate_random_candidates",
    "generate_bayes_like_candidates",
    "compute_objective_score",
    "constraints_satisfied",
    "pareto_front_indices",
    "build_metric_points",
]
