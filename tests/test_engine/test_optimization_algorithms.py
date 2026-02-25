from __future__ import annotations

from src.engine.optimization.search_algorithms import (
    generate_bayes_like_candidates,
    generate_grid_candidates,
    generate_random_candidates,
)
from src.engine.optimization.search_space import SearchDimension


DIMS = [
    SearchDimension(key="a.x", dtype="float", min_value=1.0, max_value=3.0, step=1.0),
    SearchDimension(key="b.y", dtype="float", min_value=10.0, max_value=12.0, step=1.0),
]


def test_generate_grid_candidates_budget_capped() -> None:
    rows = generate_grid_candidates(dimensions=DIMS, budget=3)
    assert len(rows) == 3


def test_generate_random_candidates_within_bounds() -> None:
    rows = generate_random_candidates(dimensions=DIMS, budget=20, seed=1)
    assert len(rows) == 20
    for row in rows:
        assert 1.0 <= row["a.x"] <= 3.0
        assert 10.0 <= row["b.y"] <= 12.0


def test_generate_bayes_like_candidates_returns_budget_size() -> None:
    rows = generate_bayes_like_candidates(dimensions=DIMS, budget=15, seed=2)
    assert len(rows) == 15
