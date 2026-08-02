"""Useful-work task layer: TSP instances and the Merkle-committed repository."""

from .repository import (
    ProblemRepository,
    RepositorySpec,
    index_from_vrf_output,
    modulo_bias,
)
from .tsp import (
    SOLVERS,
    TSPInstance,
    TSPSolution,
    brute_force,
    count_permutations,
    generate_instance,
    held_karp,
    is_feasible,
    nearest_neighbor_2opt,
    solve,
    tour_cost,
    verify_solution,
)

__all__ = [
    "SOLVERS",
    "ProblemRepository",
    "RepositorySpec",
    "TSPInstance",
    "TSPSolution",
    "brute_force",
    "count_permutations",
    "generate_instance",
    "held_karp",
    "index_from_vrf_output",
    "is_feasible",
    "modulo_bias",
    "nearest_neighbor_2opt",
    "solve",
    "tour_cost",
    "verify_solution",
]
