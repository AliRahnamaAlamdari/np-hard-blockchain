"""Symmetric Euclidean TSP: instance generation, solvers, and verification.

Instance format follows the paper (L408): JSON, ``N`` nodes, floating-point ``x`` and
``y`` on a 2D Euclidean plane.  Coordinate range and distribution are not stated
(``PAPER_SPEC.md`` §11-U2); they are config values here, defaulting to uniform on
[0, 100]^2.

Three solvers are provided.  ``brute_force`` is the paper's solver, enumerating all
``(N-1)!`` permutations with the start node fixed.  ``held_karp`` is the standard
O(2^N N^2) exact dynamic program, included because it returns the *same* optimal tour
far faster -- which is what makes the paper's "strict O(N!) growth" a property of the
chosen solver rather than of the problem, and which lets the solver-advantage
experiment quantify the mining edge a better-equipped miner would gain.
``nearest_neighbor``, ``nearest_neighbor_2opt`` and ``or_opt`` are a ladder of fast
heuristics of increasing quality, used for the quality-threshold path the paper
describes at L387 but never exercises.  A ladder is what the threshold needs: with a
single heuristic the miner and the repository publisher compute the same tour and the
bound can never bind, so the check would be present but vacuous.  With solvers of
different strength the same bound accepts some and rejects others, which is the
behaviour the acceptance rule is supposed to have.

All solvers here are deterministic: no solver draws from a random stream, so a run
seed still fixes every reported number bit-for-bit.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TSPInstance:
    """A symmetric Euclidean TSP instance."""

    instance_id: str
    coordinates: tuple[tuple[float, float], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.coordinates)

    def to_payload(self) -> dict[str, Any]:
        """Canonical dict form -- this is what the Merkle leaf commits to."""
        return {
            "instance_id": self.instance_id,
            "size": self.size,
            "coordinates": [[float(x), float(y)] for x, y in self.coordinates],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TSPInstance":
        return cls(
            instance_id=payload["instance_id"],
            coordinates=tuple((float(x), float(y)) for x, y in payload["coordinates"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class TSPSolution:
    """A tour and its cost.  ``tour`` is a permutation of ``range(N)``."""

    tour: tuple[int, ...]
    cost: float

    def to_payload(self) -> dict[str, Any]:
        return {"tour": list(self.tour), "cost": float(self.cost)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TSPSolution":
        return cls(tour=tuple(int(i) for i in payload["tour"]), cost=float(payload["cost"]))


def generate_instance(
    instance_id: str,
    size: int,
    seed: int,
    coordinate_range: tuple[float, float] = (0.0, 100.0),
    difficulty_class: str | None = None,
) -> TSPInstance:
    """Generate one instance deterministically from ``seed``."""
    if size < 3:
        raise ValueError(f"TSP size must be at least 3, got {size}")
    rng = random.Random(seed)
    low, high = coordinate_range
    coordinates = tuple(
        (rng.uniform(low, high), rng.uniform(low, high)) for _ in range(size)
    )
    metadata: dict[str, Any] = {"seed": seed, "generator": "uniform2d"}
    if difficulty_class is not None:
        metadata["difficulty_class"] = difficulty_class
    return TSPInstance(instance_id=instance_id, coordinates=coordinates, metadata=metadata)


def distance_matrix(instance: TSPInstance) -> list[list[float]]:
    coords = instance.coordinates
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def tour_cost(instance: TSPInstance, tour: Sequence[int]) -> float:
    """Total Euclidean length of the closed tour, computed directly from coordinates."""
    coords = instance.coordinates
    total = 0.0
    for position in range(len(tour)):
        x1, y1 = coords[tour[position]]
        x2, y2 = coords[tour[(position + 1) % len(tour)]]
        total += math.hypot(x1 - x2, y1 - y2)
    return total


def is_feasible(instance: TSPInstance, tour: Sequence[int]) -> bool:
    """A tour is feasible iff it visits every city exactly once."""
    n = instance.size
    if len(tour) != n:
        return False
    seen = [False] * n
    for city in tour:
        if not isinstance(city, int) or not 0 <= city < n or seen[city]:
            return False
        seen[city] = True
    return True


def count_permutations(size: int) -> int:
    """``(N-1)!`` -- the number of tours the paper's solver enumerates."""
    return math.factorial(size - 1)


def brute_force(instance: TSPInstance) -> tuple[TSPSolution, int]:
    """The paper's solver: enumerate all ``(N-1)!`` tours with city 0 fixed.

    Returns the optimal tour and the number of tours evaluated.  The symmetric
    factor-2 redundancy (every tour and its reverse are both enumerated) is *not*
    exploited, matching L408 exactly.
    """
    n = instance.size
    matrix = distance_matrix(instance)
    best_tour: tuple[int, ...] | None = None
    best_cost = math.inf
    evaluated = 0
    for permutation in itertools.permutations(range(1, n)):
        cost = matrix[0][permutation[0]]
        previous = permutation[0]
        for city in permutation[1:]:
            cost += matrix[previous][city]
            previous = city
        cost += matrix[previous][0]
        evaluated += 1
        if cost < best_cost:
            best_cost = cost
            best_tour = (0,) + permutation
    assert best_tour is not None
    return TSPSolution(tour=best_tour, cost=best_cost), evaluated


def held_karp(instance: TSPInstance) -> tuple[TSPSolution, int]:
    """Exact O(2^N * N^2) dynamic program.  Same optimum, far fewer operations."""
    n = instance.size
    matrix = distance_matrix(instance)
    full = 1 << (n - 1)
    # cost[mask][j] = cheapest path 0 -> ... -> (j+1) visiting exactly `mask`.
    cost = [[math.inf] * (n - 1) for _ in range(full)]
    parent = [[-1] * (n - 1) for _ in range(full)]
    operations = 0
    for j in range(n - 1):
        cost[1 << j][j] = matrix[0][j + 1]
    for mask in range(full):
        for j in range(n - 1):
            if not mask & (1 << j) or cost[mask][j] == math.inf:
                continue
            base = cost[mask][j]
            for k in range(n - 1):
                if mask & (1 << k):
                    continue
                operations += 1
                candidate = base + matrix[j + 1][k + 1]
                next_mask = mask | (1 << k)
                if candidate < cost[next_mask][k]:
                    cost[next_mask][k] = candidate
                    parent[next_mask][k] = j
    best_cost = math.inf
    best_last = -1
    for j in range(n - 1):
        operations += 1
        candidate = cost[full - 1][j] + matrix[j + 1][0]
        if candidate < best_cost:
            best_cost = candidate
            best_last = j
    tour: list[int] = []
    mask = full - 1
    last = best_last
    while last != -1:
        tour.append(last + 1)
        previous = parent[mask][last]
        mask ^= 1 << last
        last = previous
    tour.append(0)
    tour.reverse()
    return TSPSolution(tour=tuple(tour), cost=best_cost), operations


def _nearest_neighbor_tour(
    matrix: list[list[float]], n: int
) -> tuple[list[int], int]:
    """Greedy construction from city 0, always hopping to the nearest unvisited."""
    operations = 0
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        operations += len(unvisited)
        nearest = min(unvisited, key=lambda city: matrix[current][city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return tour, operations


def nearest_neighbor(instance: TSPInstance) -> tuple[TSPSolution, int]:
    """Construction only, no improvement pass -- the weakest rung of the ladder.

    Included so the quality threshold has something to reject: a nearest-neighbour
    tour is typically well above the 2-opt and Or-opt tours on the same instance, so
    a bound published against a stronger heuristic binds against this one.
    """
    n = instance.size
    matrix = distance_matrix(instance)
    tour, operations = _nearest_neighbor_tour(matrix, n)
    solution_tour = tuple(tour)
    return (
        TSPSolution(tour=solution_tour, cost=tour_cost(instance, solution_tour)),
        operations,
    )


def or_opt(instance: TSPInstance, max_passes: int = 50) -> tuple[TSPSolution, int]:
    """2-opt followed by Or-opt segment relocation, at a fixed iteration budget.

    Or-opt lifts a run of one to three consecutive cities out of the tour and
    reinserts it elsewhere, which 2-opt alone cannot do because 2-opt only reverses
    contiguous spans.  Running both to a fixed pass budget makes this the strongest
    heuristic in the ladder while keeping the cost polynomial and the result
    deterministic.
    """
    n = instance.size
    matrix = distance_matrix(instance)
    base, operations = nearest_neighbor_2opt(instance, max_passes=max_passes)
    tour = list(base.tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for length in (1, 2, 3):
            if n <= length + 2:
                continue
            position = 1
            while position + length <= n:
                segment = tour[position : position + length]
                remainder = tour[:position] + tour[position + length :]
                previous = remainder[position - 1]
                following = remainder[position % len(remainder)]
                removed = (
                    matrix[previous][segment[0]]
                    + matrix[segment[-1]][following]
                    - matrix[previous][following]
                )
                best_gain = 1e-12
                best_at = -1
                best_reversed = False
                for insert in range(1, len(remainder)):
                    operations += 1
                    left = remainder[insert - 1]
                    right = remainder[insert % len(remainder)]
                    if left in segment or right in segment:
                        continue
                    forward = (
                        matrix[left][segment[0]]
                        + matrix[segment[-1]][right]
                        - matrix[left][right]
                    )
                    backward = (
                        matrix[left][segment[-1]]
                        + matrix[segment[0]][right]
                        - matrix[left][right]
                    )
                    for added, is_reversed in ((forward, False), (backward, True)):
                        if removed - added > best_gain:
                            best_gain = removed - added
                            best_at = insert
                            best_reversed = is_reversed
                if best_at >= 0:
                    moved = list(reversed(segment)) if best_reversed else segment
                    tour = remainder[:best_at] + moved + remainder[best_at:]
                    improved = True
                else:
                    position += 1
    # City 0 stays the anchor so the tour is comparable with every other solver.
    origin = tour.index(0)
    tour = tour[origin:] + tour[:origin]
    solution_tour = tuple(tour)
    return (
        TSPSolution(tour=solution_tour, cost=tour_cost(instance, solution_tour)),
        operations,
    )


def nearest_neighbor_2opt(
    instance: TSPInstance, max_passes: int = 50
) -> tuple[TSPSolution, int]:
    """Heuristic: nearest-neighbour construction refined by 2-opt.  Not exact."""
    n = instance.size
    matrix = distance_matrix(instance)
    tour, operations = _nearest_neighbor_tour(matrix, n)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                operations += 1
                a, b = tour[i - 1], tour[i]
                c = tour[j]
                d = tour[(j + 1) % n]
                if matrix[a][b] + matrix[c][d] > matrix[a][c] + matrix[b][d] + 1e-12:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
    solution_tour = tuple(tour)
    return TSPSolution(tour=solution_tour, cost=tour_cost(instance, solution_tour)), operations


SOLVERS = {
    "brute_force": brute_force,
    "held_karp": held_karp,
    "nearest_neighbor": nearest_neighbor,
    "nearest_neighbor_2opt": nearest_neighbor_2opt,
    "or_opt": or_opt,
}

# Solvers that return the proven global optimum, as opposed to a feasible tour.
EXACT_SOLVERS = frozenset({"brute_force", "held_karp"})


def solve(instance: TSPInstance, solver: str = "brute_force") -> tuple[TSPSolution, int]:
    """Dispatch to a named solver, returning ``(solution, operation_count)``."""
    try:
        return SOLVERS[solver](instance)
    except KeyError:
        raise ValueError(
            f"unknown solver {solver!r}; expected one of {sorted(SOLVERS)}"
        ) from None


def verify_solution(
    instance: TSPInstance,
    solution: TSPSolution,
    quality_threshold: float | None = None,
    reference_cost: float | None = None,
    cost_tolerance: float = 1e-9,
) -> tuple[bool, str]:
    """Validator-side check ``V(I_i, w)``.

    Three conditions, in the order a validator would apply them:

    1. the tour is feasible (a permutation of all cities);
    2. the claimed cost ``c`` matches the recomputed cost, so a miner cannot understate
       its objective value;
    3. if a ``quality_threshold`` is configured, the cost is within
       ``(1 + threshold)`` of ``reference_cost``.

    Condition 3 is the "protocol-defined quality bound" of paper L387, which the paper
    describes but never instantiates (``PAPER_SPEC.md`` §11-U7).  With an exact solver
    it never binds, so it is off by default.
    """
    if not is_feasible(instance, solution.tour):
        return False, "infeasible_tour"
    recomputed = tour_cost(instance, solution.tour)
    if abs(recomputed - solution.cost) > cost_tolerance * max(1.0, abs(recomputed)):
        return False, "cost_mismatch"
    if quality_threshold is not None:
        if reference_cost is None:
            return False, "missing_reference_cost"
        if recomputed > reference_cost * (1.0 + quality_threshold) + cost_tolerance:
            return False, "quality_threshold_exceeded"
    return True, "ok"


def iter_sizes(sizes: Iterable[int]) -> list[int]:
    return [int(size) for size in sizes]
