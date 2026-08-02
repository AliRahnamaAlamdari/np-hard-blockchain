"""TSP as a :class:`ProblemClass`, reusing the existing exact solvers.

This wraps ``pouw.tasks.tsp`` so the original TSP results and the new multi-class path
share one implementation -- there is no second copy of the solver.
"""

from __future__ import annotations

from typing import Any

from .. import tsp as _tsp
from .base import ProblemClass, ProblemInstance, Solution, register


class TSPProblem(ProblemClass):
    name = "tsp"
    exact_complexity = "O((N-1)!)"  # brute force over fixed-start permutations
    verify_complexity = "O(N)"  # sum the tour length
    is_decision = False
    constraint_rejects_random = False

    def generate(self, instance_id, size, seed, coordinate_range=(0.0, 100.0), **params):
        inst = _tsp.generate_instance(
            instance_id=instance_id, size=size, seed=seed, coordinate_range=coordinate_range
        )
        return ProblemInstance(
            instance_id=instance_id,
            problem_class=self.name,
            size=size,
            data={"coordinates": [[float(x), float(y)] for x, y in inst.coordinates]},
            metadata=dict(inst.metadata),
        )

    def _to_tsp(self, instance: ProblemInstance) -> _tsp.TSPInstance:
        return _tsp.TSPInstance(
            instance_id=instance.instance_id,
            coordinates=tuple((x, y) for x, y in instance.data["coordinates"]),
            metadata=instance.metadata,
        )

    def solve_exact(self, instance):
        sol, ops = _tsp.brute_force(self._to_tsp(instance))
        return Solution(self.name, list(sol.tour), sol.cost, feasible=True), ops

    def solve_reference(self, instance):
        sol, ops = _tsp.held_karp(self._to_tsp(instance))
        return Solution(self.name, list(sol.tour), sol.cost, feasible=True), ops

    def verify(self, instance, solution, quality_threshold=None, reference_cost=None, cost_tolerance=1e-9):
        tsp_inst = self._to_tsp(instance)
        tsp_sol = _tsp.TSPSolution(tour=tuple(solution.witness), cost=solution.cost)
        return _tsp.verify_solution(
            tsp_inst, tsp_sol, quality_threshold=quality_threshold,
            reference_cost=reference_cost, cost_tolerance=cost_tolerance,
        )

    def random_witness(self, instance, rng):
        cities = list(range(instance.size))
        rng.shuffle(cities)
        cost = _tsp.tour_cost(self._to_tsp(instance), cities)
        return Solution(self.name, cities, cost, feasible=True)


register(TSPProblem())
