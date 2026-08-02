"""Graph k-colouring as a :class:`ProblemClass`.

Find the chromatic number of a random graph and a proper colouring achieving it. The
miner's exact solver searches k = 1, 2, ... and, for each k, backtracks over vertex
colourings; the worst case is ``O(k^N)``. The reference solver is a second backtracking
search with a most-constrained-vertex heuristic, which reaches the same chromatic number
with far less work -- again exhibiting the solver-choice advantage.

The objective (cost) is the number of colours used; verification checks that the
colouring is proper (no monochromatic edge) and uses exactly the claimed number of
colours, in ``O(V + E)``.
"""

from __future__ import annotations

import random
from typing import Any

from .base import ProblemClass, ProblemInstance, Solution, register


class GraphColoringProblem(ProblemClass):
    name = "graph_coloring"
    exact_complexity = "O(k^N)"  # backtracking over colourings
    verify_complexity = "O(V + E)"
    is_decision = False
    constraint_rejects_random = True

    def generate(self, instance_id, size, seed, edge_probability=0.5, **params):
        rng = random.Random(seed)
        edges = []
        for u in range(size):
            for v in range(u + 1, size):
                if rng.random() < edge_probability:
                    edges.append([u, v])
        return ProblemInstance(
            instance_id=instance_id,
            problem_class=self.name,
            size=size,
            data={"num_vertices": size, "edges": edges},
            metadata={"seed": seed, "edge_probability": edge_probability},
        )

    @staticmethod
    def _adjacency(instance: ProblemInstance):
        n = instance.size
        adj = [set() for _ in range(n)]
        for u, v in instance.data["edges"]:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _color_with_k(self, adj, n, k, order=None):
        """Backtracking: return a proper k-colouring or None. Also returns op count."""
        colors = [-1] * n
        ops = [0]
        seq = order if order is not None else list(range(n))

        def backtrack(idx: int) -> bool:
            if idx == n:
                return True
            v = seq[idx]
            used = {colors[u] for u in adj[v] if colors[u] != -1}
            for c in range(k):
                ops[0] += 1
                if c not in used:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False

        return (colors[:] if backtrack(0) else None), ops[0]

    def _chromatic(self, instance, order=None):
        adj = self._adjacency(instance)
        n = instance.size
        total_ops = 0
        for k in range(1, n + 1):
            coloring, ops = self._color_with_k(adj, n, k, order=order)
            total_ops += ops
            if coloring is not None:
                return coloring, k, total_ops
        return list(range(n)), n, total_ops  # fallback: all-different (always proper)

    def solve_exact(self, instance):
        coloring, k, ops = self._chromatic(instance, order=None)
        return Solution(self.name, coloring, float(k), feasible=True), ops

    def solve_reference(self, instance):
        # Most-constrained-first ordering: sort vertices by descending degree.
        adj = self._adjacency(instance)
        order = sorted(range(instance.size), key=lambda v: len(adj[v]), reverse=True)
        coloring, k, ops = self._chromatic(instance, order=order)
        return Solution(self.name, coloring, float(k), feasible=True), ops

    def verify(self, instance, solution, quality_threshold=None, reference_cost=None, cost_tolerance=1e-9):
        coloring = solution.witness
        n = instance.size
        if not isinstance(coloring, list) or len(coloring) != n:
            return False, "malformed_witness"
        for u, v in instance.data["edges"]:
            if coloring[u] == coloring[v]:
                return False, "improper_coloring"
        colors_used = len(set(coloring))
        if abs(float(colors_used) - solution.cost) > cost_tolerance:
            return False, "color_count_mismatch"
        if quality_threshold is not None and reference_cost is not None:
            # allow up to reference chromatic number times (1 + threshold)
            if colors_used > reference_cost * (1.0 + quality_threshold) + cost_tolerance:
                return False, "quality_threshold_exceeded"
        return True, "ok"

    def random_witness(self, instance, rng):
        # Random colouring with the number of colours the optimum would use; usually
        # improper, which is exactly what the adversarial-submission test needs.
        _, k, _ = self._chromatic(instance)
        coloring = [rng.randrange(max(1, k)) for _ in range(instance.size)]
        return Solution(self.name, coloring, float(len(set(coloring))), feasible=True)


register(GraphColoringProblem())
