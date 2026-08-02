"""0/1 Knapsack as a :class:`ProblemClass`.

Maximize total value subject to a weight budget. The miner's exact solver is brute
force over all ``2^N`` subsets (mirroring the paper's choice of an exponential
exhaustive solver for TSP); the reference solver is the standard ``O(N*C)`` dynamic
program, which returns the identical optimum far faster and so quantifies the same
solver-choice advantage that TSP exhibits.

Verification is ``O(N)``: sum the chosen weights and values and compare the claimed
value. The objective is reported as a cost to be *minimized* (negated value) so the
framework's "lower cost is better" convention is uniform across classes.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from .base import ProblemClass, ProblemInstance, Solution, register


class KnapsackProblem(ProblemClass):
    name = "knapsack"
    exact_complexity = "O(2^N)"  # brute force over subsets
    verify_complexity = "O(N)"
    is_decision = False
    constraint_rejects_random = False

    def generate(self, instance_id, size, seed, max_weight=100, max_value=100, capacity_ratio=0.5, **params):
        rng = random.Random(seed)
        weights = [rng.randint(1, max_weight) for _ in range(size)]
        values = [rng.randint(1, max_value) for _ in range(size)]
        capacity = int(sum(weights) * capacity_ratio)
        return ProblemInstance(
            instance_id=instance_id,
            problem_class=self.name,
            size=size,
            data={"weights": weights, "values": values, "capacity": capacity},
            metadata={"seed": seed, "generator": "uniform_int"},
        )

    @staticmethod
    def _fields(instance: ProblemInstance):
        d = instance.data
        return d["weights"], d["values"], int(d["capacity"])

    def solve_exact(self, instance):
        weights, values, capacity = self._fields(instance)
        n = instance.size
        best_value = -1
        best_set: list[int] = []
        ops = 0
        for mask in range(1 << n):
            ops += 1
            w = v = 0
            chosen = []
            for i in range(n):
                if mask & (1 << i):
                    w += weights[i]
                    v += values[i]
                    chosen.append(i)
            if w <= capacity and v > best_value:
                best_value = v
                best_set = chosen
        return Solution(self.name, best_set, float(-best_value), feasible=True), ops

    def solve_reference(self, instance):
        weights, values, capacity = self._fields(instance)
        n = instance.size
        dp = [0] * (capacity + 1)
        keep = [[False] * (capacity + 1) for _ in range(n)]
        ops = 0
        for i in range(n):
            wi, vi = weights[i], values[i]
            for c in range(capacity, wi - 1, -1):
                ops += 1
                cand = dp[c - wi] + vi
                if cand > dp[c]:
                    dp[c] = cand
                    keep[i][c] = True
        # reconstruct
        chosen: list[int] = []
        c = capacity
        for i in range(n - 1, -1, -1):
            if keep[i][c]:
                chosen.append(i)
                c -= weights[i]
        chosen.sort()
        best_value = sum(values[i] for i in chosen)
        return Solution(self.name, chosen, float(-best_value), feasible=True), ops

    def verify(self, instance, solution, quality_threshold=None, reference_cost=None, cost_tolerance=1e-9):
        weights, values, capacity = self._fields(instance)
        chosen = solution.witness
        if not isinstance(chosen, list) or len(set(chosen)) != len(chosen):
            return False, "malformed_witness"
        if any((not isinstance(i, int)) or i < 0 or i >= instance.size for i in chosen):
            return False, "index_out_of_range"
        total_w = sum(weights[i] for i in chosen)
        if total_w > capacity:
            return False, "capacity_exceeded"
        total_v = sum(values[i] for i in chosen)
        if abs(float(-total_v) - solution.cost) > cost_tolerance * max(1.0, total_v):
            return False, "value_mismatch"
        if quality_threshold is not None:
            if reference_cost is None:
                return False, "missing_reference_cost"
            # cost is negated value; a threshold means value within (1-thr) of optimum
            best_value = -reference_cost
            if total_v < best_value * (1.0 - quality_threshold) - cost_tolerance:
                return False, "quality_threshold_not_met"
        return True, "ok"

    def random_witness(self, instance, rng):
        weights, values, capacity = self._fields(instance)
        chosen = [i for i in range(instance.size) if rng.random() < 0.5]
        total_v = sum(values[i] for i in chosen)
        return Solution(self.name, chosen, float(-total_v), feasible=True)


register(KnapsackProblem())
