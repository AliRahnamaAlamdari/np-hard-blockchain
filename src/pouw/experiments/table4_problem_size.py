"""Reproduces the paper's Table 4 (``tab:problemsize``).

Measures useful-work solve time against instance size ``N`` at fixed PoW difficulty,
averaged over ``run.repetitions`` runs per size -- the paper's stated protocol is 50
runs (L408).

Additions over the paper:

* the standard deviation the paper discards after computing its means;
* the exact tour count ``(N-1)!`` and the measured per-tour cost, which is what makes
  the ``(N-1)!`` scaling claim checkable rather than asserted;
* Held-Karp timings for the same instances.  Held-Karp returns the *same* optimum in
  O(2^N N^2), so the "strict O(N!) growth" of the paper is a property of the chosen
  solver, not of the problem.  The speedup column is the mining advantage available
  to any miner who implements a better exact solver.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..crypto.hashing import difficulty_to_target
from ..crypto.vrf import build_vrf
from ..chain.miner import Miner
from ..reporting import figures
from ..seeding import seed_everything
from ..tasks.tsp import brute_force, count_permutations, held_karp
from .base import Experiment, mean_std


class ProblemSizeExperiment(Experiment):
    name = "table4_problem_size"
    description = "Useful-work solve time and computation share against TSP size N"

    def run(self) -> dict[str, Any]:
        sizes = self.config.sweep.get("instance_sizes", [4, 6, 8, 10])
        repetitions = int(self.config.run.repetitions)
        compare_held_karp = bool(self.config.sweep.get("compare_held_karp", True))
        difficulty_bits = self.config.pow.difficulty_bits
        target = difficulty_to_target(difficulty_bits)

        rows: list[dict[str, Any]] = []
        per_run_rows: list[dict[str, Any]] = []

        for size in sizes:
            size = int(size)
            self.logger.info(
                "N=%d: %d repetitions x %d seeds ((N-1)! = %d tours)",
                size,
                repetitions,
                len(self.config.run.seeds),
                count_permutations(size),
            )
            pow_times: list[float] = []
            solve_times: list[float] = []
            held_karp_times: list[float] = []
            operations: list[int] = []
            costs: list[float] = []
            optimal_agreement = True

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle, instance_size=size)
                vrf = build_vrf(self.config.vrf.backend, **self.config.vrf.build_kwargs())
                miner = Miner(
                    miner_id=0,
                    keys=vrf.keygen(bundle.bytes_for("miner:0")),
                    vrf=vrf,
                )

                for repetition in range(repetitions):
                    pow_result = miner.search_pow(
                        height=repetition + 1,
                        previous_hash=f"parent-{seed}-{repetition}",
                        tx_root=f"tx-{seed}-{repetition}",
                        target=target,
                        max_attempts=self.config.pow.max_attempts,
                    )
                    beta, _, index = miner.assign_task(pow_result.digest, repository)
                    instance = repository.instance(index)

                    start = time.perf_counter()
                    solution, ops = brute_force(instance)
                    solve_seconds = time.perf_counter() - start

                    pow_times.append(pow_result.seconds)
                    solve_times.append(solve_seconds)
                    operations.append(ops)
                    costs.append(solution.cost)

                    if compare_held_karp:
                        start = time.perf_counter()
                        hk_solution, _ = held_karp(instance)
                        held_karp_times.append(time.perf_counter() - start)
                        if abs(hk_solution.cost - solution.cost) > 1e-9:
                            optimal_agreement = False

                    per_run_rows.append(
                        {
                            "size": size,
                            "seed": seed,
                            "repetition": repetition,
                            "instance_id": instance.instance_id,
                            "pow_seconds": pow_result.seconds,
                            "pow_attempts": pow_result.attempts,
                            "solve_seconds": solve_seconds,
                            "solver_operations": ops,
                            "tour_cost": solution.cost,
                        }
                    )

            pow_mean, pow_std = mean_std(pow_times)
            solve_mean, solve_std = mean_std(solve_times)
            total = pow_mean + solve_mean
            tours = count_permutations(size)
            row: dict[str, Any] = {
                "instance_id": f"I-{size}-A",
                "size": size,
                "num_runs": len(solve_times),
                "mean_pow_seconds": pow_mean,
                "std_pow_seconds": pow_std,
                "mean_solve_seconds": solve_mean,
                "std_solve_seconds": solve_std,
                "useful_share_percent": solve_mean / total * 100.0 if total > 0 else 0.0,
                "tours_enumerated": tours,
                "mean_solver_operations": sum(operations) / len(operations),
                "microseconds_per_tour": solve_mean / tours * 1e6,
                "mean_tour_cost": sum(costs) / len(costs),
            }
            if compare_held_karp and held_karp_times:
                hk_mean, hk_std = mean_std(held_karp_times)
                row.update(
                    {
                        "mean_heldkarp_seconds": hk_mean,
                        "std_heldkarp_seconds": hk_std,
                        "heldkarp_speedup": solve_mean / hk_mean if hk_mean > 0 else float("inf"),
                        "heldkarp_matches_optimum": optimal_agreement,
                        "heldkarp_operations_bound": (2**size) * size * size,
                    }
                )
            rows.append(row)
            self.logger.info(
                "N=%d -> solve %.6g s +/- %.2g, share %.2f%%, %.3f us/tour%s",
                size,
                solve_mean,
                solve_std,
                row["useful_share_percent"],
                row["microseconds_per_tour"],
                f", Held-Karp {row['heldkarp_speedup']:.1f}x faster" if "heldkarp_speedup" in row else "",
            )

        growth = self._scaling_check(rows)
        figure_paths = figures.problem_size_figure(
            rows, self.output_dir / "table4_problem_size"
        )
        return {
            "table_rows": rows,
            "per_run_rows": per_run_rows,
            "scaling_check": growth,
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {
                "table4_problem_size": rows,
                "table4_per_run": per_run_rows,
            },
            "paper_reference": {
                "table": "tab:problemsize",
                "reported": [
                    {"instance_id": "I-4-A", "size": 4, "pow_s": 0.00143, "useful_s": 0.00002, "share_pct": 1.59},
                    {"instance_id": "I-6-A", "size": 6, "pow_s": 0.00162, "useful_s": 0.00022, "share_pct": 12.09},
                    {"instance_id": "I-8-A", "size": 8, "pow_s": 0.00406, "useful_s": 0.00871, "share_pct": 68.20},
                    {"instance_id": "I-10-A", "size": 10, "pow_s": 0.00129, "useful_s": 0.75249, "share_pct": 99.83},
                    {"instance_id": "I-12-A", "size": 12, "pow_s": 0.00106, "useful_s": 99.71407, "share_pct": 100.00},
                ],
            },
        }

    def _scaling_check(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compare measured time ratios against the ``(N-1)!`` tour-count ratios."""
        checks = []
        for previous, current in zip(rows, rows[1:]):
            tour_ratio = math.factorial(current["size"] - 1) / math.factorial(previous["size"] - 1)
            time_ratio = (
                current["mean_solve_seconds"] / previous["mean_solve_seconds"]
                if previous["mean_solve_seconds"] > 0
                else float("inf")
            )
            checks.append(
                {
                    "from_size": previous["size"],
                    "to_size": current["size"],
                    "tour_count_ratio": tour_ratio,
                    "measured_time_ratio": time_ratio,
                    "relative_deviation": (time_ratio - tour_ratio) / tour_ratio,
                }
            )
        return checks
