"""Seed-reseeding assignment bias versus the hashing target (reviewer comments 6 and 8).

Comments 6 and 8 are one vulnerability seen from two sides: if the lightweight PoW target
is set too low, a miner can cheaply discard unfavourable seeds and re-mine, biasing task
assignment. We make the target definition and the economic boundary precise.

Target definition. The gate accepts a header whose digest is below
``T_pow = 2^(256 - d)``; a uniform digest satisfies this with probability ``p = 2^-d``, so
the expected work for one seed is ``1/p = 2^d`` hashes.

Economic boundary. A miner that reseeds to land a task in a favoured fraction ``phi`` gains,
per successful reseed, a useful-work saving ``S`` (the solve-cost difference between the
assigned and the favoured instance), expressed in hash-equivalents. The expected marginal
benefit of the first reseed is ``phi * S``; its marginal cost is ``2^d`` hashes. Reseeding is
therefore economically irrational once ``2^d >= phi * S``, i.e. once

    d  >=  d* = log2(phi * S).

This experiment (a) validates that the achieved favoured rate follows
``1 - (1 - phi)^(K+1)`` for a reseed budget ``K``; (b) measures ``S`` and the host hash rate
so ``d*`` is a real number, not a symbol; and (c) sweeps ``d`` and reports the assignment
bias under an economically rational adversary, showing it collapses to zero at ``d >= d*``.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..chain import adversary as adv
from ..crypto.hashing import sha256
from ..crypto.vrf import build_vrf
from ..seeding import seed_everything
from ..stats_utils import mean_std, summarise
from ..tasks.problems.base import get_problem_class
from .base import Experiment


def _measure_hash_rate(iterations: int = 200_000) -> float:
    """Header hashes per second on the host, for converting solve time to hashes."""
    payload = b"reseeding-benchmark-header-of-about-one-hundred-bytes-padding-xxxxxxxxxxxx"
    start = time.perf_counter()
    for i in range(iterations):
        sha256(payload + i.to_bytes(8, "big"))
    return iterations / (time.perf_counter() - start)


class ReseedingBiasExperiment(Experiment):
    name = "reseeding_bias"
    description = "Assignment bias vs hashing target; safe-target boundary (comments 6, 8)"

    def run(self) -> dict[str, Any]:
        repo_size = int(self.config.repository.num_instances)
        phi = float(self.config.sweep.get("favoured_fraction", 0.25))
        budgets = [int(k) for k in self.config.sweep.get("reseed_budgets", [0, 1, 2, 5, 10])]
        difficulties = [int(d) for d in self.config.sweep.get("difficulty_bits", [4, 8, 12, 16, 20, 24, 28])]
        trials = int(self.config.sweep.get("trials", 400))
        favoured_size = int(self.config.sweep.get("favoured_instance_size", 8))
        assigned_size = int(self.config.sweep.get("assigned_instance_size", 11))

        # (a) Empirical favoured-rate versus reseed budget. The rate is a property of
        # uniform assignment, so a fast keyless VRF suffices to measure it; security still
        # requires the unique-output ECVRF elsewhere.
        vrf = build_vrf("hash_stub")
        rate_rows = []
        for k in budgets:
            per_seed = []
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                kp = vrf.keygen(bundle.bytes_for("reseed"))
                r = adv.reseeding_bias(vrf, kp.secret, repo_size, phi, k, trials, bundle.python_random("r"))
                per_seed.append(r.achieved_favoured_rate)
            m, s = mean_std(per_seed)
            rate_rows.append({
                "reseed_budget": k,
                "achieved_favoured_rate": m,
                "achieved_std": s,
                "theoretical_favoured_rate": 1.0 - (1.0 - phi) ** (k + 1),
            })
            self.logger.info("K=%d favoured_rate=%.3f (theory %.3f)", k, m, rate_rows[-1]["theoretical_favoured_rate"])

        # (b) Measure S (useful-work saving in hashes) and the host hash rate.
        problem = get_problem_class(self.config.sweep.get("problem_class", "tsp"))
        params = {"tsp": {"coordinate_range": (0.0, 100.0)}}.get(problem.name, {})
        hash_rate = _measure_hash_rate()
        favoured_times, assigned_times = [], []
        for seed in self.config.run.seeds:
            for rep in range(int(self.config.sweep.get("solve_reps", 5))):
                fav = problem.generate(f"f{seed}{rep}", favoured_size, seed * 100 + rep, **params)
                asg = problem.generate(f"a{seed}{rep}", assigned_size, seed * 200 + rep, **params)
                t = time.perf_counter(); problem.solve_exact(fav); favoured_times.append(time.perf_counter() - t)
                t = time.perf_counter(); problem.solve_exact(asg); assigned_times.append(time.perf_counter() - t)
        fav_mean = mean_std(favoured_times)[0]
        asg_mean = mean_std(assigned_times)[0]
        saving_seconds = max(0.0, asg_mean - fav_mean)
        saving_hashes = saving_seconds * hash_rate
        d_star = math.log2(phi * saving_hashes) if phi * saving_hashes > 1 else 0.0

        # (c) Bias versus target under an economically rational adversary. At difficulty d,
        # each reseed costs 2^d hashes; the adversary keeps reseeding while the expected
        # marginal benefit (1-phi)^k * phi * S exceeds the marginal cost 2^d.
        target_rows = []
        for d in difficulties:
            cost_per_reseed = 2.0 ** d
            rational_k = 0
            while rational_k < max(budgets):
                marginal_benefit = ((1.0 - phi) ** rational_k) * phi * saving_hashes
                if marginal_benefit <= cost_per_reseed:
                    break
                rational_k += 1
            favoured_rate = 1.0 - (1.0 - phi) ** (rational_k + 1)
            target_rows.append({
                "difficulty_bits": d,
                "expected_hashes_per_seed": cost_per_reseed,
                "rational_reseeds": rational_k,
                "achieved_favoured_rate": favoured_rate,
                "assignment_bias": favoured_rate - phi,
                "reseeding_profitable": rational_k > 0,
            })
            self.logger.info("d=%d hashes/seed=%.3g rational_reseeds=%d bias=%.3f",
                             d, cost_per_reseed, rational_k, favoured_rate - phi)

        return {
            "favoured_fraction": phi,
            "rate_vs_budget": rate_rows,
            "economics": {
                "hash_rate_hps": hash_rate,
                "favoured_solve_ms": fav_mean * 1000,
                "assigned_solve_ms": asg_mean * 1000,
                "saving_seconds": saving_seconds,
                "saving_hashes": saving_hashes,
                "safe_target_bits_d_star": d_star,
            },
            "bias_vs_target": target_rows,
            "csv_tables": {
                "reseeding_rate_vs_budget": rate_rows,
                "reseeding_bias_vs_target": target_rows,
            },
            "interpretation": (
                f"The achieved favoured rate follows 1-(1-phi)^(K+1). With a measured "
                f"useful-work saving of {saving_hashes:.3g} hash-equivalents and phi={phi}, "
                f"reseeding is economically irrational once the target difficulty d >= "
                f"d* = {d_star:.1f} bits; above that, assignment bias collapses to zero. "
                f"The safe target is therefore d >= log2(phi * S), a closed-form condition."
            ),
        }
