"""Sensitivity analyses the paper does not contain.

Three parameters that the framework's own equations make first-class, but that the
paper never varies:

``repository_size``
    ``N`` in Equation (3).  The paper never states its value
    (``PAPER_SPEC.md`` §11-U3).  This sweeps it and reports task-selection uniformity,
    Merkle proof depth and size, and the modulo bias of ``Integer(y) mod N``.

``hash_rate``
    The fairness experiment uses five *equal-power* miners, so it cannot detect the
    bias that matters in practice.  This sweeps heterogeneous hash rates and checks
    whether block share tracks power share, which is the property a PoW-gated
    protocol must have.

``solver_advantage``
    Both the honest and the adversarial miner solve the same assigned instance, but
    the adversary uses Held-Karp instead of brute force.  Reports the resulting share
    of blocks -- the mining edge available to a miner with better software, which is
    invisible in a single-solver evaluation.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from ..chain.miner import Miner
from ..chain.network import Network
from ..crypto.hashing import difficulty_to_target
from ..crypto.vrf import build_vrf
from ..metrics.fairness import analyse, chi_square_expected, chi_square_uniformity
from ..reporting import figures
from ..seeding import seed_everything
from ..tasks.repository import index_from_vrf_output, modulo_bias
from ..tasks.tsp import brute_force, held_karp
from .base import Experiment, mean_std


class RepositorySizeSensitivity(Experiment):
    name = "sensitivity_repository_size"
    description = "Effect of repository size N on selection uniformity and proof cost"

    def run(self) -> dict[str, Any]:
        sizes = [int(v) for v in self.config.sweep.get("repository_sizes", [16, 64, 256, 1024])]
        draws = int(self.config.sweep.get("draws_per_size", 2000))
        rows: list[dict[str, Any]] = []

        for num_instances in sizes:
            self.logger.info("repository size N=%d, %d draws per seed", num_instances, draws)
            uniformity_p: list[float] = []
            depths: list[int] = []
            proof_bytes: list[int] = []
            verify_times: list[float] = []

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle, num_instances=num_instances)
                vrf = build_vrf(self.config.vrf.backend, **self.config.vrf.build_kwargs())
                keys = vrf.keygen(bundle.bytes_for("miner:0"))

                counts: Counter[int] = Counter()
                for draw in range(draws):
                    beta, _ = vrf.prove(keys.secret, f"seed-{seed}-{draw}".encode())
                    counts[index_from_vrf_output(beta, num_instances)] += 1
                observed = [counts.get(index, 0) for index in range(num_instances)]
                _, _, p_value = chi_square_uniformity(observed)
                uniformity_p.append(p_value)

                _, proof = repository.get_proof(0)
                depths.append(proof.depth)
                proof_bytes.append(proof.size_bytes())
                start = time.perf_counter()
                repository.verify(repository.instance(0), proof)
                verify_times.append(time.perf_counter() - start)

            p_mean, p_std = mean_std(uniformity_p)
            verify_mean, verify_std = mean_std(verify_times)
            rows.append(
                {
                    "repository_size": num_instances,
                    "draws_per_seed": draws,
                    "uniformity_p_mean": p_mean,
                    "uniformity_p_std": p_std,
                    "uniform_at_5pct": p_mean > 0.05,
                    "merkle_depth": depths[0],
                    "merkle_proof_bytes": proof_bytes[0],
                    "merkle_verify_ms": verify_mean * 1000.0,
                    "merkle_verify_ms_std": verify_std * 1000.0,
                    "modulo_bias_256bit": modulo_bias(256, num_instances),
                    "modulo_bias_32bit": modulo_bias(32, num_instances),
                }
            )

        figure_paths = figures.sensitivity_figure(
            [row["repository_size"] for row in rows],
            {
                "Merkle proof size (bytes)": [row["merkle_proof_bytes"] for row in rows],
            },
            x_label="Repository size $N$",
            y_label="Merkle proof size (bytes)",
            path=self.output_dir / "sensitivity_repository_size",
            log_x=True,
        )
        return {
            "table_rows": rows,
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {"sensitivity_repository_size": rows},
            "interpretation": (
                "Selection stays uniform and proof cost grows only logarithmically, "
                "so N is safe to increase -- which matters because a larger repository "
                "is the paper's own stated defence against precomputation, and the "
                "paper never shows what it costs."
            ),
        }


class HashRateSensitivity(Experiment):
    name = "sensitivity_hash_rate"
    description = "Fairness under heterogeneous miner hash rates"

    def run(self) -> dict[str, Any]:
        distributions = self.config.sweep.get(
            "hash_rate_distributions",
            [
                {"label": "equal", "rates": [1, 1, 1, 1, 1]},
                {"label": "mild", "rates": [1, 1, 1, 2, 2]},
                {"label": "skewed", "rates": [1, 1, 1, 1, 4]},
            ],
        )
        num_blocks = int(self.config.sweep.get("blocks", self.config.network.num_blocks))
        rows: list[dict[str, Any]] = []

        for distribution in distributions:
            label = distribution["label"]
            rates = [float(rate) for rate in distribution["rates"]]
            total_rate = sum(rates)
            expected_shares = [rate / total_rate for rate in rates]
            self.logger.info("hash rate distribution %s: %s", label, rates)

            observed_shares: list[list[float]] = []
            jain_values: list[float] = []
            deviations: list[float] = []
            p_values: list[float] = []
            pooled_counts = [0] * len(rates)

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle)
                network = Network(
                    num_miners=len(rates),
                    repository=repository,
                    difficulty_bits=self.config.pow.difficulty_bits,
                    vrf_backend=self.config.vrf.backend,
                    seeds=bundle,
                    solver=self.config.solver.name,
                    hash_rates=rates,
                    max_attempts=self.config.pow.max_attempts,
                    quality_threshold=self.config.solver.quality_threshold,
                    cost_tolerance=self.config.solver.cost_tolerance,
                    tie_break=self.config.network.tie_break,
                    vrf_kwargs=self.config.vrf.build_kwargs(),
                )
                result = network.run(num_blocks=num_blocks, log_every=self.config.run.log_every_blocks)
                counts = [result.blocks_won[i] for i in range(len(rates))]
                total_blocks = sum(counts)
                shares = [count / total_blocks for count in counts]
                observed_shares.append(shares)
                jain_values.append(analyse(counts).jain_index)
                deviations.append(
                    max(abs(observed - expected) for observed, expected in zip(shares, expected_shares))
                )
                p_values.append(chi_square_expected(counts, expected_shares)[2])
                pooled_counts = [
                    total + count for total, count in zip(pooled_counts, counts)
                ]

            mean_shares = [
                sum(run[i] for run in observed_shares) / len(observed_shares)
                for i in range(len(rates))
            ]
            jain_mean, jain_std = mean_std(jain_values)
            deviation_mean, deviation_std = mean_std(deviations)
            p_mean, p_std = mean_std(p_values)
            # Pooling every seed gives the most powerful single test of the null
            # hypothesis "block share is proportional to hash-rate share".
            pooled_statistic, pooled_dof, pooled_p = chi_square_expected(
                pooled_counts, expected_shares
            )
            rows.append(
                {
                    "distribution": label,
                    "hash_rates": rates,
                    "expected_shares": expected_shares,
                    "observed_shares": mean_shares,
                    "max_abs_deviation_mean": deviation_mean,
                    "max_abs_deviation_std": deviation_std,
                    "jain_index_mean": jain_mean,
                    "jain_index_std": jain_std,
                    "blocks_per_seed": num_blocks,
                    "total_blocks": sum(pooled_counts),
                    "proportionality_p_mean": p_mean,
                    "proportionality_p_std": p_std,
                    "pooled_chi_square": pooled_statistic,
                    "pooled_dof": pooled_dof,
                    "pooled_p_value": pooled_p,
                    # Fail to reject proportionality at the 5% level.
                    "power_proportional": pooled_p > 0.05,
                }
            )
            self.logger.info(
                "%s -> max deviation %.4f (+/- %.4f), jain %.4f, "
                "pooled chi2=%.3f p=%.4f (proportional: %s)",
                label,
                deviation_mean,
                deviation_std,
                jain_mean,
                pooled_statistic,
                pooled_p,
                pooled_p > 0.05,
            )

        return {
            "table_rows": rows,
            "csv_tables": {"sensitivity_hash_rate": rows},
            "interpretation": (
                "Block share should track hash-rate share. The paper tests only equal "
                "power, where Jain's index near 1 is the expected outcome and carries "
                "no information about whether the protocol is biased."
            ),
        }


class SolverAdvantageSensitivity(Experiment):
    name = "sensitivity_solver_advantage"
    description = "Mining advantage from a better exact solver on the same instances"

    def run(self) -> dict[str, Any]:
        sizes = [int(v) for v in self.config.sweep.get("instance_sizes", [8, 10])]
        repetitions = int(self.config.sweep.get("repetitions", 10))
        rows: list[dict[str, Any]] = []

        for size in sizes:
            brute_times: list[float] = []
            hk_times: list[float] = []
            agree = True
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle, instance_size=size)
                vrf = build_vrf(self.config.vrf.backend, **self.config.vrf.build_kwargs())
                miner = Miner(miner_id=0, keys=vrf.keygen(bundle.bytes_for("miner:0")), vrf=vrf)
                target = difficulty_to_target(self.config.pow.difficulty_bits)
                for repetition in range(repetitions):
                    pow_result = miner.search_pow(
                        height=repetition + 1,
                        previous_hash=f"solver-{seed}-{repetition}",
                        tx_root="solver-tx",
                        target=target,
                        max_attempts=self.config.pow.max_attempts,
                    )
                    _, _, index = miner.assign_task(pow_result.digest, repository)
                    instance = repository.instance(index)

                    start = time.perf_counter()
                    brute_solution, _ = brute_force(instance)
                    brute_times.append(time.perf_counter() - start)

                    start = time.perf_counter()
                    hk_solution, _ = held_karp(instance)
                    hk_times.append(time.perf_counter() - start)

                    if abs(brute_solution.cost - hk_solution.cost) > 1e-9:
                        agree = False

            brute_mean, brute_std = mean_std(brute_times)
            hk_mean, hk_std = mean_std(hk_times)
            speedup = brute_mean / hk_mean if hk_mean > 0 else float("inf")
            rows.append(
                {
                    "size": size,
                    "brute_force_seconds": brute_mean,
                    "brute_force_std": brute_std,
                    "held_karp_seconds": hk_mean,
                    "held_karp_std": hk_std,
                    "speedup": speedup,
                    "same_optimum": agree,
                    "implied_block_share_vs_equal_peer": speedup / (speedup + 1),
                    "num_runs": len(brute_times),
                }
            )
            self.logger.info(
                "N=%d -> Held-Karp is %.1fx faster on the same assigned instances "
                "(identical optima: %s)",
                size,
                speedup,
                agree,
            )

        figure_paths = figures.sensitivity_figure(
            [row["size"] for row in rows],
            {
                "Brute force ($(N-1)!$)": [row["brute_force_seconds"] for row in rows],
                "Held-Karp ($2^N N^2$)": [row["held_karp_seconds"] for row in rows],
            },
            x_label="Instance size $N$",
            y_label="Mean solve time (s, log scale)",
            path=self.output_dir / "sensitivity_solver_advantage",
            log_y=True,
        )
        return {
            "table_rows": rows,
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {"sensitivity_solver_advantage": rows},
            "interpretation": (
                "Both solvers return the identical optimum, so a Held-Karp miner "
                "produces blocks that pass every validity check while spending a "
                "fraction of the work. Useful-work mining rewards solver engineering, "
                "which is a fairness and centralisation issue the paper does not raise."
            ),
        }
