"""Reproduces the paper's Table 3 (``tab:timeallocation``).

Sweeps the Stage-1 difficulty across the Low/Medium/High tiers and measures how
block-production time divides between the PoW gate and the useful work.  The paper
gives no numeric target for any tier (``PAPER_SPEC.md`` §11-U4), so the tiers are
config values in ``configs/experiments/table3_time_allocation.yaml``.

Two things the paper does not report are added because they change how the headline
number should be read:

* error bars across independent seeds -- the paper reports a single value per tier;
* a third share that counts VRF, Merkle, packaging and validation time as part of
  block production.  The paper's two columns sum to 100% by construction because that
  overhead is excluded from the denominator, so "90.6% useful" is an upper bound that
  ignores every cost the framework itself adds.
"""

from __future__ import annotations

from typing import Any

from ..chain.network import Network
from ..metrics.timing import compute_shares
from ..reporting import figures
from ..seeding import seed_everything
from .base import Experiment, mean_std


class TimeAllocationExperiment(Experiment):
    name = "table3_time_allocation"
    description = "Time share between the PoW gate and useful work across difficulty tiers"
    # Prefix for the CSV tables; kept separate from ``name`` so the original
    # experiment's file names do not change.
    table_prefix = "table3"

    def run(self) -> dict[str, Any]:
        tiers = self.config.sweep.get("difficulty_tiers")
        if not tiers:
            raise ValueError(
                "config.sweep.difficulty_tiers must list the tiers, e.g. "
                "[{label: Low, difficulty_bits: 12}, ...]"
            )
        blocks_per_tier = int(self.config.sweep.get("blocks_per_tier", self.config.network.num_blocks))

        rows: list[dict[str, Any]] = []
        per_seed_rows: list[dict[str, Any]] = []

        for tier in tiers:
            label = tier["label"]
            difficulty_bits = int(tier["difficulty_bits"])
            instance_size = int(tier.get("instance_size", self.config.repository.instance_size))
            self.logger.info(
                "tier %s: difficulty=%d bits, instance size N=%d, %d blocks x %d seeds",
                label,
                difficulty_bits,
                instance_size,
                blocks_per_tier,
                len(self.config.run.seeds),
            )

            seed_shares: list[float] = []
            seed_pow: list[float] = []
            seed_solve: list[float] = []
            seed_overhead_shares: list[float] = []
            seed_miner_overhead_shares: list[float] = []
            all_pow: list[float] = []
            all_solve: list[float] = []
            all_overhead: list[float] = []
            all_miner_overhead: list[float] = []
            total_attempts = 0

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle, instance_size=instance_size)
                network = Network(
                    num_miners=self.config.network.num_miners,
                    repository=repository,
                    difficulty_bits=difficulty_bits,
                    vrf_backend=self.config.vrf.backend,
                    seeds=bundle,
                    solver=self.config.solver.name,
                    hash_rates=self.config.network.hash_rates,
                    max_attempts=self.config.pow.max_attempts,
                    quality_threshold=self.config.solver.quality_threshold,
                    cost_tolerance=self.config.solver.cost_tolerance,
                    tie_break=self.config.network.tie_break,
                    vrf_kwargs=self.config.vrf.build_kwargs(),
                )
                result = network.run(
                    num_blocks=blocks_per_tier,
                    log_every=self.config.run.log_every_blocks,
                )
                pow_seconds = [record.pow_seconds for record in result.blocks]
                solve_seconds = [record.solve_seconds for record in result.blocks]
                # Miner-side overhead: what the block producer itself pays on top of
                # PoW and solving (VRF proving, Merkle proof retrieval, packaging).
                miner_overhead = [
                    record.vrf_prove_seconds
                    + record.merkle_proof_seconds
                    + record.package_seconds
                    for record in result.blocks
                ]
                # Network-wide overhead additionally counts validation at every node.
                network_overhead = [
                    miner + sum(record.validation_seconds.values())
                    for miner, record in zip(miner_overhead, result.blocks)
                ]
                report = compute_shares(pow_seconds, solve_seconds, network_overhead)
                miner_report = compute_shares(pow_seconds, solve_seconds, miner_overhead)

                seed_shares.append(report.useful_share_percent)
                seed_pow.append(report.mean_pow_seconds)
                seed_solve.append(report.mean_solve_seconds)
                seed_overhead_shares.append(report.useful_share_percent_with_overhead)
                seed_miner_overhead_shares.append(
                    miner_report.useful_share_percent_with_overhead
                )
                all_pow.extend(pow_seconds)
                all_solve.extend(solve_seconds)
                all_overhead.extend(network_overhead)
                all_miner_overhead.extend(miner_overhead)
                total_attempts += sum(record.network_attempts for record in result.blocks)

                per_seed_rows.append(
                    {
                        "difficulty_label": label,
                        "difficulty_bits": difficulty_bits,
                        "instance_size": instance_size,
                        "seed": seed,
                        "blocks": len(result.blocks),
                        "converged": result.converged,
                        **report.to_dict(),
                        "useful_share_percent_with_miner_overhead": (
                            miner_report.useful_share_percent_with_overhead
                        ),
                    }
                )

            pooled = compute_shares(all_pow, all_solve, all_overhead)
            pooled_miner = compute_shares(all_pow, all_solve, all_miner_overhead)
            share_mean, share_std = mean_std(seed_shares)
            pow_mean, pow_std = mean_std(seed_pow)
            solve_mean, solve_std = mean_std(seed_solve)
            overhead_share_mean, overhead_share_std = mean_std(seed_overhead_shares)
            miner_overhead_share_mean, miner_overhead_share_std = mean_std(
                seed_miner_overhead_shares
            )

            rows.append(
                {
                    "difficulty_label": label,
                    "difficulty_bits": difficulty_bits,
                    "instance_size": instance_size,
                    "blocks_per_seed": blocks_per_tier,
                    "num_seeds": len(self.config.run.seeds),
                    "mean_pow_seconds": pow_mean,
                    "std_pow_seconds": pow_std,
                    "mean_solve_seconds": solve_mean,
                    "std_solve_seconds": solve_std,
                    "pow_share_percent": 100.0 - share_mean,
                    "useful_share_percent": share_mean,
                    "useful_share_std": share_std,
                    "useful_share_percent_with_miner_overhead": miner_overhead_share_mean,
                    "useful_share_percent_with_overhead": overhead_share_mean,
                    "useful_share_with_miner_overhead_std": miner_overhead_share_std,
                    "useful_share_with_overhead_std": overhead_share_std,
                    "mean_miner_overhead_seconds": pooled_miner.mean_overhead_seconds,
                    "mean_overhead_seconds": pooled.mean_overhead_seconds,
                    "pooled_useful_share_percent": pooled.useful_share_percent,
                    "mean_of_ratios_useful_share": pooled.useful_share_percent_mean_of_ratios,
                    # In-protocol hash rate: every attempt of the race over its wall time.
                    "hash_rate_hps": total_attempts / sum(all_pow),
                }
            )
            self.logger.info(
                "tier %s -> useful share %.2f%% +/- %.2f "
                "(with miner overhead: %.2f%%, network-wide: %.2f%%)",
                label,
                share_mean,
                share_std,
                miner_overhead_share_mean,
                overhead_share_mean,
            )

        figure_paths = figures.time_allocation_figure(rows, self.output_dir / self.name)
        return {
            "table_rows": rows,
            "per_seed_rows": per_seed_rows,
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {
                self.name: rows,
                f"{self.table_prefix}_per_seed": per_seed_rows,
            },
            "paper_reference": {
                "table": "tab:timeallocation",
                "reported": [
                    {"difficulty": "Low", "pow_s": 0.1660, "useful_s": 1.6009, "pow_pct": 9.40, "useful_pct": 90.60},
                    {"difficulty": "Medium", "pow_s": 1.8997, "useful_s": 1.6699, "pow_pct": 53.22, "useful_pct": 46.78},
                    {"difficulty": "High", "pow_s": 67.4985, "useful_s": 1.5932, "pow_pct": 97.69, "useful_pct": 2.31},
                ],
                "note": (
                    "Absolute seconds are machine dependent and are not expected to "
                    "match. The reproducible claim is that the useful-work share is "
                    "monotone decreasing in PoW difficulty and spans a wide range."
                ),
            },
        }


class TimeAllocationSweepExperiment(TimeAllocationExperiment):
    """The same measurement over a dense difficulty sweep, for the share-vs-$d$ figure.

    Kept as its own experiment so the three-tier run behind the paper's quoted
    numbers is not overwritten by the sweep.
    """

    name = "time_allocation_sweep"
    description = "Useful-work share across a dense sweep of gate difficulty"
    table_prefix = "time_allocation_sweep"
