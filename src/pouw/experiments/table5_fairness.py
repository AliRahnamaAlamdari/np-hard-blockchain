"""Reproduces the paper's Table 5 (``tab:fairness``) and its derived statistics.

The paper runs 5 equal-power miners for 500 blocks once, and reports the block
counts, a population standard deviation, a max/min ratio, and Jain's index (L506).
All four are reproduced here.

Additions:

* multiple seeds, so the spread of the fairness statistics themselves is visible --
  the paper's single realisation cannot distinguish "fair protocol" from "lucky run";
* a Pearson chi-square goodness-of-fit test against uniformity, the standard test for
  exactly this claim, which the paper does not report;
* the Gini coefficient, because Jain's index saturates near 1 and separates plausible
  outcomes poorly.

Checkpointing is enabled here because this is the longest run in the suite.
"""

from __future__ import annotations

from typing import Any

from .. import checkpoint as checkpoint_module
from ..chain.network import Network
from ..metrics.fairness import analyse
from ..metrics.convergence import analyse_convergence
from ..reporting import figures
from ..seeding import seed_everything
from .base import Experiment, mean_std


class FairnessExperiment(Experiment):
    name = "table5_fairness"
    description = "Block distribution across equal-power miners, with a uniformity test"

    def run(self) -> dict[str, Any]:
        num_miners = self.config.network.num_miners
        num_blocks = self.config.network.num_blocks
        config_hash = checkpoint_module.config_fingerprint(self.config)

        per_seed: list[dict[str, Any]] = []
        pooled_counts = {miner_id: 0 for miner_id in range(num_miners)}
        seed_count_maps: list[dict[int, int]] = []

        for seed in self.config.run.seeds:
            bundle = seed_everything(seed)
            repository = self.build_repository(bundle)
            network = Network(
                num_miners=num_miners,
                repository=repository,
                difficulty_bits=self.config.pow.difficulty_bits,
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

            checkpoint_file = self.checkpoint_file(seed)
            records: list[dict[str, Any]] = []
            if self.config.run.checkpoint_every_blocks > 0:
                resumed = checkpoint_module.load(checkpoint_file, config_hash)
                if resumed is not None and resumed.completed_blocks >= num_blocks:
                    self.logger.info("seed %d already complete; using checkpoint", seed)
                    records = resumed.records

            if not records:
                interval = self.config.run.checkpoint_every_blocks

                def on_block(height: int, record: Any, _seed: int = seed) -> None:
                    records.append(record.to_dict())
                    if interval and height % interval == 0:
                        checkpoint_module.save(
                            self.checkpoint_file(_seed),
                            checkpoint_module.Checkpoint(
                                experiment=self.name,
                                config_hash=config_hash,
                                seed=_seed,
                                completed_blocks=height,
                                records=records,
                                state={},
                            ),
                        )

                result = network.run(
                    num_blocks=num_blocks,
                    log_every=self.config.run.log_every_blocks,
                    on_block=on_block,
                )
                counts = dict(result.blocks_won)
                convergence = analyse_convergence(
                    result.node_tips, [record.all_accepted for record in result.blocks]
                ).to_dict()
            else:
                counts = {miner_id: 0 for miner_id in range(num_miners)}
                for record in records:
                    counts[record["winner_id"]] += 1
                convergence = {"converged": True, "resumed": True}

            report = analyse([counts[miner_id] for miner_id in sorted(counts)])
            seed_count_maps.append(counts)
            for miner_id, count in counts.items():
                pooled_counts[miner_id] += count

            per_seed.append(
                {
                    "seed": seed,
                    # Keyed by miner id. Distinct from the report's positional
                    # ``counts`` list, which would otherwise collide on this key.
                    "counts_by_miner": {str(k): v for k, v in sorted(counts.items())},
                    "convergence": convergence,
                    **report.to_dict(),
                }
            )
            self.logger.info(
                "seed %d -> jain=%.4f std=%.4f max/min=%.4f chi2=%.4f p=%.4f",
                seed,
                report.jain_index,
                report.std_dev,
                report.max_min_ratio,
                report.chi_square,
                report.chi_square_p_value,
            )
            if self.config.run.checkpoint_every_blocks > 0:
                checkpoint_module.clear(self.checkpoint_file(seed))

        summary_rows = []
        for statistic in (
            "std_dev",
            "max_min_ratio",
            "jain_index",
            "gini",
            "chi_square",
            "chi_square_p_value",
        ):
            values = [entry[statistic] for entry in per_seed]
            mean, std = mean_std(values)
            summary_rows.append(
                {
                    "statistic": statistic,
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                }
            )

        representative_seed = self.config.run.seeds[0]
        representative_counts = seed_count_maps[0]
        table_rows = [
            {
                "miner_id": miner_id,
                "node_label": f"Node {miner_id}",
                "blocks_won": count,
                "percentage": count / sum(representative_counts.values()) * 100.0,
            }
            for miner_id, count in sorted(representative_counts.items())
        ]

        figure_paths = figures.fairness_figure(
            representative_counts,
            sum(representative_counts.values()),
            self.output_dir / "table5_fairness",
            seed_counts=seed_count_maps,
        )

        return {
            "representative_seed": representative_seed,
            "table_rows": table_rows,
            "per_seed": per_seed,
            "summary_rows": summary_rows,
            "pooled_counts": {str(k): v for k, v in sorted(pooled_counts.items())},
            "pooled_report": analyse(
                [pooled_counts[miner_id] for miner_id in sorted(pooled_counts)]
            ).to_dict(),
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {
                "table5_fairness": table_rows,
                "table5_per_seed": [
                    {
                        k: v
                        for k, v in entry.items()
                        if k not in ("counts", "counts_by_miner", "convergence")
                    }
                    for entry in per_seed
                ],
                "table5_summary": summary_rows,
            },
            "paper_reference": {
                "table": "tab:fairness",
                "reported_counts": [101, 92, 105, 91, 111],
                "reported_std_dev": 7.6420,
                "reported_max_min_ratio": 1.2198,
                "reported_jain_index": 0.9942,
                "note": (
                    "The paper's counts are one realisation of a multinomial draw. "
                    "Its reported statistics are exactly reproducible from those "
                    "counts and are pinned in tests/test_metrics.py."
                ),
            },
        }
