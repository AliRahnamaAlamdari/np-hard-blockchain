"""Chain convergence and useful-work correctness (paper metrics M1 and §Convergence).

The paper reports this qualitatively: "No persistent chain inconsistency was
observed" (L483), with no fork count, no round count, and no independent check that
the mined solutions were actually optimal (``PAPER_SPEC.md`` §11-U15).

This experiment produces both numbers: the number of nodes agreeing on the tip after
every round, and the fraction of mined tours that match an independently computed
optimum from a different algorithm.
"""

from __future__ import annotations

from typing import Any

from ..chain.network import Network
from ..metrics.convergence import analyse_convergence, analyse_correctness
from ..seeding import seed_everything
from ..tasks.tsp import held_karp
from .base import Experiment, mean_std


class ConvergenceExperiment(Experiment):
    name = "convergence"
    description = "Node agreement on the chain tip and independent optimality checking"

    def run(self) -> dict[str, Any]:
        check_optimality = bool(self.config.sweep.get("check_optimality", True))
        rows: list[dict[str, Any]] = []

        for seed in self.config.run.seeds:
            bundle = seed_everything(seed)
            repository = self.build_repository(bundle)
            network = Network(
                num_miners=self.config.network.num_miners,
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
            result = network.run(
                num_blocks=self.config.network.num_blocks,
                log_every=self.config.run.log_every_blocks,
            )
            convergence = analyse_convergence(
                result.node_tips, [record.all_accepted for record in result.blocks]
            )

            correctness: dict[str, Any] = {}
            if check_optimality:
                achieved = []
                optimal = []
                for record in result.blocks:
                    instance = repository.instance(record.instance_id)
                    reference, _ = held_karp(instance)
                    achieved.append(record.solution_cost)
                    optimal.append(reference.cost)
                correctness = analyse_correctness(achieved, optimal).to_dict()

            row = {
                "seed": seed,
                **convergence.to_dict(),
                **{f"correctness_{k}": v for k, v in correctness.items()},
            }
            rows.append(row)
            self.logger.info(
                "seed %d -> converged=%s tips=%d accepted=%d/%d%s",
                seed,
                convergence.converged,
                convergence.distinct_tips,
                convergence.blocks_accepted_by_all,
                convergence.num_blocks,
                f" optimal={correctness.get('optimality_rate', float('nan')):.4f}"
                if correctness
                else "",
            )

        acceptance = [row["acceptance_rate"] for row in rows]
        acceptance_mean, acceptance_std = mean_std(acceptance)
        summary = {
            "num_seeds": len(rows),
            "all_converged": all(row["converged"] for row in rows),
            "acceptance_rate_mean": acceptance_mean,
            "acceptance_rate_std": acceptance_std,
            "total_blocks": sum(row["num_blocks"] for row in rows),
        }
        if check_optimality:
            optimality = [row["correctness_optimality_rate"] for row in rows]
            optimality_mean, optimality_std = mean_std(optimality)
            summary["optimality_rate_mean"] = optimality_mean
            summary["optimality_rate_std"] = optimality_std
            summary["max_relative_gap"] = max(row["correctness_max_relative_gap"] for row in rows)

        self.logger.info("summary: %s", summary)
        return {
            "rows": rows,
            "summary": summary,
            "csv_tables": {"convergence": rows},
            "interpretation": (
                "Convergence is reported as a count of distinct node tips rather than "
                "an assertion, and optimality is checked against a second exact "
                "algorithm rather than assumed from the solver's construction."
            ),
        }
