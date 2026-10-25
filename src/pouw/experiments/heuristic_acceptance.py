"""Exercises the threshold-based acceptance path of ``VerifySolution``.

The paper describes a "protocol-defined quality bound" (L387) for solutions that are
feasible but not provably optimal, and never instantiates it
(``PAPER_SPEC.md`` §11-U7).  Without an instantiation the claim that the framework
supports approximate useful work is untested, and the objection that the evaluation
only covers instances small enough to solve exactly stands.

This experiment closes that gap.  The repository publishes a reference cost per
instance, committed inside the Merkle leaf, and validators accept a submitted tour
when its recomputed cost is within ``(1 + threshold)`` of that bound.  Miners run
heuristics of differing strength on instances far beyond exact reach, so the same
bound accepts some blocks and rejects others -- which is what makes the check a
check.

Two gaps are reported and they are not the same quantity.  The gap to the *bound* is
what validators enforce.  The gap to the *optimum* is what the useful work is
actually worth, and it can only be measured where an exact solve is still
affordable; above that size it is unavailable and the table says so rather than
substituting the bound for it.
"""

from __future__ import annotations

from typing import Any

from ..crypto.hashing import difficulty_to_target
from ..crypto.vrf import build_vrf
from ..chain.miner import Miner
from ..chain.validator import Validator
from ..seeding import seed_everything
from ..tasks.repository import reference_cost as published_reference_cost
from ..tasks.tsp import held_karp
from .base import Experiment, mean_std

# Above this size an exact solve costs more than the whole rest of the experiment.
DEFAULT_EXACT_MAX_SIZE = 20


class HeuristicAcceptanceExperiment(Experiment):
    name = "heuristic_acceptance"
    description = "Threshold-based acceptance of heuristic solutions beyond exact reach"

    def run(self) -> dict[str, Any]:
        sizes = [int(v) for v in self.config.sweep.get("instance_sizes", [50, 100, 200])]
        solvers = list(
            self.config.sweep.get(
                "miner_solvers", ["nearest_neighbor", "nearest_neighbor_2opt", "or_opt"]
            )
        )
        thresholds = [float(v) for v in self.config.sweep.get("thresholds", [0.02, 0.10])]
        blocks = int(self.config.sweep.get("blocks_per_cell", 5))
        exact_max = int(
            self.config.sweep.get("exact_reference_max_size", DEFAULT_EXACT_MAX_SIZE)
        )
        if self.config.repository.reference_solver is None:
            raise ValueError(
                "heuristic_acceptance requires repository.reference_solver: the "
                "quality bound has to be published by the repository"
            )
        target = difficulty_to_target(self.config.pow.difficulty_bits)

        rows: list[dict[str, Any]] = []
        per_block_rows: list[dict[str, Any]] = []

        for size in sizes:
            exact_available = size <= exact_max
            # Solve each cell once per (solver, seed) and score it against every
            # threshold, so raising the threshold list costs no extra solving.
            for solver in solvers:
                samples: list[dict[str, Any]] = []
                for seed in self.config.run.seeds:
                    bundle = seed_everything(seed)
                    repository = self.build_repository(bundle, instance_size=size)
                    vrf = build_vrf(
                        self.config.vrf.backend, **self.config.vrf.build_kwargs()
                    )
                    miner = Miner(
                        miner_id=0,
                        keys=vrf.keygen(bundle.bytes_for("miner:0")),
                        vrf=vrf,
                    )
                    for block_index in range(blocks):
                        block, timings = miner.mine(
                            height=block_index + 1,
                            previous_hash=f"parent-{seed}-{block_index}",
                            tx_root=f"tx-{seed}-{block_index}",
                            target=target,
                            repository=repository,
                            solver=solver,
                            max_attempts=self.config.pow.max_attempts,
                        )
                        bound = published_reference_cost(block.instance)
                        optimum = None
                        if exact_available:
                            optimum = held_karp(block.instance)[0].cost
                        samples.append(
                            {
                                "seed": seed,
                                "block": block_index,
                                "block_object": block,
                                "repository": repository,
                                "vrf": vrf,
                                "cost": block.solution.cost,
                                "bound": bound,
                                "optimum": optimum,
                                "solve_seconds": timings["solve_s"],
                            }
                        )

                for threshold in thresholds:
                    accepted = 0
                    verify_seconds: list[float] = []
                    for sample in samples:
                        validator = Validator(
                            validator_id=0,
                            vrf=sample["vrf"],
                            repository=sample["repository"],
                            target=target,
                            quality_threshold=threshold,
                            cost_tolerance=self.config.solver.cost_tolerance,
                        )
                        result = validator.validate(sample["block_object"])
                        verify_seconds.append(result.total_seconds)
                        accepted += int(result.accepted)
                        per_block_rows.append(
                            {
                                "solver": solver,
                                "size": size,
                                "threshold": threshold,
                                "seed": sample["seed"],
                                "block": sample["block"],
                                "cost": sample["cost"],
                                "reference_cost": sample["bound"],
                                "optimum_cost": sample["optimum"],
                                "gap_to_bound_percent": _gap(
                                    sample["cost"], sample["bound"]
                                ),
                                "gap_to_optimum_percent": _gap(
                                    sample["cost"], sample["optimum"]
                                ),
                                "accepted": result.accepted,
                                "reject_reason": result.reason.value
                                if result.reason
                                else None,
                                "solve_seconds": sample["solve_seconds"],
                                "verify_seconds": result.total_seconds,
                            }
                        )

                    bound_gaps = [
                        _gap(sample["cost"], sample["bound"]) for sample in samples
                    ]
                    optimum_gaps = [
                        _gap(sample["cost"], sample["optimum"])
                        for sample in samples
                        if sample["optimum"] is not None
                    ]
                    solve_mean, solve_std = mean_std(
                        [sample["solve_seconds"] for sample in samples]
                    )
                    verify_mean, _ = mean_std(verify_seconds)
                    gap_mean, gap_std = mean_std(bound_gaps)
                    rows.append(
                        {
                            "solver": solver,
                            "size": size,
                            "threshold": threshold,
                            "num_blocks": len(samples),
                            "accepted": accepted,
                            "acceptance_rate": accepted / len(samples),
                            "mean_gap_to_bound_percent": gap_mean,
                            "std_gap_to_bound_percent": gap_std,
                            "mean_gap_to_optimum_percent": (
                                sum(optimum_gaps) / len(optimum_gaps)
                                if optimum_gaps
                                else None
                            ),
                            "exact_optimum_available": exact_available,
                            "mean_solve_ms": solve_mean * 1000.0,
                            "std_solve_ms": solve_std * 1000.0,
                            "mean_verify_ms": verify_mean * 1000.0,
                            "reference_solver": self.config.repository.reference_solver,
                        }
                    )
                    self.logger.info(
                        "N=%d %s @ threshold %.0f%% -> accepted %d/%d, "
                        "gap to bound %.2f%%, solve %.2f ms",
                        size,
                        solver,
                        threshold * 100.0,
                        accepted,
                        len(samples),
                        gap_mean,
                        solve_mean * 1000.0,
                    )

        return {
            "table_rows": rows,
            "per_block_rows": per_block_rows,
            "csv_tables": {
                "heuristic_acceptance": rows,
                "heuristic_acceptance_per_block": per_block_rows,
            },
            "interpretation": (
                "The quality bound is published by the repository and committed in "
                "the Merkle leaf, so a miner cannot raise it without breaking its "
                "own membership proof. Acceptance therefore depends on the witness, "
                "its recomputed cost, and the committed bound alone -- never on "
                "which solver produced the tour."
            ),
        }


def _gap(cost: float, reference: float | None) -> float | None:
    """Percentage by which ``cost`` exceeds ``reference``; None when unavailable."""
    if reference is None or reference <= 0:
        return None
    return (cost / reference - 1.0) * 100.0
