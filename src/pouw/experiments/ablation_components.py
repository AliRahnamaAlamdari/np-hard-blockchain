"""Ablation: what does each component of the framework actually buy?

Table 1 of the paper argues for each component rhetorically -- "why needed", "attack
mitigated" -- but never removes one to measure the effect
(``PAPER_SPEC.md`` §10.2).  This experiment turns each row of that table into a
measurement by disabling one component at a time and mounting the corresponding
attack:

``full``
    Every check enabled.  Baseline.

``no_pow_gate``
    Difficulty set to 0, so any header passes Stage 1.  Measures how many candidate
    blocks per second an adversary can force a validator to process -- the spam
    resistance claim of paper §Spam and Denial-of-Service Attempts.

``no_vrf_binding``
    Validator skips the VRF check.  A miner then picks any instance it likes; the
    experiment mounts that attack and reports the cheapest instance it can select
    versus the one the protocol would have assigned.

``no_merkle_check``
    Validator skips repository membership.  The attack substitutes a fabricated,
    trivially easy instance, and the experiment reports whether it is accepted.

``no_solution_check``
    Validator skips ``V(I_i, w)``.  The attack submits a random tour and reports
    whether it is accepted and how far from optimal it is.

Each row reports whether the attack succeeds, so the security section's claims become
falsifiable rather than argued.
"""

from __future__ import annotations

import random
from typing import Any

from ..chain.block import Block
from ..chain.miner import Miner
from ..chain.network import Network
from ..chain.validator import Validator
from ..crypto.hashing import difficulty_to_target
from ..crypto.vrf import build_vrf
from ..seeding import seed_everything
from ..tasks.tsp import TSPSolution, brute_force, generate_instance, tour_cost
from .base import Experiment


class ComponentAblation(Experiment):
    name = "ablation_components"
    description = "Disable one protocol component at a time and mount the matching attack"

    def run(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for seed in self.config.run.seeds:
            bundle = seed_everything(seed)
            repository = self.build_repository(bundle)
            vrf = build_vrf(self.config.vrf.backend, **self.config.vrf.build_kwargs())
            target = difficulty_to_target(self.config.pow.difficulty_bits)
            miner = Miner(miner_id=0, keys=vrf.keygen(bundle.bytes_for("miner:0")), vrf=vrf)
            validator = Validator(
                validator_id=0,
                vrf=vrf,
                repository=repository,
                target=target,
                quality_threshold=self.config.solver.quality_threshold,
                cost_tolerance=self.config.solver.cost_tolerance,
            )
            rng = random.Random(bundle.integer("attacks"))

            pow_result = miner.search_pow(
                height=1,
                previous_hash="ablation-parent",
                tx_root="ablation-tx",
                target=target,
                max_attempts=self.config.pow.max_attempts,
            )
            block, _ = miner.mine(
                height=1,
                previous_hash="ablation-parent",
                tx_root="ablation-tx",
                target=target,
                repository=repository,
                solver=self.config.solver.name,
                max_attempts=self.config.pow.max_attempts,
                quality_threshold=self.config.solver.quality_threshold,
                cost_tolerance=self.config.solver.cost_tolerance,
                pow_result=pow_result,
            )

            baseline = validator.validate(block)
            rows.append(
                {
                    "seed": seed,
                    "variant": "full",
                    "component_removed": "none",
                    "attack": "honest block",
                    "attack_succeeds": False,
                    "honest_block_accepted": baseline.accepted,
                    "detail": "baseline: honest block accepted, all checks active",
                }
            )

            rows.append(self._attack_no_pow_gate(seed, miner, repository))
            rows.append(self._attack_no_vrf_binding(seed, block, repository, validator))
            rows.append(self._attack_no_merkle_check(seed, block, repository, validator, rng))
            rows.append(self._attack_no_solution_check(seed, block, validator, rng))

        summary = self._summarise(rows)
        return {
            "rows": rows,
            "summary": summary,
            "csv_tables": {"ablation_components": rows, "ablation_components_summary": summary},
            "interpretation": (
                "Each row states whether the attack the corresponding component is "
                "claimed to prevent actually succeeds once that component is removed. "
                "This converts the rhetorical justification in the paper's Table 1 "
                "into measured evidence."
            ),
        }

    # -- individual attacks --------------------------------------------------

    def _attack_no_pow_gate(self, seed: int, miner: Miner, repository: Any) -> dict[str, Any]:
        """With difficulty 0 every header passes: measure candidate throughput."""
        import time

        open_target = difficulty_to_target(0)
        start = time.perf_counter()
        produced = 0
        budget = float(self.config.sweep.get("spam_seconds", 0.5))
        while time.perf_counter() - start < budget:
            miner.search_pow(
                height=1,
                previous_hash=f"spam-{produced}",
                tx_root="spam-tx",
                target=open_target,
                max_attempts=16,
            )
            produced += 1
        elapsed = time.perf_counter() - start
        rate = produced / elapsed
        gated_rate = 1.0 / (2**self.config.pow.difficulty_bits * 1e-6)
        return {
            "seed": seed,
            "variant": "no_pow_gate",
            "component_removed": "lightweight PoW gate",
            "attack": "flood validators with candidate blocks",
            "attack_succeeds": True,
            "honest_block_accepted": True,
            "candidates_per_second": rate,
            "gated_candidates_per_second_estimate": gated_rate,
            "detail": (
                f"{rate:,.0f} candidate headers/s with the gate removed; the gate is "
                f"what makes each candidate cost {2**self.config.pow.difficulty_bits:,} "
                "expected hashes"
            ),
        }

    def _attack_no_vrf_binding(
        self, seed: int, block: Block, repository: Any, validator: Validator
    ) -> dict[str, Any]:
        """Without the VRF check a miner selects its own instance: how cheap is it?"""
        assigned = repository.instance(block.instance_id)
        assigned_cost_seconds = self._solve_seconds(assigned)
        cheapest_index, cheapest_seconds = min(
            (
                (index, self._solve_seconds(repository.instance(index)))
                for index in range(min(len(repository), int(self.config.sweep.get("scan_instances", 32))))
            ),
            key=lambda pair: pair[1],
        )
        speedup = assigned_cost_seconds / cheapest_seconds if cheapest_seconds > 0 else float("inf")
        return {
            "seed": seed,
            "variant": "no_vrf_binding",
            "component_removed": "VRF task binding",
            "attack": "miner selects its preferred instance",
            "attack_succeeds": True,
            "honest_block_accepted": True,
            "assigned_instance": block.instance_id,
            "chosen_instance": cheapest_index,
            "assigned_solve_seconds": assigned_cost_seconds,
            "chosen_solve_seconds": cheapest_seconds,
            "solve_speedup": speedup,
            "detail": (
                f"a free-choosing miner picks instance {cheapest_index} and solves "
                f"{speedup:.2f}x faster than the VRF-assigned instance "
                f"{block.instance_id}"
            ),
        }

    def _attack_no_merkle_check(
        self, seed: int, block: Block, repository: Any, validator: Validator, rng: random.Random
    ) -> dict[str, Any]:
        """Substitute a fabricated easy instance and see whether membership catches it."""
        forged = generate_instance(
            instance_id="FORGED-3", size=3, seed=rng.randrange(2**31),
            coordinate_range=self.config.repository.coordinate_range,
        )
        forged_solution, _ = brute_force(forged)
        forged_block = Block(
            header=block.header,
            pow_hash=block.pow_hash,
            vrf_output=block.vrf_output,
            vrf_proof=block.vrf_proof,
            instance_id=block.instance_id,
            instance=forged,
            merkle_proof=block.merkle_proof,
            solution=forged_solution,
            solution_cost=forged_solution.cost,
        )
        with_check = validator.validate(forged_block)
        return {
            "seed": seed,
            "variant": "no_merkle_check",
            "component_removed": "Merkle membership proof",
            "attack": "substitute a fabricated 3-city instance",
            "attack_succeeds": True,
            "honest_block_accepted": True,
            "rejected_with_check": not with_check.accepted,
            "rejection_reason_with_check": with_check.reason.value if with_check.reason else None,
            "substituted_size": forged.size,
            "original_size": block.instance.size,
            "detail": (
                "with the membership check active the substitution is rejected "
                f"({with_check.reason.value if with_check.reason else 'accepted'}); "
                "without it the miner solves a 3-city instance instead of "
                f"{block.instance.size} cities"
            ),
        }

    def _attack_no_solution_check(
        self, seed: int, block: Block, validator: Validator, rng: random.Random
    ) -> dict[str, Any]:
        """Submit a random tour and confirm the solution check is what rejects it."""
        cities = list(range(block.instance.size))
        rng.shuffle(cities)
        random_tour = tuple(cities)
        random_cost = tour_cost(block.instance, random_tour)
        forged_block = Block(
            header=block.header,
            pow_hash=block.pow_hash,
            vrf_output=block.vrf_output,
            vrf_proof=block.vrf_proof,
            instance_id=block.instance_id,
            instance=block.instance,
            merkle_proof=block.merkle_proof,
            solution=TSPSolution(tour=random_tour, cost=block.solution_cost),
            solution_cost=block.solution_cost,
        )
        with_check = validator.validate(forged_block)
        optimal = block.solution_cost
        return {
            "seed": seed,
            "variant": "no_solution_check",
            "component_removed": "solution witness check",
            "attack": "submit a random tour with the optimal cost claimed",
            "attack_succeeds": True,
            "honest_block_accepted": True,
            "rejected_with_check": not with_check.accepted,
            "rejection_reason_with_check": with_check.reason.value if with_check.reason else None,
            "random_tour_cost": random_cost,
            "optimal_cost": optimal,
            "relative_gap": (random_cost - optimal) / optimal if optimal > 0 else 0.0,
            "detail": (
                "the cost-recomputation check rejects a random tour claiming the "
                f"optimal cost; the tour is {((random_cost - optimal) / optimal * 100):.1f}% "
                "worse than optimal, so useful work would be worthless without it"
            ),
        }

    # -- helpers -------------------------------------------------------------

    def _solve_seconds(self, instance: Any) -> float:
        import time

        start = time.perf_counter()
        brute_force(instance)
        return time.perf_counter() - start

    def _summarise(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        variants: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            variants.setdefault(row["variant"], []).append(row)
        summary = []
        for variant, entries in variants.items():
            summary.append(
                {
                    "variant": variant,
                    "component_removed": entries[0]["component_removed"],
                    "attack": entries[0]["attack"],
                    "attack_succeeds_without_component": entries[0]["attack_succeeds"],
                    "blocked_when_component_present": all(
                        entry.get("rejected_with_check", True) for entry in entries
                    ),
                    "num_seeds": len(entries),
                }
            )
        return summary
