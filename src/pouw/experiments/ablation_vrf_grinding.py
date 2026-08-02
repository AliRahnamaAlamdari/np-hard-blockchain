"""Ablation: does the VRF primitive actually deliver goals G3 and G4?

The paper's grinding argument (L375) is that one PoW seed yields one task, so
sampling ``k`` tasks costs ``k/p`` hash attempts.  That argument holds only if the VRF
has **unique output** -- exactly one acceptable ``(y, pi)`` per (key, seed) pair.

The paper never names a VRF construction and describes its primitive as "VRF-like
signing" over an ECDSA library (``PAPER_SPEC.md`` §11-I6).  ECDSA signatures are not
unique: the per-signature nonce is free, so a miner can re-sign the same seed and get
a different task each time at zero PoW cost.

This experiment measures that directly.  For each backend, fix a single valid PoW
seed and evaluate the VRF repeatedly, counting how many distinct task indices become
reachable.  A real VRF is pinned at 1.  A malleable one approaches the repository
size, which collapses the grinding cost from ``k/p`` to ``1/p`` and lets a miner
choose its instance -- the precise failure Table 1 of the paper claims the VRF
prevents.

The "effective grinding cost" column is the headline: hash attempts required per
distinct task obtained.
"""

from __future__ import annotations

from typing import Any

from ..crypto.hashing import difficulty_to_target, expected_attempts
from ..crypto.vrf import build_vrf
from ..chain.miner import Miner
from ..reporting import figures
from ..seeding import seed_everything
from ..tasks.repository import index_from_vrf_output
from .base import Experiment, mean_std


class VRFGrindingAblation(Experiment):
    name = "ablation_vrf_grinding"
    description = "Distinct tasks reachable per PoW seed, by VRF primitive"

    def run(self) -> dict[str, Any]:
        backends = self.config.sweep.get(
            "vrf_backends", ["ecvrf", "ecdsa_pseudo", "hash_stub"]
        )
        attempt_budgets = [int(v) for v in self.config.sweep.get("attempt_budgets", [1, 2, 5, 10, 50, 100])]
        max_attempts_budget = max(attempt_budgets)
        target = difficulty_to_target(self.config.pow.difficulty_bits)

        rows: list[dict[str, Any]] = []
        curves: dict[str, list[float]] = {}

        for backend in backends:
            self.logger.info("grinding against backend %s", backend)
            vrf_kwargs = {"curve_name": self.config.vrf.curve} if backend == "ecdsa_pseudo" else {}
            vrf = build_vrf(backend, **vrf_kwargs)
            supports_nonce_grinding = backend == "ecdsa_pseudo"

            per_budget: dict[int, list[float]] = {budget: [] for budget in attempt_budgets}
            verified_all = True

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle)
                miner = Miner(
                    miner_id=0, keys=vrf.keygen(bundle.bytes_for("miner:0")), vrf=vrf
                )
                pow_result = miner.search_pow(
                    height=1,
                    previous_hash="grinding-parent",
                    tx_root="grinding-tx",
                    target=target,
                    max_attempts=self.config.pow.max_attempts,
                )
                seed_bytes = pow_result.digest

                distinct: set[int] = set()
                trajectory: list[int] = []
                for attempt in range(max_attempts_budget):
                    if supports_nonce_grinding:
                        beta, proof = vrf.prove(miner.keys.secret, seed_bytes, attempt=attempt)
                    else:
                        beta, proof = vrf.prove(miner.keys.secret, seed_bytes)
                    # Every proof produced here must pass the real validator check --
                    # otherwise the "attack" is only producing rejected blocks.
                    if vrf.verify(miner.keys.public, seed_bytes, proof) != beta:
                        verified_all = False
                    distinct.add(index_from_vrf_output(beta, len(repository)))
                    trajectory.append(len(distinct))

                for budget in attempt_budgets:
                    per_budget[budget].append(float(trajectory[budget - 1]))

            curve = [mean_std(per_budget[budget])[0] for budget in attempt_budgets]
            curves[backend] = curve
            max_distinct = curve[-1]
            hashes_per_seed = expected_attempts(self.config.pow.difficulty_bits)

            rows.append(
                {
                    "vrf_backend": backend,
                    "unique_output": vrf.is_unique_output(),
                    "all_proofs_verified": verified_all,
                    "distinct_tasks_1_eval": curve[0],
                    "distinct_tasks_max_eval": max_distinct,
                    "evaluations_at_max": max_attempts_budget,
                    "expected_hashes_per_seed": hashes_per_seed,
                    "hashes_per_distinct_task": hashes_per_seed / max_distinct if max_distinct else float("inf"),
                    "grinding_cost_reduction": max_distinct,
                    "repository_size": self.config.repository.num_instances,
                }
            )
            self.logger.info(
                "%s -> %.1f distinct tasks from ONE PoW seed after %d VRF evaluations "
                "(all proofs verified: %s)",
                backend,
                max_distinct,
                max_attempts_budget,
                verified_all,
            )

        figure_paths = figures.grinding_figure(
            attempt_budgets, curves, self.output_dir / "ablation_vrf_grinding"
        )

        return {
            "table_rows": rows,
            "attempt_budgets": attempt_budgets,
            "curves": curves,
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {"ablation_vrf_grinding": rows},
            "interpretation": (
                "A unique-output VRF pins distinct tasks per seed at 1, which is what "
                "the paper's k/p grinding bound assumes. A malleable 'VRF-like' "
                "signature lets one paid seed reach many tasks, reducing the cost of "
                "task selection to a single PoW and defeating design goals G3 and G4."
            ),
        }
