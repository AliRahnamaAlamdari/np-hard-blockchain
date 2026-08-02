"""Measured comparison of in-house baseline designs (reviewer comment 5).

Runs ``pure_pow``, ``miner_chosen``, ``vrf_only``, and ``ours`` on a common harness and
measures verification cost, task-control resistance, grinding freedom, and precomputation
exposure. External protocols (Ofelimos, PoSo, VRF-PoW) are *not* run -- no runnable
reference implementations are available -- and remain literature-reported in the paper's
comparison table. This experiment produces only the measured columns.
"""

from __future__ import annotations

import random
from typing import Any

from ..chain.baselines import BASELINES, measure_baseline
from ..seeding import seed_everything
from ..stats_utils import mean_std
from ..tasks.problems.repository import GenericRepository, GenericRepositorySpec
from .base import Experiment


class BaselineComparisonExperiment(Experiment):
    name = "baselines"
    description = "Measured comparison of in-house baseline designs on a common harness"

    def run(self) -> dict[str, Any]:
        pclass = self.config.sweep.get("problem_class", "tsp")
        size = int(self.config.sweep.get("instance_size", 10))
        difficulty = self.config.pow.difficulty_bits
        params = {"tsp": {"coordinate_range": (0.0, 100.0)}}.get(pclass, {})

        rows: list[dict[str, Any]] = []
        for name, spec in BASELINES.items():
            verify_means = []
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repo = GenericRepository.from_spec(
                    GenericRepositorySpec(pclass, num_instances=16, instance_size=size,
                                          seed=bundle.integer("repo") % (2**31), params=params))
                m = measure_baseline(spec, repo, difficulty, bundle.bytes_for("miner"),
                                     trials=int(self.config.sweep.get("verify_trials", 200)),
                                     rng=random.Random(bundle.integer("bl")))
                verify_means.append(m.verify_ms)
                last = m
            vmean, vstd = mean_std(verify_means)
            rows.append({
                "baseline": name,
                "verify_ms": vmean,
                "verify_ms_std": vstd,
                "useful_work": last.useful_work,
                "spam_entry_cost": last.spam_entry_cost,
                "task_control_resistance": last.task_control_resistance,
                "grinding_tasks_per_seed": last.grinding_tasks_per_seed,
                "precomputation_exposure": last.precomputation_exposure,
            })
            self.logger.info("%s verify=%.4f±%.4f ms control=%s grinding=%.0f",
                             name, vmean, vstd, last.task_control_resistance,
                             last.grinding_tasks_per_seed)

        return {
            "table_rows": rows,
            "csv_tables": {"baselines": rows},
            "note": (
                "In-house design-class representatives, measured on one harness. External "
                "protocols (Ofelimos, Proof-of-Solution, VRF-PoW) are literature-reported "
                "in the paper, not measured here."
            ),
        }
