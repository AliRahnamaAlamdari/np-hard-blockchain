"""Reproduces the paper's Table 6 (``tab:validationcost``) and the asymmetry claim.

Profiles the four validation steps of Algorithm 2 and compares their total against
the useful-work solve time.

The paper reports a total of 0.005069 ms (5.069 microseconds), of which 0.001958 ms
is "VRF Proof Verification".  No elliptic-curve verification runs in 1.958
microseconds in CPython -- it is roughly three orders of magnitude too fast -- so this
experiment measures all three VRF backends.  The comparison identifies which
primitive the paper's reported latency is consistent with, which matters because a
keyless verification step is not a VRF check at all (``PAPER_SPEC.md`` §11-I6).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..chain.network import Network
from ..metrics.timing import compute_asymmetry
from ..reporting import figures
from ..seeding import seed_everything
from .base import Experiment, mean_std

STEP_LABELS = {
    "pow_check": "1. PoW Check (Hash \\& Difficulty)",
    "vrf_verify": "2. VRF Proof Verification",
    "index_check": "3. Task Index Reconstruction",
    "merkle_verify": "4. Merkle Proof Verification",
    "solution_check": "5. Solution Feasibility \\& Cost Check",
}


class ValidationCostExperiment(Experiment):
    name = "table6_validation_cost"
    description = "Per-step validation latency and the solve-versus-verify asymmetry"

    def run(self) -> dict[str, Any]:
        backends = self.config.sweep.get("vrf_backends", [self.config.vrf.backend])
        num_blocks = int(self.config.sweep.get("blocks", self.config.network.num_blocks))

        backend_results: list[dict[str, Any]] = []
        primary_rows: list[dict[str, Any]] = []
        primary_report = None

        for backend in backends:
            self.logger.info("profiling validation with vrf backend %s", backend)
            step_samples: dict[str, list[float]] = defaultdict(list)
            solve_samples: list[float] = []
            merkle_depths: list[int] = []
            proof_bytes: list[int] = []

            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle)
                vrf_kwargs = (
                    {"curve_name": self.config.vrf.curve} if backend == "ecdsa_pseudo" else {}
                )
                network = Network(
                    num_miners=self.config.network.num_miners,
                    repository=repository,
                    difficulty_bits=self.config.pow.difficulty_bits,
                    vrf_backend=backend,
                    seeds=bundle,
                    solver=self.config.solver.name,
                    hash_rates=self.config.network.hash_rates,
                    max_attempts=self.config.pow.max_attempts,
                    quality_threshold=self.config.solver.quality_threshold,
                    cost_tolerance=self.config.solver.cost_tolerance,
                    tie_break=self.config.network.tie_break,
                    vrf_kwargs=vrf_kwargs,
                )
                result = network.run(
                    num_blocks=num_blocks, log_every=self.config.run.log_every_blocks
                )
                for record in result.blocks:
                    solve_samples.append(record.solve_seconds)
                    for label, value in record.validation_steps.items():
                        step_samples[label].append(value)
                merkle_depths.append(repository.depth)
                proof_bytes.append(32 * repository.depth)

            report = compute_asymmetry(solve_samples, step_samples)
            rows = [
                {
                    "step_key": key,
                    "label": STEP_LABELS.get(key, key),
                    "seconds": report.step_mean_seconds[key],
                    "milliseconds": report.step_mean_milliseconds[key],
                    "microseconds": report.step_mean_milliseconds[key] * 1000.0,
                }
                for key in STEP_LABELS
                if key in report.step_mean_seconds
            ]
            solve_mean, solve_std = mean_std(solve_samples)
            entry = {
                "vrf_backend": backend,
                "steps": rows,
                "total_verify_ms": report.total_verify_milliseconds,
                "mean_solve_ms": solve_mean * 1000.0,
                "std_solve_ms": solve_std * 1000.0,
                "asymmetry_ratio": report.asymmetry_ratio,
                "num_validations": len(solve_samples) * self.config.network.num_miners,
                "merkle_depth": merkle_depths[0] if merkle_depths else 0,
                "merkle_proof_bytes": proof_bytes[0] if proof_bytes else 0,
            }
            backend_results.append(entry)
            self.logger.info(
                "%s -> total verify %.6f ms, solve %.3f ms, asymmetry %.0fx",
                backend,
                report.total_verify_milliseconds,
                solve_mean * 1000.0,
                report.asymmetry_ratio,
            )
            if backend == self.config.vrf.backend:
                primary_rows = rows
                primary_report = entry

        primary_report = primary_report or backend_results[0]
        primary_rows = primary_rows or primary_report["steps"]
        asymmetry_rows = self._asymmetry_vs_size()

        figure_paths = figures.asymmetry_figure(
            {row["label"].split(". ", 1)[-1]: row["milliseconds"] for row in primary_rows},
            primary_report["mean_solve_ms"],
            self.output_dir / "table6_validation_cost",
        )

        return {
            "primary_backend": primary_report["vrf_backend"],
            "table_rows": primary_rows,
            "total_verify_ms": primary_report["total_verify_ms"],
            "mean_solve_ms": primary_report["mean_solve_ms"],
            "asymmetry_ratio": primary_report["asymmetry_ratio"],
            "asymmetry_vs_size": asymmetry_rows,
            "backend_comparison": [
                {
                    "vrf_backend": entry["vrf_backend"],
                    "vrf_verify_ms": next(
                        (s["milliseconds"] for s in entry["steps"] if s["step_key"] == "vrf_verify"),
                        float("nan"),
                    ),
                    "total_verify_ms": entry["total_verify_ms"],
                    "asymmetry_ratio": entry["asymmetry_ratio"],
                }
                for entry in backend_results
            ],
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {
                "table6_validation_cost": primary_rows,
                # Omitted entirely rather than written empty when the sweep is disabled.
                **({"table6_asymmetry_vs_size": asymmetry_rows} if asymmetry_rows else {}),
                "table6_backend_comparison": [
                    {
                        "vrf_backend": entry["vrf_backend"],
                        "total_verify_ms": entry["total_verify_ms"],
                        "mean_solve_ms": entry["mean_solve_ms"],
                        "asymmetry_ratio": entry["asymmetry_ratio"],
                    }
                    for entry in backend_results
                ],
            },
            "paper_reference": {
                "table": "tab:validationcost",
                "reported_steps_ms": {
                    "1. PoW Check (Hash & Difficulty)": 0.000458,
                    "2. VRF Proof Verification": 0.001958,
                    "3. Merkle Proof Verification": 0.002435,
                    "4. Solution Feasibility & Cost Check": 0.000218,
                },
                "reported_total_ms": 0.005069,
                "reported_solve_ms": 785.08,
                "reported_asymmetry_ratio": 155000,
                "note": (
                    "The paper's 785.08 ms solve time matches neither its Table 3 "
                    "(1600.9 ms) nor its Table 4 (752.49 ms at N=10); see "
                    "PAPER_SPEC.md section 11-I7."
                ),
            },
        }

    def _asymmetry_vs_size(self) -> list[dict[str, Any]]:
        """Measure the solve/verify ratio at several instance sizes.

        Verification cost is nearly independent of N -- only the feasibility check
        scales, and it does so linearly -- while solving grows as (N-1)!.  The ratio
        is therefore a property of the chosen instance size, not of the protocol, and
        reporting it as a single number invites the reader to treat it as a constant.
        """
        sizes = [int(v) for v in self.config.sweep.get("asymmetry_sizes", [])]
        if not sizes:
            return []

        rows: list[dict[str, Any]] = []
        blocks = int(self.config.sweep.get("asymmetry_blocks", 3))
        for size in sizes:
            self.logger.info("asymmetry sweep at N=%d (%d blocks/seed)", size, blocks)
            solve_samples: list[float] = []
            verify_samples: list[float] = []
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                repository = self.build_repository(bundle, instance_size=size)
                network = Network(
                    num_miners=self.config.network.num_miners,
                    repository=repository,
                    difficulty_bits=self.config.pow.difficulty_bits,
                    vrf_backend=self.config.vrf.backend,
                    seeds=bundle,
                    solver=self.config.solver.name,
                    max_attempts=self.config.pow.max_attempts,
                    quality_threshold=self.config.solver.quality_threshold,
                    cost_tolerance=self.config.solver.cost_tolerance,
                    tie_break=self.config.network.tie_break,
                    vrf_kwargs=self.config.vrf.build_kwargs(),
                )
                result = network.run(num_blocks=blocks, log_every=0)
                for record in result.blocks:
                    solve_samples.append(record.solve_seconds)
                    verify_samples.append(sum(record.validation_steps.values()))

            solve_mean, solve_std = mean_std(solve_samples)
            verify_mean, verify_std = mean_std(verify_samples)
            rows.append(
                {
                    "size": size,
                    "mean_solve_ms": solve_mean * 1000.0,
                    "std_solve_ms": solve_std * 1000.0,
                    "mean_verify_ms": verify_mean * 1000.0,
                    "std_verify_ms": verify_std * 1000.0,
                    "asymmetry_ratio": solve_mean / verify_mean if verify_mean > 0 else float("inf"),
                    "num_blocks": len(solve_samples),
                }
            )
            self.logger.info(
                "N=%d -> solve %.3f ms, verify %.4f ms, asymmetry %.0fx",
                size,
                solve_mean * 1000.0,
                verify_mean * 1000.0,
                rows[-1]["asymmetry_ratio"],
            )
        return rows
