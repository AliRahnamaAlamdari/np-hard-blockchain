"""Reproduces the paper's Table 4 (``tab:problemsize``) across every solver backend.

Measures useful-work solve time against instance size ``N`` at fixed PoW difficulty,
averaged over ``run.repetitions`` runs per size -- the paper's stated protocol is 50
runs (L408).

Three things the paper does not report are added, because each is a question a
reader is entitled to ask of the reported shares:

* the standard deviation the paper discards after computing its means;
* the exact tour count ``(N-1)!`` and the measured per-tour cost, which is what makes
  the ``(N-1)!`` scaling claim checkable rather than asserted;
* every measurement repeated per solver backend.  The paper's exhaustive solver caps
  the reachable size at ``N=12``; the exact dynamic program returns the *same*
  optimum in ``O(2^N N^2)`` and reaches ``N=20``, and the heuristic reaches several
  hundred.  Reporting one solver's curve alone makes the ``(N-1)!`` growth look like
  a property of the problem when it is a property of the chosen solver.

Each repetition runs the whole block path -- Stage-1 search, VRF proof, Merkle proof
retrieval, solve, packaging, and one validator's execution of Algorithm 2 -- so the
row carries the cryptographic overhead terms as well as the two the paper reports.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..crypto.hashing import difficulty_to_target
from ..crypto.vrf import build_vrf
from ..chain.miner import Miner
from ..chain.validator import Validator
from ..metrics.scaling import FitError, crossover_size, fit_growth, fit_linear
from ..metrics.timing import compute_shares
from ..reporting import figures
from ..seeding import seed_everything
from ..tasks.tsp import count_permutations, held_karp
from .base import Experiment, mean_std

# Held-Karp is only run as a comparison baseline up to a size where it is affordable.
DEFAULT_HELD_KARP_MAX_SIZE = 20

# Above this size (N-1)! no longer fits a float, so only its logarithm is reported.
EXACT_TOUR_COUNT_MAX_SIZE = 25


def log10_tours(size: int) -> float:
    """``log10((N-1)!)`` -- the tour count the exhaustive solver would enumerate.

    Reported for every size, including the heuristic's, because it is the quantity
    that makes the heuristic's reach legible: at N=800 the enumeration a
    brute-force solver would face has more than 1900 digits.
    """
    return math.lgamma(size) / math.log(10.0)


def _fit_tables(fits: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """CSV tables for the growth fits, absent when nothing could be fitted."""
    if not fits:
        return {}
    return {
        "table4_growth_fits": [
            {
                "solver": fit.solver,
                "growth_label": fit.growth_label,
                "coefficient": fit.coefficient,
                "exponent": fit.exponent,
                "r_squared": fit.r_squared,
                "max_abs_percent_error": fit.max_abs_percent_error,
                "sizes_fitted": " ".join(str(s) for s in fit.sizes_fitted),
                "sizes_excluded": " ".join(str(s) for s in fit.sizes_excluded),
            }
            for fit in fits.values()
        ],
        "table4_growth_fit_residuals": [
            {"solver": fit.solver, "growth_label": fit.growth_label, **residual.to_dict()}
            for fit in fits.values()
            for residual in fit.residuals
        ],
    }


class ProblemSizeExperiment(Experiment):
    name = "table4_problem_size"
    description = "Useful-work solve time and computation share against TSP size N"

    # -- sweep resolution ----------------------------------------------------

    def _resolve_sweeps(self) -> list[dict[str, Any]]:
        """Normalise both config shapes into one list of per-solver sweep specs.

        The legacy shape (``instance_sizes`` plus the config's single
        ``solver.name``) is what the paper's own configs use and is preserved
        exactly, so those runs remain reproducible.  The extended shape
        (``solver_sweeps``) gives each backend its own size list and repetition
        budget, because the affordable range differs by orders of magnitude.
        """
        declared = self.config.sweep.get("solver_sweeps")
        if declared:
            sweeps = []
            for entry in declared:
                sweeps.append(
                    {
                        "solver": str(entry.get("solver", self.config.solver.name)),
                        "instance_sizes": [int(s) for s in entry["instance_sizes"]],
                        "repetitions": int(
                            entry.get("repetitions", self.config.run.repetitions)
                        ),
                        "repetitions_by_size": {
                            int(k): int(v)
                            for k, v in (entry.get("repetitions_by_size") or {}).items()
                        },
                    }
                )
            return sweeps
        return [
            {
                "solver": self.config.solver.name,
                "instance_sizes": [
                    int(s) for s in self.config.sweep.get("instance_sizes", [4, 6, 8, 10])
                ],
                "repetitions": int(self.config.run.repetitions),
                "repetitions_by_size": {},
            }
        ]

    # -- measurement ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        sweeps = self._resolve_sweeps()
        compare_held_karp = bool(self.config.sweep.get("compare_held_karp", True))
        held_karp_max = int(
            self.config.sweep.get("held_karp_max_size", DEFAULT_HELD_KARP_MAX_SIZE)
        )
        difficulty_bits = self.config.pow.difficulty_bits
        target = difficulty_to_target(difficulty_bits)

        rows: list[dict[str, Any]] = []
        per_run_rows: list[dict[str, Any]] = []

        for sweep in sweeps:
            solver = sweep["solver"]
            for size in sweep["instance_sizes"]:
                repetitions = sweep["repetitions_by_size"].get(
                    size, sweep["repetitions"]
                )
                self.logger.info(
                    "solver=%s N=%d: %d repetitions x %d seeds "
                    "((N-1)! = 10^%.1f tours)",
                    solver,
                    size,
                    repetitions,
                    len(self.config.run.seeds),
                    log10_tours(size),
                )
                row, runs = self._measure_cell(
                    solver=solver,
                    size=size,
                    repetitions=repetitions,
                    target=target,
                    compare_held_karp=(
                        compare_held_karp
                        and size <= held_karp_max
                        # Comparing Held-Karp against itself would double the
                        # cost of the most expensive cells and measure nothing.
                        and solver != "held_karp"
                    ),
                )
                rows.append(row)
                per_run_rows.extend(runs)

        growth = self._scaling_check(rows)
        fits = self._growth_fits(rows)
        verify_fit = self._verification_fit(rows)
        crossovers = self._verification_crossovers(rows, fits, verify_fit)
        figure_rows = [row for row in rows if row["solver"] == sweeps[0]["solver"]]
        figure_paths = figures.problem_size_figure(
            figure_rows or rows, self.output_dir / "table4_problem_size"
        )
        return {
            "table_rows": rows,
            "per_run_rows": per_run_rows,
            "scaling_check": growth,
            "growth_fits": [fit.to_dict() for fit in fits.values()],
            "verification_fit": verify_fit.to_dict() if verify_fit else None,
            "verification_crossovers": [c.to_dict() for c in crossovers],
            "figures": [str(path) for path in figure_paths],
            "csv_tables": {
                "table4_problem_size": rows,
                "table4_per_run": per_run_rows,
                # Omitted entirely rather than written empty when every measured
                # time sits below the fit floor, as it does on toy configs.
                **_fit_tables(fits),
                # The crossover needs a growth fit, so it shares the fit's fate.
                **(
                    {"table4_verification_crossover": [c.to_dict() for c in crossovers]}
                    if crossovers
                    else {}
                ),
            },
            "paper_reference": {
                "table": "tab:problemsize",
                "reported": [
                    {"instance_id": "I-4-A", "size": 4, "pow_s": 0.00143, "useful_s": 0.00002, "share_pct": 1.59},
                    {"instance_id": "I-6-A", "size": 6, "pow_s": 0.00162, "useful_s": 0.00022, "share_pct": 12.09},
                    {"instance_id": "I-8-A", "size": 8, "pow_s": 0.00406, "useful_s": 0.00871, "share_pct": 68.20},
                    {"instance_id": "I-10-A", "size": 10, "pow_s": 0.00129, "useful_s": 0.75249, "share_pct": 99.83},
                    {"instance_id": "I-12-A", "size": 12, "pow_s": 0.00106, "useful_s": 99.71407, "share_pct": 100.00},
                ],
            },
        }

    def _measure_cell(
        self,
        solver: str,
        size: int,
        repetitions: int,
        target: int,
        compare_held_karp: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one (solver, size) cell over every seed and aggregate it."""
        pow_times: list[float] = []
        solve_times: list[float] = []
        miner_overhead: list[float] = []
        validation_times: list[float] = []
        held_karp_times: list[float] = []
        operations: list[int] = []
        costs: list[float] = []
        optimal_agreement = True
        max_relative_gap = 0.0
        all_accepted = True
        per_run_rows: list[dict[str, Any]] = []

        for seed in self.config.run.seeds:
            bundle = seed_everything(seed)
            repository = self.build_repository(bundle, instance_size=size)
            vrf = build_vrf(self.config.vrf.backend, **self.config.vrf.build_kwargs())
            miner = Miner(
                miner_id=0,
                keys=vrf.keygen(bundle.bytes_for("miner:0")),
                vrf=vrf,
            )
            validator = Validator(
                validator_id=0,
                vrf=vrf,
                repository=repository,
                target=target,
                quality_threshold=self.config.solver.quality_threshold,
                cost_tolerance=self.config.solver.cost_tolerance,
            )

            for repetition in range(repetitions):
                block, timings = miner.mine(
                    height=repetition + 1,
                    previous_hash=f"parent-{seed}-{repetition}",
                    tx_root=f"tx-{seed}-{repetition}",
                    target=target,
                    repository=repository,
                    solver=solver,
                    max_attempts=self.config.pow.max_attempts,
                    quality_threshold=self.config.solver.quality_threshold,
                    cost_tolerance=self.config.solver.cost_tolerance,
                )
                validation = validator.validate(block)
                all_accepted = all_accepted and validation.accepted

                overhead = (
                    timings["vrf_prove_s"]
                    + timings["merkle_proof_s"]
                    + timings["package_s"]
                )
                pow_times.append(timings["pow_s"])
                solve_times.append(timings["solve_s"])
                miner_overhead.append(overhead)
                validation_times.append(validation.total_seconds)
                operations.append(timings["solver_operations"])
                costs.append(block.solution.cost)

                relative_gap = 0.0
                if compare_held_karp:
                    start = time.perf_counter()
                    hk_solution, _ = held_karp(block.instance)
                    held_karp_times.append(time.perf_counter() - start)
                    if hk_solution.cost > 0:
                        relative_gap = (
                            block.solution.cost - hk_solution.cost
                        ) / hk_solution.cost
                    max_relative_gap = max(max_relative_gap, relative_gap)
                    if abs(hk_solution.cost - block.solution.cost) > 1e-9:
                        optimal_agreement = False

                per_run_rows.append(
                    {
                        "solver": solver,
                        "size": size,
                        "seed": seed,
                        "repetition": repetition,
                        "instance_id": block.instance.instance_id,
                        "pow_seconds": timings["pow_s"],
                        "pow_attempts": timings["pow_attempts"],
                        "solve_seconds": timings["solve_s"],
                        "vrf_prove_seconds": timings["vrf_prove_s"],
                        "merkle_proof_seconds": timings["merkle_proof_s"],
                        "package_seconds": timings["package_s"],
                        "validation_seconds": validation.total_seconds,
                        "solver_operations": timings["solver_operations"],
                        "tour_cost": block.solution.cost,
                        "relative_gap_to_optimum": relative_gap,
                        "accepted": validation.accepted,
                    }
                )

        pow_mean, pow_std = mean_std(pow_times)
        solve_mean, solve_std = mean_std(solve_times)
        network_overhead = [
            miner + validate
            for miner, validate in zip(miner_overhead, validation_times)
        ]
        shares = compute_shares(pow_times, solve_times, network_overhead)
        miner_shares = compute_shares(pow_times, solve_times, miner_overhead)
        # (N-1)! has ~1900 digits at N=800, so the exact count is carried only
        # where it is a usable number and its logarithm always.
        tours = count_permutations(size) if size <= EXACT_TOUR_COUNT_MAX_SIZE else None

        row: dict[str, Any] = {
            "solver": solver,
            "instance_id": f"I-{size}-A",
            "size": size,
            "num_runs": len(solve_times),
            "mean_pow_seconds": pow_mean,
            "std_pow_seconds": pow_std,
            "mean_solve_seconds": solve_mean,
            "std_solve_seconds": solve_std,
            "mean_miner_overhead_seconds": miner_shares.mean_overhead_seconds,
            "mean_validation_seconds": sum(validation_times) / len(validation_times),
            "useful_share_percent": shares.useful_share_percent,
            "useful_share_percent_with_miner_overhead": (
                miner_shares.useful_share_percent_with_overhead
            ),
            "useful_share_percent_with_overhead": (
                shares.useful_share_percent_with_overhead
            ),
            "tours_enumerated": tours,
            "log10_tours_enumerated": log10_tours(size),
            "mean_solver_operations": sum(operations) / len(operations),
            "microseconds_per_tour": solve_mean / tours * 1e6 if tours else None,
            "mean_tour_cost": sum(costs) / len(costs),
            "all_accepted": all_accepted,
        }
        if compare_held_karp and held_karp_times:
            hk_mean, hk_std = mean_std(held_karp_times)
            row.update(
                {
                    "mean_heldkarp_seconds": hk_mean,
                    "std_heldkarp_seconds": hk_std,
                    "heldkarp_speedup": solve_mean / hk_mean if hk_mean > 0 else float("inf"),
                    "heldkarp_matches_optimum": optimal_agreement,
                    "max_relative_gap_to_optimum": max_relative_gap,
                    "heldkarp_operations_bound": (2**size) * size * size,
                }
            )
        self.logger.info(
            "solver=%s N=%d -> solve %.6g s +/- %.2g, share %.2f%% "
            "(with miner overhead %.2f%%, network-wide %.2f%%)%s",
            solver,
            size,
            solve_mean,
            solve_std,
            row["useful_share_percent"],
            row["useful_share_percent_with_miner_overhead"],
            row["useful_share_percent_with_overhead"],
            f", Held-Karp {row['heldkarp_speedup']:.1f}x faster"
            if "heldkarp_speedup" in row
            else "",
        )
        return row, per_run_rows

    # -- analysis ------------------------------------------------------------

    def _scaling_check(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compare measured time ratios against the ``(N-1)!`` tour-count ratios."""
        checks = []
        for solver in dict.fromkeys(row["solver"] for row in rows):
            series = [row for row in rows if row["solver"] == solver]
            for previous, current in zip(series, series[1:]):
                log_tour_ratio = log10_tours(current["size"]) - log10_tours(
                    previous["size"]
                )
                tour_ratio = (
                    10.0**log_tour_ratio if log_tour_ratio < 300.0 else float("inf")
                )
                time_ratio = (
                    current["mean_solve_seconds"] / previous["mean_solve_seconds"]
                    if previous["mean_solve_seconds"] > 0
                    else float("inf")
                )
                checks.append(
                    {
                        "solver": solver,
                        "from_size": previous["size"],
                        "to_size": current["size"],
                        "tour_count_ratio": tour_ratio,
                        "log10_tour_count_ratio": log_tour_ratio,
                        "measured_time_ratio": time_ratio,
                        "relative_deviation": (time_ratio - tour_ratio) / tour_ratio
                        if math.isfinite(tour_ratio) and tour_ratio > 0
                        else float("nan"),
                    }
                )
        return checks

    def _growth_fits(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit each solver's measured times against its own theoretical growth."""
        floor = float(self.config.sweep.get("fit_min_seconds", 1.0e-3))
        fits: dict[str, Any] = {}
        for solver in dict.fromkeys(row["solver"] for row in rows):
            series = sorted(
                (row for row in rows if row["solver"] == solver),
                key=lambda row: row["size"],
            )
            try:
                fit = fit_growth(
                    solver,
                    [row["size"] for row in series],
                    [row["mean_solve_seconds"] for row in series],
                    min_seconds=floor,
                )
            except FitError as error:
                self.logger.warning("no growth fit for %s: %s", solver, error)
                continue
            fits[solver] = fit
            self.logger.info(
                "%s: t = %.3g * (%s)^%.4f, R2=%.6f, max residual %.1f%%",
                solver,
                fit.coefficient,
                fit.growth_label,
                fit.exponent,
                fit.r_squared,
                fit.max_abs_percent_error,
            )
        return fits

    def _verification_fit(self, rows: list[dict[str, Any]]) -> Any:
        """Model validator-side cost against instance size, pooled over solvers.

        Validation is the same work whichever solver produced the witness, so every
        row contributes.  The intercept is the size-independent part -- dominated by
        the VRF check -- and the slope is what the feasibility and membership checks
        add per node, the term the paper's "verification is independent of problem
        size" claim asserts to be negligible.
        """
        override = self.config.sweep.get("verify_seconds")
        if override is not None:
            return None
        pooled: dict[int, list[float]] = {}
        for row in rows:
            pooled.setdefault(row["size"], []).append(row["mean_validation_seconds"])
        sizes = sorted(pooled)
        means = [sum(pooled[size]) / len(pooled[size]) for size in sizes]
        try:
            fit = fit_linear(sizes, means)
        except FitError as error:
            self.logger.warning("no verification fit: %s", error)
            return None
        if fit.slope_is_resolved:
            self.logger.info(
                "verification: v(N) = %.4f ms + %.6f ms/node * N, R2=%.4f over N=%s",
                fit.intercept * 1000.0,
                fit.slope * 1000.0,
                fit.r_squared,
                sizes,
            )
        else:
            self.logger.info(
                "verification: flat at %.4f ms over N=%d..%d; the linear term is "
                "unresolved (slope %.2e ms/node, R2=%.3f), so validation cost is "
                "independent of instance size across the measured range",
                fit.mean_seconds * 1000.0,
                sizes[0],
                sizes[-1],
                fit.slope * 1000.0,
                fit.r_squared,
            )
        return fit

    def _verification_crossovers(
        self, rows: list[dict[str, Any]], fits: dict[str, Any], verify_fit: Any
    ) -> list[Any]:
        """Per-solver instance size at which solving overtakes verification.

        Both sides come from this same experiment -- same host, same VRF backend,
        same run -- so the crossover is internally consistent and needs no figure
        imported from another table.
        """
        override = self.config.sweep.get("verify_seconds")
        verify: Any
        if override is not None:
            verify = float(override)
        elif verify_fit is not None:
            verify = verify_fit
        else:
            samples = [row["mean_validation_seconds"] for row in rows]
            verify = sum(samples) / len(samples) if samples else 0.0

        crossovers = []
        for solver, fit in fits.items():
            series = sorted(
                (row for row in rows if row["solver"] == solver),
                key=lambda row: row["size"],
            )
            crossover = crossover_size(
                fit,
                verify,
                sizes=[row["size"] for row in series],
                mean_solve_seconds=[row["mean_solve_seconds"] for row in series],
                mean_verify_seconds=[
                    row["mean_validation_seconds"] for row in series
                ],
            )
            crossovers.append(crossover)
            self.logger.info(
                "%s: solving overtakes verification at N=%s measured, N=%s fitted "
                "(verification there %.4f ms)",
                solver,
                crossover.measured_size,
                crossover.fitted_size,
                crossover.verify_seconds_at_crossover * 1000.0,
            )
        return crossovers
