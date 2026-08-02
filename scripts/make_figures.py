#!/usr/bin/env python3
"""Regenerate figures from serialised results without rerunning experiments.

Each experiment already writes its own figures during ``execute()``.  This script
rebuilds them from the JSON payloads, which is useful after changing plot styling.

Usage::

    python scripts/make_figures.py
    python scripts/make_figures.py table5_fairness --output-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pouw.reporting import figures  # noqa: E402
from pouw.reporting.io import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _results(results_dir: Path, name: str) -> dict | None:
    path = results_dir / name / f"{name}.json"
    return read_json(path)["results"] if path.exists() else None


def time_allocation(results: dict, output: Path) -> list[Path]:
    return figures.time_allocation_figure(results["table_rows"], output / "table3_time_allocation")


def problem_size(results: dict, output: Path) -> list[Path]:
    return figures.problem_size_figure(results["table_rows"], output / "table4_problem_size")


def fairness(results: dict, output: Path) -> list[Path]:
    counts = {row["miner_id"]: row["blocks_won"] for row in results["table_rows"]}
    seed_counts = [
        {int(k): v for k, v in entry["counts_by_miner"].items()}
        for entry in results["per_seed"]
    ]
    return figures.fairness_figure(
        counts, sum(counts.values()), output / "table5_fairness", seed_counts=seed_counts
    )


def validation_cost(results: dict, output: Path) -> list[Path]:
    step_ms = {
        row["label"].split(". ", 1)[-1]: row["milliseconds"] for row in results["table_rows"]
    }
    return figures.asymmetry_figure(
        step_ms, results["mean_solve_ms"], output / "table6_validation_cost"
    )


def vrf_grinding(results: dict, output: Path) -> list[Path]:
    return figures.grinding_figure(
        results["attempt_budgets"], results["curves"], output / "ablation_vrf_grinding"
    )


def repository_size(results: dict, output: Path) -> list[Path]:
    rows = results["table_rows"]
    return figures.sensitivity_figure(
        [row["repository_size"] for row in rows],
        {"Merkle proof size (bytes)": [row["merkle_proof_bytes"] for row in rows]},
        x_label="Repository size $N$",
        y_label="Merkle proof size (bytes)",
        path=output / "sensitivity_repository_size",
        log_x=True,
    )


def solver_advantage(results: dict, output: Path) -> list[Path]:
    rows = results["table_rows"]
    return figures.sensitivity_figure(
        [row["size"] for row in rows],
        {
            "Brute force ($(N-1)!$)": [row["brute_force_seconds"] for row in rows],
            "Held-Karp ($2^N N^2$)": [row["held_karp_seconds"] for row in rows],
        },
        x_label="Instance size $N$",
        y_label="Mean solve time (s, log scale)",
        path=output / "sensitivity_solver_advantage",
        log_y=True,
    )


def hash_rate(results: dict, output: Path) -> list[Path]:
    rows = results["table_rows"]
    return figures.sensitivity_figure(
        list(range(len(rows))),
        {"Max deviation from power share": [row["max_abs_deviation_mean"] for row in rows]},
        x_label="Hash-rate distribution (index)",
        y_label="Max |observed - expected| block share",
        path=output / "sensitivity_hash_rate",
        errors={"Max deviation from power share": [row["max_abs_deviation_std"] for row in rows]},
    )


BUILDERS = {
    "table3_time_allocation": time_allocation,
    "table4_problem_size": problem_size,
    "table5_fairness": fairness,
    "table6_validation_cost": validation_cost,
    "ablation_vrf_grinding": vrf_grinding,
    "sensitivity_repository_size": repository_size,
    "sensitivity_solver_advantage": solver_advantage,
    "sensitivity_hash_rate": hash_rate,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate figures from results")
    parser.add_argument("experiments", nargs="*", help="defaults to all available")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    names = args.experiments or list(BUILDERS)

    written: list[Path] = []
    for name in names:
        if name not in BUILDERS:
            print(f"skipping {name}: no figure builder", file=sys.stderr)
            continue
        results = _results(results_dir, name)
        if results is None:
            print(f"skipping {name}: no results at {results_dir / name}")
            continue
        try:
            written.extend(BUILDERS[name](results, output_dir))
        except (KeyError, IndexError) as error:
            print(f"skipping {name}: results incomplete ({error})", file=sys.stderr)

    for path in written:
        print(f"wrote {path}")
    if not written:
        print("\nNo figures written. Run experiments first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
