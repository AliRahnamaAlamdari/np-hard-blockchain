#!/usr/bin/env python3
"""Emit LaTeX table code from serialised experiment results.

Reads ``results/<experiment>/<experiment>.json`` and writes paste-ready LaTeX to
``results/latex/``.  Nothing here modifies ``sn-article.tex``.

Usage::

    python scripts/make_tables.py                 # every experiment with results
    python scripts/make_tables.py table5_fairness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pouw.reporting import latex  # noqa: E402
from pouw.reporting.io import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _results(results_dir: Path, name: str) -> dict | None:
    path = results_dir / name / f"{name}.json"
    if not path.exists():
        return None
    return read_json(path)["results"]


def time_allocation(results: dict) -> str:
    return latex.time_allocation_table(results["table_rows"])


def problem_size(results: dict) -> str:
    return latex.problem_size_table(results["table_rows"])


def fairness(results: dict) -> str:
    counts = {row["miner_id"]: row["blocks_won"] for row in results["table_rows"]}
    total = sum(counts.values())
    main = latex.fairness_table(counts, total)
    summary = latex.fairness_summary_table(results["summary_rows"])
    return f"{main}\n\n{summary}"


def validation_cost(results: dict) -> str:
    return latex.validation_cost_table(
        results["table_rows"],
        results["total_verify_ms"],
        results["mean_solve_ms"],
        results["asymmetry_ratio"],
    )


def vrf_grinding(results: dict) -> str:
    rows = [
        [
            row["vrf_backend"].replace("_", r"\_"),
            "Yes" if row["unique_output"] else "No",
            f"{row['distinct_tasks_1_eval']:.0f}",
            f"{row['distinct_tasks_max_eval']:.0f}",
            f"{row['hashes_per_distinct_task']:,.0f}",
        ]
        for row in results["table_rows"]
    ]
    return latex.generic_table(
        headers=[
            "VRF backend",
            "Unique output",
            "Tasks (1 eval.)",
            f"Tasks ({results['attempt_budgets'][-1]} evals.)",
            "Hashes per distinct task",
        ],
        rows=rows,
        caption=(
            "Task grinding resistance by VRF primitive. A unique-output VRF admits one "
            "task per paid PoW seed; a malleable signature scheme admits many, which "
            "invalidates the $k/p$ resampling bound."
        ),
        label="tab:vrfgrinding",
    )


def component_ablation(results: dict) -> str:
    rows = [
        [
            row["component_removed"],
            row["attack"],
            "Succeeds" if row["attack_succeeds_without_component"] else "Fails",
            "Blocked" if row["blocked_when_component_present"] else "Not blocked",
        ]
        for row in results["summary"]
        if row["variant"] != "full"
    ]
    return latex.generic_table(
        headers=["Component removed", "Attack", "Without component", "With component"],
        rows=rows,
        caption=(
            "Component ablation. Each row states whether the attack the component is "
            "claimed to mitigate actually succeeds when that component is disabled."
        ),
        label="tab:componentablation",
        column_spec="p{0.20\\linewidth}p{0.34\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}",
        star=True,
    )


def repository_size(results: dict) -> str:
    rows = [
        [
            str(row["repository_size"]),
            f"{row['uniformity_p_mean']:.3f}",
            str(row["merkle_depth"]),
            str(row["merkle_proof_bytes"]),
            f"{row['merkle_verify_ms']:.4f}",
        ]
        for row in results["table_rows"]
    ]
    return latex.generic_table(
        headers=[
            "Repository size $N$",
            "Uniformity $p$",
            "Merkle depth",
            "Proof size (B)",
            "Verify (ms)",
        ],
        rows=rows,
        caption=(
            "Sensitivity to repository size $N$. Task selection remains uniform "
            "($\\chi^2$ goodness-of-fit) while proof cost grows logarithmically."
        ),
        label="tab:repositorysize",
    )


def hash_rate(results: dict) -> str:
    rows = [
        [
            row["distribution"],
            ", ".join(f"{rate:g}" for rate in row["hash_rates"]),
            f"{row['max_abs_deviation_mean']:.4f} $\\pm$ {row['max_abs_deviation_std']:.4f}",
            f"{row['jain_index_mean']:.4f}",
            "Yes" if row["power_proportional"] else "No",
        ]
        for row in results["table_rows"]
    ]
    return latex.generic_table(
        headers=[
            "Distribution",
            "Hash rates",
            "Max deviation from power share",
            "Jain index",
            "Proportional",
        ],
        rows=rows,
        caption=(
            "Fairness under heterogeneous hash rates. Block share should track "
            "hash-rate share; Jain's index alone does not test this."
        ),
        label="tab:hashratesensitivity",
        star=True,
    )


def solver_advantage(results: dict) -> str:
    rows = [
        [
            str(row["size"]),
            f"{row['brute_force_seconds']:.6g}",
            f"{row['held_karp_seconds']:.6g}",
            f"{row['speedup']:.1f}$\\times$",
            "Yes" if row["same_optimum"] else "No",
        ]
        for row in results["table_rows"]
    ]
    return latex.generic_table(
        headers=[
            "Size $N$",
            "Brute force (s)",
            "Held-Karp (s)",
            "Speedup",
            "Identical optimum",
        ],
        rows=rows,
        caption=(
            "Mining advantage from a better exact solver on identical assigned "
            "instances. Both solvers return the same optimum, so the faster miner "
            "produces fully valid blocks at a fraction of the cost."
        ),
        label="tab:solveradvantage",
    )


def convergence(results: dict) -> str:
    summary = results["summary"]
    rows = [
        ["Seeds", str(summary["num_seeds"])],
        ["Total blocks", str(summary["total_blocks"])],
        ["All nodes converged", "Yes" if summary["all_converged"] else "No"],
        [
            "Block acceptance rate",
            f"{summary['acceptance_rate_mean']:.4f} $\\pm$ {summary['acceptance_rate_std']:.4f}",
        ],
    ]
    if "optimality_rate_mean" in summary:
        rows.append(
            [
                "Solutions verified optimal",
                f"{summary['optimality_rate_mean']:.4f} $\\pm$ {summary['optimality_rate_std']:.4f}",
            ]
        )
    return latex.generic_table(
        headers=["Quantity", "Value"],
        rows=rows,
        caption=(
            "Chain convergence and useful-work correctness. Optimality is checked "
            "against an independent exact algorithm rather than assumed."
        ),
        label="tab:convergence",
    )


BUILDERS = {
    "table3_time_allocation": time_allocation,
    "table4_problem_size": problem_size,
    "table5_fairness": fairness,
    "table6_validation_cost": validation_cost,
    "convergence": convergence,
    "ablation_vrf_grinding": vrf_grinding,
    "ablation_components": component_ablation,
    "sensitivity_repository_size": repository_size,
    "sensitivity_hash_rate": hash_rate,
    "sensitivity_solver_advantage": solver_advantage,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit LaTeX tables from results")
    parser.add_argument("experiments", nargs="*", help="defaults to all available")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/latex")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    output_dir = ROOT / args.output_dir
    names = args.experiments or list(BUILDERS)

    written = 0
    for name in names:
        if name not in BUILDERS:
            print(f"skipping {name}: no LaTeX builder", file=sys.stderr)
            continue
        results = _results(results_dir, name)
        if results is None:
            print(f"skipping {name}: no results at {results_dir / name}")
            continue
        try:
            content = BUILDERS[name](results)
        except (KeyError, IndexError) as error:
            print(f"skipping {name}: results incomplete ({error})", file=sys.stderr)
            continue
        path = latex.write_table(output_dir / f"{name}.tex", content)
        print(f"wrote {path}")
        written += 1

    if written == 0:
        print("\nNo tables written. Run experiments first, e.g.:")
        print("  python scripts/run_experiment.py table5_fairness")
        return 1
    print(f"\n{written} table file(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
