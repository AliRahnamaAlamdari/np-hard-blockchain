#!/usr/bin/env python3
"""Build the manuscript's figures from the serialised results.

Unlike ``make_figures.py``, which redraws each experiment's own diagnostic plot,
this script produces the figures that appear in the paper, sized to the journal's
print widths so they are included without scaling:

    Fig1.pdf  framework flowchart (vector redraw)              full width
    Fig2.pdf  reseeding: favoured rate vs K, bias vs d         full width
    Fig3.pdf  useful-work share vs gate difficulty             one column
    Fig4.pdf  solve and validation time vs instance size       one column

Inputs (run the experiments first; see run_all.sh):

    results/reseeding_bias/reseeding_bias.json
    results/time_allocation_sweep/time_allocation_sweep.json
    results/table4_problem_size/table4_problem_size.json
    results/table6_validation_cost/table6_validation_cost.json

Usage::

    python scripts/make_paper_figures.py                       # writes into wiley-submission/
    python scripts/make_paper_figures.py --output-dir results/figures/paper
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from pouw.reporting.figures import PALETTE  # noqa: E402
from pouw.reporting.io import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Print widths of the Wiley USG two-column layout: \columnwidth = 244.7 pt and
# \textwidth = 506.5 pt.  Figures are drawn at these sizes and included at
# width=\columnwidth or \textwidth, so the 8.5 pt text below prints at 8.5 pt --
# the size of the template's caption font.
CM = 1 / 2.54
SINGLE = 8.6 * CM
DOUBLE = 17.8 * CM
FONT_PT = 8.5

SOLVERS = [
    # key in results, legend label, colour, marker, growth model g(N) used by the fit
    ("brute_force", "Exhaustive", PALETTE["pow"], "o", lambda n: math.gamma(n)),
    ("held_karp", "Held–Karp", PALETTE["useful"], "s", lambda n: 2.0**n * n * n),
    ("nearest_neighbor_2opt", "2-opt", PALETTE["overhead"], "^", lambda n: n * n),
]


def _style() -> None:
    plt.rcParams.update(
        {
            # Type 42 (TrueType) embedding; journal preflight rejects Type 3 fonts.
            "pdf.fonttype": 42,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": FONT_PT,
            "axes.labelsize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "standard",
        }
    )


def _results(results_dir: Path, name: str) -> dict:
    path = results_dir / name / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run: python scripts/run_experiment.py {name}")
    return read_json(path)["results"]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- Fig. 1
def framework_figure(path: Path) -> Path:
    """Vector redraw of the two-stage flowchart (boxes, arrows and labels only)."""
    fig = plt.figure(figsize=(DOUBLE, 7.2 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 178)
    ax.set_ylim(0, 72)
    ax.axis("off")

    fill = "#E6E6E6"

    def box(x, y, w, h, title, body=None, face=fill, lw=0.8, dashed=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="square,pad=0", facecolor=face, edgecolor="black",
                linewidth=lw, linestyle="--" if dashed else "-",
            )
        )
        if body is None:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontweight="bold")
        else:
            ax.text(x + w / 2, y + h - 3, title, ha="center", va="top", fontweight="bold",
                    linespacing=1.15)
            ax.text(x + w / 2, y + 4, body, ha="center", va="bottom", linespacing=1.15)

    def arrow(start, end, dashed=False, label=None, label_xy=None):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                color="black", linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0,
            )
        )
        if label:
            ax.text(*label_xy, label, ha="center", va="bottom", fontsize=FONT_PT)

    row_y, row_h, w = 15, 30, 25
    xs = {"s1": 4, "s2": 41, "solve": 87, "valid": 118, "cons": 149}

    box(xs["s1"], row_y, w, row_h, "Stage 1:\nlightweight\nPoW gate", "Generates entropy,\ncontrols spam")
    box(xs["s2"], row_y, 31, row_h, "Stage 2:\nVRF-based task\nassignment", "Verifiable random\ntask selection")
    box(xs["solve"], row_y, w, row_h, "Useful-work\nsolving", "Miner solves the\nassigned instance")
    box(xs["valid"], row_y, w, row_h, "Validation and\nblock production", "Validators verify\nall four proofs")
    box(xs["cons"], row_y, w, row_h, "Consensus and\ninclusion", "Block added\nto the ledger")

    box(26, 49, 17, 11, "Miner\nprivate key", face="white")
    box(41, 1.5, 31, 9, "Merkle task repository", face="white")
    box(66, 58, 46, 11, "Adjustable system settings", face="white", dashed=True)

    ax.text(xs["s1"] + w / 2, row_y - 2.5, "(difficulty $d$ tunable)", ha="center", va="top")
    ax.text(xs["solve"] + w / 2, row_y - 2.5, "(task parameters\ntunable)", ha="center", va="top")

    mid = row_y + row_h / 2
    arrow((xs["s1"] + w, mid), (xs["s2"], mid), label="seed $s$", label_xy=((xs["s1"] + w + xs["s2"]) / 2, mid + 1))
    arrow((xs["s2"] + 31, mid), (xs["solve"], mid), label="instance $I_i$", label_xy=((xs["s2"] + 31 + xs["solve"]) / 2, mid + 1))
    arrow((xs["solve"] + w, mid), (xs["valid"], mid))
    arrow((xs["valid"] + w, mid), (xs["cons"], mid))
    ax.plot([34.5, 34.5], [49, 40], color="black", linewidth=0.8)
    arrow((34.5, 40), (xs["s2"], 40))
    arrow((56.5, 10.5), (56.5, row_y))
    ax.plot([66, xs["s1"] + w / 2], [66, 66], color="black", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((xs["s1"] + w / 2, 66), (xs["s1"] + w / 2, row_y + row_h),
                                 arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 linestyle=(0, (3, 2)), color="black", shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((xs["solve"] + w / 2, 58), (xs["solve"] + w / 2, row_y + row_h),
                                 arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 linestyle=(0, (3, 2)), color="black", shrinkA=0, shrinkB=0))
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 2
def reseeding_figure(reseeding: dict, sweep: dict, path: Path) -> tuple[Path, float, float]:
    phi = float(reseeding["favoured_fraction"])
    economics = reseeding["economics"]
    d_star = float(economics["safe_target_bits_d_star"])
    # In-protocol hash rate: all race attempts over race wall time, pooled across the sweep.
    rows = sweep["table_rows"]
    pow_total = sum(row["mean_pow_seconds"] for row in rows)
    in_protocol_rate = sum(row["hash_rate_hps"] * row["mean_pow_seconds"] for row in rows) / pow_total
    d_star_protocol = math.log2(phi * economics["saving_seconds"] * in_protocol_rate)

    fig, (left, right) = plt.subplots(1, 2, figsize=(DOUBLE, 5.6 * CM))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.2, top=0.96, wspace=0.28)

    budget = reseeding["rate_vs_budget"]
    k = [row["reseed_budget"] for row in budget]
    smooth = [i / 10 for i in range(0, 101)]
    left.plot(smooth, [1 - (1 - phi) ** (x + 1) for x in smooth], color=PALETTE["reference"],
              linestyle="-", label=r"$1-(1-\phi)^{K+1}$")
    left.errorbar(k, [row["achieved_favoured_rate"] for row in budget],
                  yerr=[row["achieved_std"] for row in budget], fmt="o", color=PALETTE["pow"],
                  capsize=2, elinewidth=0.7, label="Measured (mean $\\pm$ s.d.)")
    left.set_xlabel("Reseed budget $K$ (extra seeds)")
    left.set_ylabel("Favored-task rate")
    left.set_ylim(0, 1.02)
    left.legend(loc="lower right")
    left.text(0.02, 0.97, "(a)", transform=left.transAxes, va="top", fontweight="bold")

    bias = reseeding["bias_vs_target"]
    d = [row["difficulty_bits"] for row in bias]
    right.plot(d, [row["assignment_bias"] for row in bias], color=PALETTE["accent"],
               marker="s", linestyle="-", label="Rational adversary")
    right.axvline(d_star, color="black", linestyle="--", linewidth=0.9, label="$d^*$, raw hash rate")
    right.axvline(d_star_protocol, color="black", linestyle=":", linewidth=1.1,
                  label="$d^*$, in-protocol rate")
    right.text(d_star + 0.5, 0.4, f"$d^*\\approx{d_star:.1f}$\n(raw hash rate)", ha="left", va="center")
    right.text(d_star_protocol - 0.5, 0.4, f"$d^*\\approx{d_star_protocol:.1f}$\n(in-protocol rate)",
               ha="right", va="center")
    right.set_xlabel("Gate difficulty $d$ (bits)")
    right.set_ylabel("Assignment bias")
    right.set_xticks(d)
    right.set_ylim(-0.03, 0.8)
    right.text(0.02, 0.97, "(b)", transform=right.transAxes, va="top", fontweight="bold")
    return _save(fig, path), d_star, d_star_protocol


# --------------------------------------------------------------------------- Fig. 3
def time_allocation_figure(sweep: dict, path: Path) -> Path:
    rows = sorted(sweep["table_rows"], key=lambda row: row["difficulty_bits"])
    d = [row["difficulty_bits"] for row in rows]
    fig, ax = plt.subplots(figsize=(SINGLE, 6.2 * CM))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.97)
    series = [
        ("useful_share_percent", "useful_share_std", "PoW + solve", PALETTE["useful"], "o", "-"),
        ("useful_share_percent_with_miner_overhead", "useful_share_with_miner_overhead_std",
         "+ miner overhead", PALETTE["pow"], "s", "--"),
        ("useful_share_percent_with_overhead", "useful_share_with_overhead_std",
         "+ validation", PALETTE["overhead"], "^", ":"),
    ]
    for key, std_key, label, colour, marker, style in series:
        ax.errorbar(d, [row[key] for row in rows], yerr=[row[std_key] for row in rows],
                    color=colour, marker=marker, linestyle=style, capsize=2, elinewidth=0.6,
                    label=label)
    for bits, tier in ((8, "Low"), (12, "Medium"), (17, "High")):
        ax.axvline(bits, color="#BBBBBB", linewidth=0.6, zorder=0)
        ax.text(bits, 101, tier, ha="center", va="bottom", fontsize=FONT_PT - 1, color="#555555")
    ax.set_xlabel("Gate difficulty $d$ (bits)")
    ax.set_ylabel("Useful-work share (%)")
    ax.set_xticks(range(min(d), max(d) + 1, 2))
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), title="Denominator",
              title_fontsize=FONT_PT, handlelength=2.6)
    fig.subplots_adjust(top=0.95)
    return _save(fig, path)


# --------------------------------------------------------------------------- Fig. 4
def solve_verify_figure(problem_size: dict, validation: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(SINGLE, 7.4 * CM))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.8)
    rows = problem_size["table_rows"]
    fits = {fit["solver"]: fit for fit in problem_size["growth_fits"]}
    for key, label, colour, marker, growth in SOLVERS:
        cells = sorted((r for r in rows if r["solver"] == key), key=lambda r: r["size"])
        ax.plot([r["size"] for r in cells], [r["mean_solve_seconds"] * 1e3 for r in cells],
                marker, color=colour, markersize=3.5, label=f"Solve, {label}")
        fit = fits[key]
        lo, hi = min(fit["sizes_fitted"]), max(fit["sizes_fitted"])
        grid = [lo + (hi - lo) * i / 200 for i in range(201)]
        ax.plot(grid, [fit["coefficient"] * growth(n) ** fit["exponent"] * 1e3 for n in grid],
                color=colour, linestyle="--", linewidth=0.9)

    asym = validation["asymmetry_vs_size"]
    ax.plot([r["size"] for r in asym], [r["mean_verify_ms"] for r in asym], "D", color="black",
            markersize=3, markerfacecolor="white", label="Validation")
    vfit = problem_size["verification_fit"]
    grid = [4 * (800 / 4) ** (i / 200) for i in range(201)]
    ax.plot(grid, [(vfit["intercept_seconds"] + vfit["slope_seconds_per_node"] * n) * 1e3 for n in grid],
            color="black", linestyle="--", linewidth=0.9)
    ax.plot([], [], color="#666666", linestyle="--", linewidth=0.9, label="Fitted model")

    for crossover in problem_size["verification_crossovers"]:
        n = crossover["fitted_continuous_size"]
        ax.axvline(n, color="#999999", linestyle=(0, (2, 2)), linewidth=0.6, zorder=0)
        ax.text(n * 1.04, 1.3e-3, f"{n:.3g}", rotation=90, va="bottom", ha="left",
                fontsize=FONT_PT - 1, color="#555555")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.5, 1000)
    ax.set_ylim(1e-3, 1e5)
    ax.set_xlabel("Instance size $N$ (log scale)")
    ax.set_ylabel("Time per block (ms, log scale)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.2)
    return _save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the manuscript figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="wiley-submission")
    args = parser.parse_args()

    _style()
    results_dir = ROOT / args.results_dir
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    reseeding = _results(results_dir, "reseeding_bias")
    sweep = _results(results_dir, "time_allocation_sweep")
    problem_size = _results(results_dir, "table4_problem_size")
    validation = _results(results_dir, "table6_validation_cost")

    written = [framework_figure(output / "Fig1.pdf")]
    path, d_star, d_star_protocol = reseeding_figure(reseeding, sweep, output / "Fig2.pdf")
    written.append(path)
    written.append(time_allocation_figure(sweep, output / "Fig3.pdf"))
    written.append(solve_verify_figure(problem_size, validation, output / "Fig4.pdf"))

    for item in written:
        print(f"wrote {item}")
    print(f"d* (raw hash rate) = {d_star:.2f} bits; d* (in-protocol rate) = {d_star_protocol:.2f} bits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
