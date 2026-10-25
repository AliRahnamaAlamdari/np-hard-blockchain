"""Matplotlib figures for results the paper reports only as tables.

The paper contains one figure and it is a flowchart -- no quantitative result is
plotted (``PAPER_SPEC.md`` §8.9).  These are the plots a reviewer would expect: the
difficulty/share tuning curve, the exponential solve-time growth against ``(N-1)!``,
the fairness distribution with its uniform expectation, and the solve/verify
asymmetry on log axes.

Figures are saved as both PDF (for the manuscript) and PNG (for quick inspection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_DPI = 200
FIGURE_SIZE = (6.0, 4.0)

# Colour-blind-safe qualitative palette (Okabe-Ito), used consistently across figures.
PALETTE = {
    "pow": "#0072B2",
    "useful": "#E69F00",
    "overhead": "#009E73",
    "reference": "#666666",
    "accent": "#D55E00",
    "alt": "#CC79A7",
}


def _style() -> None:
    plt.rcParams.update(
        {
            # Springer preflight rejects Type 3 fonts; 42 embeds TrueType instead.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def _save(fig: "plt.Figure", path: Path) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        target = path.with_suffix(suffix)
        fig.savefig(target, dpi=FIGURE_DPI, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written


def time_allocation_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> list[Path]:
    """Stacked share of block-production time per difficulty tier, with error bars."""
    _style()
    fig, (ax_share, ax_abs) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    labels = [row["difficulty_label"] for row in rows]
    positions = range(len(labels))
    pow_share = [row["pow_share_percent"] for row in rows]
    useful_share = [row["useful_share_percent"] for row in rows]
    useful_err = [row.get("useful_share_std", 0.0) for row in rows]

    ax_share.bar(positions, pow_share, color=PALETTE["pow"], label="PoW gate")
    ax_share.bar(
        positions,
        useful_share,
        bottom=pow_share,
        color=PALETTE["useful"],
        yerr=useful_err,
        capsize=3,
        ecolor="black",
        label="Useful work",
    )
    ax_share.set_xticks(list(positions), labels)
    ax_share.set_ylabel("Share of block-production time (%)")
    ax_share.set_ylim(0, 100)
    ax_share.set_xlabel("PoW difficulty tier")
    ax_share.legend(frameon=False, fontsize=8)

    ax_abs.errorbar(
        positions,
        [row["mean_pow_seconds"] for row in rows],
        yerr=[row["std_pow_seconds"] for row in rows],
        marker="o",
        color=PALETTE["pow"],
        capsize=3,
        label="PoW",
    )
    ax_abs.errorbar(
        positions,
        [row["mean_solve_seconds"] for row in rows],
        yerr=[row["std_solve_seconds"] for row in rows],
        marker="s",
        color=PALETTE["useful"],
        capsize=3,
        label="Useful work",
    )
    ax_abs.set_yscale("log")
    ax_abs.set_xticks(list(positions), labels)
    ax_abs.set_ylabel("Mean time per block (s, log scale)")
    ax_abs.set_xlabel("PoW difficulty tier")
    ax_abs.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def problem_size_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> list[Path]:
    """Solve time against N on a log axis, overlaid with the (N-1)! reference curve."""
    _style()
    fig, (ax_time, ax_share) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    sizes = [row["size"] for row in rows]
    solve = [row["mean_solve_seconds"] for row in rows]
    errors = [row["std_solve_seconds"] for row in rows]

    ax_time.errorbar(
        sizes, solve, yerr=errors, marker="o", color=PALETTE["useful"], capsize=3,
        label="Measured brute force",
    )
    # Only the sizes where the Held-Karp baseline was actually run carry the key;
    # the extended sweep caps it, so plot the sub-series rather than assuming all.
    held_karp = [
        (row["size"], row["mean_heldkarp_seconds"])
        for row in rows
        if row.get("mean_heldkarp_seconds") is not None
    ]
    if held_karp:
        ax_time.plot(
            [size for size, _ in held_karp],
            [seconds for _, seconds in held_karp],
            marker="^",
            color=PALETTE["overhead"],
            label="Held-Karp (exact)",
        )
    import math

    anchor = solve[-1] / math.factorial(sizes[-1] - 1)
    ax_time.plot(
        sizes,
        [anchor * math.factorial(size - 1) for size in sizes],
        linestyle="--",
        color=PALETTE["reference"],
        label=r"$(N-1)!$ reference",
    )
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Instance size $N$")
    ax_time.set_ylabel("Mean solve time (s, log scale)")
    ax_time.legend(frameon=False, fontsize=8)

    ax_share.plot(
        sizes,
        [row["useful_share_percent"] for row in rows],
        marker="o",
        color=PALETTE["accent"],
    )
    ax_share.set_xlabel("Instance size $N$")
    ax_share.set_ylabel("Useful-work share (%)")
    ax_share.set_ylim(0, 105)
    return _save(fig, path)


def fairness_figure(
    counts: Mapping[int, int], total: int, path: Path, seed_counts: Sequence[Mapping[int, int]] | None = None
) -> list[Path]:
    """Blocks won per miner against the uniform expectation, with across-seed spread."""
    _style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    miner_ids = sorted(counts)
    values = [counts[miner_id] for miner_id in miner_ids]
    expected = total / len(miner_ids)

    if seed_counts:
        spread = [
            [run[miner_id] for run in seed_counts if miner_id in run] for miner_id in miner_ids
        ]
        errors = [
            (max(values_) - min(values_)) / 2 if len(values_) > 1 else 0.0 for values_ in spread
        ]
        ax.bar(miner_ids, values, color=PALETTE["pow"], yerr=errors, capsize=3, ecolor="black")
    else:
        ax.bar(miner_ids, values, color=PALETTE["pow"])

    ax.axhline(expected, color=PALETTE["accent"], linestyle="--", label=f"Expected = {expected:.0f}")
    ax.set_xlabel("Miner ID")
    ax.set_ylabel("Blocks won")
    ax.set_xticks(miner_ids)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def asymmetry_figure(
    step_ms: Mapping[str, float], solve_ms: float, path: Path
) -> list[Path]:
    """Per-step validation latency against solve cost, on a log axis."""
    _style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    labels = list(step_ms.keys()) + ["useful-work solve"]
    values = list(step_ms.values()) + [solve_ms]
    colors = [PALETTE["pow"]] * len(step_ms) + [PALETTE["accent"]]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency (ms, log scale)")
    ax.invert_yaxis()
    return _save(fig, path)


def sensitivity_figure(
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    x_label: str,
    y_label: str,
    path: Path,
    errors: Mapping[str, Sequence[float]] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> list[Path]:
    """Generic line plot used by the sensitivity experiments."""
    _style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    palette_cycle = [PALETTE["pow"], PALETTE["useful"], PALETTE["overhead"], PALETTE["accent"], PALETTE["alt"]]
    for index, (label, values) in enumerate(series.items()):
        error = errors.get(label) if errors else None
        ax.errorbar(
            x_values,
            values,
            yerr=error,
            marker="o",
            capsize=3,
            color=palette_cycle[index % len(palette_cycle)],
            label=label,
        )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def grinding_figure(
    attempts: Sequence[int],
    distinct_tasks: Mapping[str, Sequence[float]],
    path: Path,
) -> list[Path]:
    """Distinct tasks obtainable per PoW seed, by VRF backend.

    This is the picture behind ``PAPER_SPEC.md`` §11-I6: a unique-output VRF is a flat
    line at 1, a malleable one grows without bound.
    """
    _style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = [PALETTE["pow"], PALETTE["accent"], PALETTE["overhead"]]
    for index, (label, values) in enumerate(distinct_tasks.items()):
        ax.plot(attempts, values, marker="o", color=colors[index % len(colors)], label=label)
    ax.set_xlabel("VRF evaluations per PoW seed")
    ax.set_ylabel("Distinct tasks reachable")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)
