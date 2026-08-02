#!/usr/bin/env python3
"""Generate figures for the reviewer-response experiments from serialized results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pouw.reporting.figures import PALETTE, _save, _style  # noqa: E402
from pouw.reporting.io import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"


def _res(name):
    p = ROOT / "results" / name / f"{name}.json"
    return read_json(p)["results"] if p.exists() else None


def consensus_figures():
    c = _res("consensus")
    if not c:
        return []
    written = []
    # fork rate vs delay, with CI band
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    delays = [r["delay"] for r in c["delay_rows"]]
    fork = [r["fork_rate"]["mean"] for r in c["delay_rows"]]
    err = [r["fork_rate"]["ci95_half"] for r in c["delay_rows"]]
    ax.errorbar(delays, fork, yerr=err, marker="o", color=PALETTE["pow"], capsize=3, label="Measured (model)")
    ax.set_xlabel("Propagation delay / target block interval")
    ax.set_ylabel("Fork rate (orphaned blocks)")
    ax.legend(frameon=False, fontsize=8)
    written += _save(fig, OUT / "consensus_fork_vs_delay")

    # fork rate & convergence vs miner count
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    miners = [r["miners"] for r in c["scale_rows"]]
    fork = [r["fork_rate"]["mean"] for r in c["scale_rows"]]
    err = [r["fork_rate"]["ci95_half"] for r in c["scale_rows"]]
    ax.errorbar(miners, fork, yerr=err, marker="s", color=PALETTE["useful"], capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Number of miners (log scale)")
    ax.set_ylabel("Fork rate")
    ax.set_ylim(0, max(fork) * 1.5 + 0.05)
    written += _save(fig, OUT / "consensus_scale")
    return written


def adversarial_figures():
    a = _res("adversarial")
    if not a:
        return []
    written = []
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    beta = [r["adversary_hashrate_share"] for r in a["malicious_majority"]]
    share = [r["adversary_mainchain_share"] for r in a["malicious_majority"]]
    err = [r["std"] for r in a["malicious_majority"]]
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["reference"], label="share = hash rate")
    ax.errorbar(beta, share, yerr=err, marker="o", color=PALETTE["accent"], capsize=3, label="Measured (model)")
    ax.axvline(0.5, color="gray", linewidth=0.6)
    ax.set_xlabel("Adversary hash-rate share")
    ax.set_ylabel("Adversary main-chain block share")
    ax.legend(frameon=False, fontsize=8)
    written += _save(fig, OUT / "adversarial_majority")
    return written


def problem_class_figure():
    p = _res("problem_classes")
    if not p:
        return []
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    classes = sorted({r["problem_class"] for r in p["table_rows"]})
    colors = [PALETTE["pow"], PALETTE["useful"], PALETTE["overhead"], PALETTE["accent"]]
    for idx, cls in enumerate(classes):
        rows = [r for r in p["table_rows"] if r["problem_class"] == cls]
        rows.sort(key=lambda r: r["size"])
        ax.plot([r["size"] for r in rows], [r["mean_verify_ms"] for r in rows],
                marker="o", color=colors[idx % len(colors)], label=cls)
    ax.set_xlabel("Instance size $N$")
    ax.set_ylabel("Validator verification time (ms)")
    ax.set_title("Verification cost is class-independent", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, OUT / "problem_class_verify")


def reviewer2_figures():
    """Figures for the Reviewer-2 experiments (reseeding, precompute, validation)."""
    written = []
    # reseeding: assignment bias vs target
    r = _res("reseeding_bias")
    if r:
        _style()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        rows = r["bias_vs_target"]
        d = [x["difficulty_bits"] for x in rows]
        bias = [x["assignment_bias"] for x in rows]
        ax.plot(d, bias, marker="o", color=PALETTE["accent"])
        d_star = r["economics"]["safe_target_bits_d_star"]
        ax.axvline(d_star, color="gray", linestyle="--", label=f"safe target $d^*$={d_star:.1f}")
        ax.set_xlabel("PoW difficulty $d$ (bits)")
        ax.set_ylabel("Achievable assignment bias")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Reseeding bias collapses above the safe target", fontsize=9)
        written += _save(fig, OUT / "reseeding_bias_vs_target")
    # feasibility vs optimality
    v = _res("validation_scale")
    if v:
        _style()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        rows = [x for x in v["feasibility_vs_optimality"] if x["problem_class"] == "tsp"]
        rows.sort(key=lambda x: x["size"])
        sizes = [x["size"] for x in rows]
        ax.plot(sizes, [x["feasibility_ms"] for x in rows], marker="o", color=PALETTE["pow"], label="Feasibility check $V(I,w)$")
        ax.plot(sizes, [x["optimality_verify_ms"] for x in rows], marker="s", color=PALETTE["accent"], label="Optimality re-solve")
        ax.set_yscale("log")
        ax.set_xlabel("Instance size $N$ (TSP)")
        ax.set_ylabel("Validation time (ms, log scale)")
        ax.legend(frameon=False, fontsize=8)
        written += _save(fig, OUT / "feasibility_vs_optimality")
    # precompute freshness
    p = _res("precomputation_scale")
    if p:
        _style()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        rows = p["freshness_breakeven"]
        rs = [x["repository_size"] for x in rows]
        rho = [x["breakeven_churn_fraction_per_epoch"] for x in rows]
        ax.plot(rs, rho, marker="o", color=PALETTE["overhead"])
        ax.axhline(1.0, color="gray", linestyle="--", label="feasible threshold ($\\rho^*=1$)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Repository size $|\\mathcal{R}|$")
        ax.set_ylabel("Break-even churn $\\rho^*$ per epoch")
        ax.legend(frameon=False, fontsize=8)
        written += _save(fig, OUT / "precompute_freshness")
    return written


def main() -> int:
    written = []
    written += consensus_figures()
    written += adversarial_figures()
    written += problem_class_figure()
    written += reviewer2_figures()
    for w in written:
        print("wrote", w)
    return 0 if written else 1

if __name__ == "__main__":
    raise SystemExit(main())
