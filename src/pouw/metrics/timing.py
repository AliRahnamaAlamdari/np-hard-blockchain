"""Time-allocation metrics (paper metric M2) and the solve/verify asymmetry (M3).

The paper's percentage columns sum to 100 by construction because only PoW and solve
time enter the denominator; VRF proving, Merkle proof construction, packaging, and
validation are all excluded.  Both that two-way share and a full accounting over
every measured stage are computed here, because the difference is exactly the
overhead the paper's headline number leaves out.

``PAPER_SPEC.md`` §11-U12 notes that the paper does not say whether shares are a ratio
of means or a mean of ratios.  Both are returned.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from ..stats_utils import mean_std, summarise


@dataclass(frozen=True)
class ShareReport:
    """Time allocation between the security gate and the useful work."""

    num_blocks: int
    mean_pow_seconds: float
    mean_solve_seconds: float
    std_pow_seconds: float
    std_solve_seconds: float
    pow_share_percent: float
    useful_share_percent: float
    pow_share_percent_mean_of_ratios: float
    useful_share_percent_mean_of_ratios: float
    useful_share_percent_std: float
    mean_overhead_seconds: float
    useful_share_percent_with_overhead: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _std(values: Sequence[float]) -> float:
    return mean_std(values)[1] if len(values) > 1 else 0.0


def compute_shares(
    pow_seconds: Sequence[float],
    solve_seconds: Sequence[float],
    overhead_seconds: Sequence[float] | None = None,
) -> ShareReport:
    """Compute the paper's two-way share plus an overhead-inclusive variant.

    ``overhead_seconds`` should carry the per-block sum of VRF proving, Merkle proof
    construction, packaging, and validation.  When supplied, the third share treats it
    as part of block-production time rather than discarding it.
    """
    if len(pow_seconds) != len(solve_seconds):
        raise ValueError("pow_seconds and solve_seconds must have equal length")
    if not pow_seconds:
        raise ValueError("cannot compute shares over zero blocks")

    total_pow = sum(pow_seconds)
    total_solve = sum(solve_seconds)
    denominator = total_pow + total_solve
    if denominator <= 0:
        raise ValueError("total block-production time is zero")

    per_block_useful = [
        solve / (pow_ + solve) * 100.0 if (pow_ + solve) > 0 else 0.0
        for pow_, solve in zip(pow_seconds, solve_seconds)
    ]

    overhead = list(overhead_seconds) if overhead_seconds is not None else [0.0] * len(pow_seconds)
    total_overhead = sum(overhead)
    full_denominator = denominator + total_overhead

    useful_share = total_solve / denominator * 100.0
    return ShareReport(
        num_blocks=len(pow_seconds),
        mean_pow_seconds=total_pow / len(pow_seconds),
        mean_solve_seconds=total_solve / len(solve_seconds),
        std_pow_seconds=_std(pow_seconds),
        std_solve_seconds=_std(solve_seconds),
        pow_share_percent=100.0 - useful_share,
        useful_share_percent=useful_share,
        pow_share_percent_mean_of_ratios=100.0 - statistics.fmean(per_block_useful),
        useful_share_percent_mean_of_ratios=statistics.fmean(per_block_useful),
        useful_share_percent_std=_std(per_block_useful),
        mean_overhead_seconds=total_overhead / len(pow_seconds),
        useful_share_percent_with_overhead=total_solve / full_denominator * 100.0,
    )


@dataclass(frozen=True)
class AsymmetryReport:
    """Solve-versus-verify cost asymmetry (paper metric M3)."""

    mean_solve_seconds: float
    mean_verify_seconds: float
    asymmetry_ratio: float
    step_mean_seconds: dict[str, float]
    step_mean_milliseconds: dict[str, float]
    total_verify_milliseconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_asymmetry(
    solve_seconds: Sequence[float],
    step_seconds: dict[str, Sequence[float]],
) -> AsymmetryReport:
    """Per-step validation latencies and the solve/verify ratio."""
    if not solve_seconds:
        raise ValueError("cannot compute asymmetry over zero blocks")
    step_means = {
        label: statistics.fmean(values) for label, values in step_seconds.items() if values
    }
    total_verify = sum(step_means.values())
    mean_solve = statistics.fmean(solve_seconds)
    return AsymmetryReport(
        mean_solve_seconds=mean_solve,
        mean_verify_seconds=total_verify,
        asymmetry_ratio=mean_solve / total_verify if total_verify > 0 else float("inf"),
        step_mean_seconds=step_means,
        step_mean_milliseconds={k: v * 1000.0 for k, v in step_means.items()},
        total_verify_milliseconds=total_verify * 1000.0,
    )


def aggregate_across_seeds(values: Sequence[float]) -> dict[str, float]:
    """Mean, population std, min and max across independent seeds."""
    return summarise(values)
