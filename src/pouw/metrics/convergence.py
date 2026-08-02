"""Chain-convergence and correctness metrics (paper metric M1).

The paper reports convergence only as prose -- "No persistent chain inconsistency was
observed" (L483) -- with no fork count, reorg depth, or round count
(``PAPER_SPEC.md`` §11-U15).  These functions produce the numbers that claim needs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConvergenceReport:
    """Whether every node agrees on the tip, and how many blocks were accepted."""

    num_nodes: int
    num_blocks: int
    converged: bool
    distinct_tips: int
    tip_hash: str | None
    blocks_accepted_by_all: int
    blocks_rejected: int
    acceptance_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyse_convergence(
    node_tips: Mapping[int, str],
    acceptance_flags: Sequence[bool],
) -> ConvergenceReport:
    """Formalise M1: converged iff the set of node tip hashes has one element."""
    tips = set(node_tips.values())
    accepted = sum(1 for flag in acceptance_flags if flag)
    num_blocks = len(acceptance_flags)
    return ConvergenceReport(
        num_nodes=len(node_tips),
        num_blocks=num_blocks,
        converged=len(tips) == 1,
        distinct_tips=len(tips),
        tip_hash=next(iter(tips)) if len(tips) == 1 else None,
        blocks_accepted_by_all=accepted,
        blocks_rejected=num_blocks - accepted,
        acceptance_rate=accepted / num_blocks if num_blocks else 0.0,
    )


@dataclass(frozen=True)
class CorrectnessReport:
    """Whether the useful work produced was actually optimal.

    The paper claims exactness by construction (an exhaustive solver) but never checks
    it.  This compares every mined solution against an independent exact solver, so
    "correctness" becomes a measured property.
    """

    num_checked: int
    num_optimal: int
    optimality_rate: float
    max_relative_gap: float
    mean_relative_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyse_correctness(
    achieved_costs: Sequence[float],
    optimal_costs: Sequence[float],
    tolerance: float = 1e-9,
) -> CorrectnessReport:
    """Compare mined tour costs against independently computed optima."""
    if len(achieved_costs) != len(optimal_costs):
        raise ValueError("achieved and optimal cost sequences must have equal length")
    if not achieved_costs:
        return CorrectnessReport(0, 0, 0.0, 0.0, 0.0)
    gaps = [
        (achieved - optimal) / optimal if optimal > 0 else 0.0
        for achieved, optimal in zip(achieved_costs, optimal_costs)
    ]
    optimal_count = sum(1 for gap in gaps if gap <= tolerance)
    return CorrectnessReport(
        num_checked=len(gaps),
        num_optimal=optimal_count,
        optimality_rate=optimal_count / len(gaps),
        max_relative_gap=max(gaps),
        mean_relative_gap=sum(gaps) / len(gaps),
    )
