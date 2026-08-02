"""Fairness metrics (paper metric M4) plus the significance test the paper omits.

Definitions follow ``PAPER_SPEC.md`` §9-M4.  For blocks won ``x_j`` across ``n``
miners with total ``B`` and expectation ``E = B/n``:

    StdDev = sqrt( (1/n) * sum_j (x_j - E)^2 )      population std, ddof = 0
    MaxMin = max_j x_j / min_j x_j
    Jain   = (sum_j x_j)^2 / ( n * sum_j x_j^2 )

The ``ddof = 0`` convention is not stated in the paper; it is the one that reproduces
the paper's reported 7.6420 exactly (the sample std of the same counts is 8.5440),
and ``tests/test_metrics.py`` pins that against the paper's own numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence


@dataclass(frozen=True)
class FairnessReport:
    """All fairness statistics for one block-distribution outcome."""

    counts: tuple[int, ...]
    num_miners: int
    total_blocks: int
    expected_per_miner: float
    std_dev: float
    sample_std_dev: float
    max_min_ratio: float
    jain_index: float
    gini: float
    chi_square: float
    chi_square_dof: int
    chi_square_p_value: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = list(self.counts)
        return payload


def std_dev(counts: Sequence[float], ddof: int = 0) -> float:
    """Standard deviation of block counts.  ``ddof=0`` matches the paper."""
    n = len(counts)
    if n - ddof <= 0:
        raise ValueError("not enough observations for the requested ddof")
    mean = sum(counts) / n
    return math.sqrt(sum((value - mean) ** 2 for value in counts) / (n - ddof))


def max_min_ratio(counts: Sequence[float]) -> float:
    """Ratio of the luckiest to the unluckiest miner.  ``inf`` if any miner won none."""
    smallest = min(counts)
    if smallest == 0:
        return math.inf
    return max(counts) / smallest


def jain_index(counts: Sequence[float]) -> float:
    """Jain's fairness index in ``[1/n, 1]``; 1 means perfectly equal shares."""
    n = len(counts)
    if n == 0:
        raise ValueError("jain_index requires at least one miner")
    total = sum(counts)
    sum_squares = sum(value * value for value in counts)
    if sum_squares == 0:
        return 1.0
    return (total * total) / (n * sum_squares)


def gini(counts: Sequence[float]) -> float:
    """Gini coefficient of the reward distribution; 0 = equal, 1 = fully concentrated.

    Not in the paper.  Reported alongside Jain because Jain saturates near 1 and is
    insensitive to the tail, whereas a reviewer asking about concentration wants a
    metric that separates plausible outcomes.
    """
    ordered = sorted(counts)
    n = len(ordered)
    total = sum(ordered)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def chi_square_uniformity(counts: Sequence[int]) -> tuple[float, int, float]:
    """Pearson chi-square goodness-of-fit against a uniform block distribution.

    Returns ``(statistic, dof, p_value)``.  This is the test the paper's fairness
    claim needs and does not report: without it, "no observable systematic bias" is an
    eyeball judgement on a single realisation.
    """
    n = len(counts)
    if n < 2:
        raise ValueError("chi-square test requires at least two miners")
    total = sum(counts)
    expected = total / n
    if expected == 0:
        raise ValueError("cannot test uniformity with zero blocks")
    statistic = sum((count - expected) ** 2 / expected for count in counts)
    dof = n - 1
    try:
        from scipy import stats

        p_value = float(stats.chi2.sf(statistic, dof))
    except ImportError:  # pragma: no cover - scipy is a pinned dependency
        p_value = float("nan")
    return statistic, dof, p_value


def chi_square_expected(
    counts: Sequence[int], expected_proportions: Sequence[float]
) -> tuple[float, int, float]:
    """Chi-square goodness-of-fit against arbitrary expected proportions.

    Used for heterogeneous hash rates, where the null hypothesis is "block share is
    proportional to hash-rate share" rather than "all miners are equal".  A fixed
    tolerance on the observed deviation cannot answer that question: the deviation
    expected purely from sampling shrinks as the number of blocks grows, so any fixed
    threshold either passes trivially or fails trivially depending on run length.
    """
    if len(counts) != len(expected_proportions):
        raise ValueError("counts and expected_proportions must have equal length")
    if len(counts) < 2:
        raise ValueError("chi-square test requires at least two miners")
    total = sum(counts)
    expected = [proportion * total for proportion in expected_proportions]
    if any(value <= 0 for value in expected):
        raise ValueError("every expected count must be positive")
    statistic = sum(
        (observed - predicted) ** 2 / predicted
        for observed, predicted in zip(counts, expected)
    )
    dof = len(counts) - 1
    try:
        from scipy import stats

        p_value = float(stats.chi2.sf(statistic, dof))
    except ImportError:  # pragma: no cover - scipy is a pinned dependency
        p_value = float("nan")
    return statistic, dof, p_value


def analyse(counts: Sequence[int]) -> FairnessReport:
    """Compute every fairness statistic for one outcome."""
    counts = tuple(int(value) for value in counts)
    n = len(counts)
    total = sum(counts)
    statistic, dof, p_value = chi_square_uniformity(counts)
    return FairnessReport(
        counts=counts,
        num_miners=n,
        total_blocks=total,
        expected_per_miner=total / n,
        std_dev=std_dev(counts, ddof=0),
        sample_std_dev=std_dev(counts, ddof=1) if n > 1 else 0.0,
        max_min_ratio=max_min_ratio(counts),
        jain_index=jain_index(counts),
        gini=gini(counts),
        chi_square=statistic,
        chi_square_dof=dof,
        chi_square_p_value=p_value,
    )
