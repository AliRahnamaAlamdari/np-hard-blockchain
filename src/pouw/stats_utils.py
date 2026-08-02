"""Aggregation helpers that tolerate non-finite values.

``statistics.pstdev`` raises on ``inf``/``nan`` inputs, and these do occur legitimately:
the max/min block ratio is infinite whenever some miner wins no blocks, which is a
normal outcome in short runs.  Aggregating must not crash on it, and must not silently
report a finite number either -- the non-finite entries are counted and reported.
"""

from __future__ import annotations

import math
from typing import Sequence


def finite(values: Sequence[float]) -> list[float]:
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population standard deviation over the finite entries.

    Returns ``(nan, nan)`` when nothing finite is present.  With one finite value the
    standard deviation is 0.0, matching the ``ddof=0`` convention used throughout.
    """
    usable = finite(values)
    if not usable:
        return math.nan, math.nan
    mean = sum(usable) / len(usable)
    if len(usable) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in usable) / len(usable)
    return mean, math.sqrt(variance)


def confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return ``(mean, half_width, ...)`` as ``(low, mean, high)`` of a t-based CI.

    Uses the Student-t critical value for ``n-1`` degrees of freedom, which is the
    correct small-sample interval for the seed counts used here (often n=3..10). Falls
    back to the sample mean with a zero-width interval when fewer than two finite
    observations are present, and degrades to a normal approximation only if SciPy is
    unavailable (it is a pinned dependency, so this path is not normally taken).
    """
    usable = finite(values)
    if not usable:
        return math.nan, math.nan, math.nan
    n = len(usable)
    mean = sum(usable) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in usable) / (n - 1)  # sample variance
    std_error = math.sqrt(variance / n)
    try:
        from scipy import stats

        crit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except ImportError:  # pragma: no cover - scipy is pinned
        crit = 1.96
    half = crit * std_error
    return mean - half, mean, mean + half


def ci_half_width(values: Sequence[float], confidence: float = 0.95) -> float:
    """Half-width of the t-based CI (the ``±`` reported next to a mean)."""
    low, mean, high = confidence_interval(values, confidence)
    if math.isnan(mean):
        return math.nan
    return (high - low) / 2.0


def summarise(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean, sample std, min, max, count, and a t-based confidence interval."""
    usable = finite(values)
    mean, std = mean_std(values)
    low, _, high = confidence_interval(values, confidence)
    # Report the sample standard deviation alongside the population one used elsewhere,
    # so downstream reporting can choose the convention it needs.
    if len(usable) > 1:
        sample_var = sum((v - mean) ** 2 for v in usable) / (len(usable) - 1)
        sample_std = math.sqrt(sample_var)
    else:
        sample_std = 0.0
    return {
        "mean": mean,
        "std": std,
        "sample_std": sample_std,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_half_width": ci_half_width(values, confidence),
        "min": min(usable) if usable else math.nan,
        "max": max(usable) if usable else math.nan,
        "n": len(values),
        "n_finite": len(usable),
    }
