"""Growth-model fits for measured solver times (objection: extrapolation).

A solve-time table stops at whatever size the budget allowed, and every claim about
sizes beyond it is an extrapolation.  This module makes that extrapolation checkable
rather than asserted: each solver's measured times are regressed against its own
theoretical growth term, and the fitted exponent, the coefficient of determination,
and the per-point residuals are reported alongside any predicted value.

The model is ``t(N) = a * g(N)^b`` where ``g`` is the solver's growth term.  Taking
logarithms makes it linear, ``ln t = ln a + b ln g(N)``, so ordinary least squares
recovers ``a`` and ``b``.  A fitted ``b`` near ``1.0`` says the theoretical term
describes the measurement; a ``b`` far from ``1.0`` says it does not, and that is
exactly the thing a reader should be told before trusting an extrapolated number.

Small instances are excluded from the fit by a configurable floor on the measured
time, because below roughly a millisecond the timing is dominated by fixed
interpreter overhead rather than by the enumeration -- the same effect the paper
already reports as a 46% relative standard deviation at N=4.  Excluded points are
named in the result rather than silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Theoretical growth term per solver backend.  The label is what the paper prints.
GROWTH_MODELS: dict[str, tuple[str, Callable[[int], float]]] = {
    "brute_force": ("(N-1)!", lambda n: float(math.factorial(n - 1))),
    "held_karp": ("2^N N^2", lambda n: float((1 << n) * n * n)),
    "nearest_neighbor_2opt": ("N^2", lambda n: float(n * n)),
}

# Below this measured mean, timings are interpreter overhead rather than algorithm.
DEFAULT_MIN_SECONDS = 1.0e-3


@dataclass(frozen=True)
class Residual:
    """One measured point against what the fitted model predicts for it."""

    size: int
    measured_seconds: float
    predicted_seconds: float
    log_residual: float
    percent_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "measured_seconds": self.measured_seconds,
            "predicted_seconds": self.predicted_seconds,
            "log_residual": self.log_residual,
            "percent_error": self.percent_error,
        }


@dataclass(frozen=True)
class GrowthFit:
    """``t(N) = coefficient * g(N)^exponent`` fitted in log space."""

    solver: str
    growth_label: str
    coefficient: float
    exponent: float
    r_squared: float
    sizes_fitted: list[int]
    sizes_excluded: list[int]
    min_seconds: float
    residuals: list[Residual] = field(default_factory=list)

    @property
    def max_abs_percent_error(self) -> float:
        return max((abs(r.percent_error) for r in self.residuals), default=0.0)

    def predict(self, size: int) -> float:
        """Predicted solve seconds at ``size`` under the fitted model."""
        growth = GROWTH_MODELS[self.solver][1](size)
        return self.coefficient * (growth ** self.exponent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "growth_label": self.growth_label,
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "r_squared": self.r_squared,
            "sizes_fitted": list(self.sizes_fitted),
            "sizes_excluded": list(self.sizes_excluded),
            "min_seconds": self.min_seconds,
            "max_abs_percent_error": self.max_abs_percent_error,
            "residuals": [r.to_dict() for r in self.residuals],
        }


class FitError(ValueError):
    """Raised when there are too few usable points to fit a two-parameter model."""


def fit_growth(
    solver: str,
    sizes: Sequence[int],
    mean_seconds: Sequence[float],
    min_seconds: float = DEFAULT_MIN_SECONDS,
) -> GrowthFit:
    """Least-squares fit of ``ln t`` on ``ln g(N)`` for one solver backend."""
    if solver not in GROWTH_MODELS:
        raise FitError(
            f"no growth model for solver {solver!r}; known: {sorted(GROWTH_MODELS)}"
        )
    if len(sizes) != len(mean_seconds):
        raise FitError("sizes and mean_seconds must have equal length")

    label, growth = GROWTH_MODELS[solver]
    kept: list[tuple[int, float]] = []
    excluded: list[int] = []
    for size, seconds in zip(sizes, mean_seconds):
        if seconds >= min_seconds and seconds > 0.0:
            kept.append((int(size), float(seconds)))
        else:
            excluded.append(int(size))
    if len(kept) < 3:
        raise FitError(
            f"{solver}: {len(kept)} point(s) above the {min_seconds:g}s floor; "
            "a two-parameter fit needs at least 3"
        )

    xs = [math.log(growth(size)) for size, _ in kept]
    ys = [math.log(seconds) for _, seconds in kept]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0.0:
        raise FitError(f"{solver}: growth term is constant across the fitted sizes")
    exponent = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / variance
    log_coefficient = y_mean - exponent * x_mean
    coefficient = math.exp(log_coefficient)

    total_ss = sum((y - y_mean) ** 2 for y in ys)
    residual_ss = sum(
        (y - (log_coefficient + exponent * x)) ** 2 for x, y in zip(xs, ys)
    )
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 1.0

    residuals = []
    for (size, seconds), x, y in zip(kept, xs, ys):
        predicted = math.exp(log_coefficient + exponent * x)
        residuals.append(
            Residual(
                size=size,
                measured_seconds=seconds,
                predicted_seconds=predicted,
                log_residual=y - (log_coefficient + exponent * x),
                percent_error=(predicted - seconds) / seconds * 100.0,
            )
        )

    return GrowthFit(
        solver=solver,
        growth_label=label,
        coefficient=coefficient,
        exponent=exponent,
        r_squared=r_squared,
        sizes_fitted=[size for size, _ in kept],
        sizes_excluded=excluded,
        min_seconds=min_seconds,
        residuals=residuals,
    )


@dataclass(frozen=True)
class LinearFit:
    """``v(N) = intercept + slope * N`` -- the model for validator-side cost.

    Algorithm 2 is dominated by a size-independent VRF verification, but the
    feasibility check recomputes the submitted tour and the membership check hashes
    a leaf whose payload carries the instance, so the cost is flat *plus* a term
    linear in ``N``.  Fitting both terms is what makes "verification is independent
    of problem size" a measured statement with a stated slope rather than an
    assertion, and it is what the solve-versus-verify crossover must be solved
    against once the heuristic reaches sizes where the linear term is visible.
    """

    intercept: float
    slope: float
    r_squared: float
    sizes_fitted: list[int]
    mean_seconds: float = 0.0
    min_r_squared: float = 0.0

    @property
    def slope_is_resolved(self) -> bool:
        """Whether the linear term is distinguishable from zero at all.

        A slope that is negative, or one whose fit explains almost none of the
        variance, means the size-dependent part of validation is smaller than the
        run-to-run noise in the size-independent part.  That is a real and useful
        finding -- it is what "verification is independent of problem size" means
        when stated as a measurement -- but it is not a model to extrapolate with,
        so :meth:`predict` falls back to the flat mean instead of continuing a
        downward line into negative time.
        """
        return self.slope > 0.0 and self.r_squared >= self.min_r_squared

    def predict(self, size: float) -> float:
        if not self.slope_is_resolved:
            return self.mean_seconds
        return self.intercept + self.slope * size

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept_seconds": self.intercept,
            "slope_seconds_per_node": self.slope,
            "r_squared": self.r_squared,
            "slope_is_resolved": self.slope_is_resolved,
            "flat_mean_seconds": self.mean_seconds,
            "model_used": "linear" if self.slope_is_resolved else "flat_mean",
            "min_r_squared": self.min_r_squared,
            "sizes_fitted": list(self.sizes_fitted),
        }


# Below this coefficient of determination the linear term is treated as unresolved.
DEFAULT_MIN_R_SQUARED = 0.5


def fit_linear(
    sizes: Sequence[int],
    seconds: Sequence[float],
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
) -> LinearFit:
    """Least-squares fit of measured verification cost against instance size."""
    if len(sizes) != len(seconds):
        raise FitError("sizes and seconds must have equal length")
    points = sorted({int(s): float(v) for s, v in zip(sizes, seconds)}.items())
    if len(points) < 2:
        raise FitError("a linear fit needs at least two distinct sizes")
    xs = [float(size) for size, _ in points]
    ys = [value for _, value in points]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0.0:
        raise FitError("verification sizes are all equal")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / variance
    intercept = y_mean - slope * x_mean
    total_ss = sum((y - y_mean) ** 2 for y in ys)
    residual_ss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return LinearFit(
        intercept=intercept,
        slope=slope,
        r_squared=1.0 - residual_ss / total_ss if total_ss > 0 else 1.0,
        sizes_fitted=[size for size, _ in points],
        mean_seconds=y_mean,
        min_r_squared=min_r_squared,
    )


@dataclass(frozen=True)
class Crossover:
    """The instance size above which solving costs more than verifying."""

    solver: str
    verify_seconds_at_crossover: float
    fitted_size: int | None
    fitted_continuous_size: float | None
    measured_size: int | None
    measured_bracket: tuple[int, int] | None
    measured_solve_seconds: float | None
    measured_verify_seconds: float | None
    search_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "verify_seconds_at_crossover": self.verify_seconds_at_crossover,
            "verify_ms_at_crossover": self.verify_seconds_at_crossover * 1000.0,
            "fitted_size": self.fitted_size,
            "fitted_continuous_size": self.fitted_continuous_size,
            "measured_size": self.measured_size,
            "measured_bracket_low": self.measured_bracket[0]
            if self.measured_bracket
            else None,
            "measured_bracket_high": self.measured_bracket[1]
            if self.measured_bracket
            else None,
            "measured_solve_ms": self.measured_solve_seconds * 1000.0
            if self.measured_solve_seconds is not None
            else None,
            "measured_verify_ms": self.measured_verify_seconds * 1000.0
            if self.measured_verify_seconds is not None
            else None,
            "search_limit": self.search_limit,
        }


def crossover_size(
    fit: GrowthFit,
    verify: float | LinearFit,
    sizes: Sequence[int] | None = None,
    mean_solve_seconds: Sequence[float] | None = None,
    mean_verify_seconds: Sequence[float] | None = None,
    search_limit: int = 4096,
) -> Crossover:
    """Smallest ``N`` at which solving costs at least as much as verifying.

    Two answers are returned and the paper reports both.  ``measured_size`` is the
    smallest *measured* size whose mean solve time already reaches the mean
    validation time recorded at that same size, so it rests on nothing but the
    timings and is the figure to quote wherever it exists.  ``fitted_size`` solves
    the growth fit against the verification model and is the only answer available
    when the crossover lies outside the measured range; it inherits whatever error
    the residuals show.

    ``verify`` may be a constant, appropriate while validation is flat in ``N``, or
    a :class:`LinearFit`, which is what the heuristic's sizes require once the
    feasibility check's linear term becomes visible.
    """
    verify_at = verify.predict if isinstance(verify, LinearFit) else (lambda n: float(verify))

    fitted_size: int | None = None
    for size in range(3, search_limit + 1):
        if fit.predict(size) >= verify_at(size):
            fitted_size = size
            break

    continuous: float | None = None
    growth = GROWTH_MODELS[fit.solver][1]
    if fitted_size is not None and fitted_size > 3:
        low, high = float(fitted_size - 1), float(fitted_size)
        for _ in range(60):
            mid = (low + high) / 2.0
            # g is defined on integers only, so interpolate its logarithm instead.
            log_growth = _interpolate_log_growth(growth, mid)
            value = fit.coefficient * math.exp(fit.exponent * log_growth)
            if value >= verify_at(mid):
                high = mid
            else:
                low = mid
        continuous = high

    measured_size: int | None = None
    bracket: tuple[int, int] | None = None
    measured_solve: float | None = None
    measured_verify: float | None = None
    if sizes is not None and mean_solve_seconds is not None and mean_verify_seconds is not None:
        triples = sorted(
            zip((int(s) for s in sizes), mean_solve_seconds, mean_verify_seconds)
        )
        previous: int | None = None
        for size, solve, verify_seconds in triples:
            if solve >= verify_seconds:
                measured_size = size
                measured_solve = solve
                measured_verify = verify_seconds
                bracket = (previous, size) if previous is not None else None
                break
            previous = size

    return Crossover(
        solver=fit.solver,
        verify_seconds_at_crossover=(
            verify_at(fitted_size) if fitted_size is not None else verify_at(3)
        ),
        fitted_size=fitted_size,
        fitted_continuous_size=continuous,
        measured_size=measured_size,
        measured_bracket=bracket,
        measured_solve_seconds=measured_solve,
        measured_verify_seconds=measured_verify,
        search_limit=search_limit,
    )


def _interpolate_log_growth(growth: Callable[[int], float], size: float) -> float:
    """``ln g`` at a non-integer size, by linear interpolation between neighbours."""
    low = math.floor(size)
    high = low + 1
    fraction = size - low
    log_low = math.log(growth(low))
    log_high = math.log(growth(high))
    return log_low + fraction * (log_high - log_low)
