"""Evaluation metrics, implemented directly from the definitions in PAPER_SPEC.md §9."""

from .convergence import (
    ConvergenceReport,
    CorrectnessReport,
    analyse_convergence,
    analyse_correctness,
)
from .fairness import (
    FairnessReport,
    analyse,
    chi_square_uniformity,
    gini,
    jain_index,
    max_min_ratio,
    std_dev,
)
from .timing import (
    AsymmetryReport,
    ShareReport,
    aggregate_across_seeds,
    compute_asymmetry,
    compute_shares,
)

__all__ = [
    "AsymmetryReport",
    "ConvergenceReport",
    "CorrectnessReport",
    "FairnessReport",
    "ShareReport",
    "aggregate_across_seeds",
    "analyse",
    "analyse_convergence",
    "analyse_correctness",
    "chi_square_uniformity",
    "compute_asymmetry",
    "compute_shares",
    "gini",
    "jain_index",
    "max_min_ratio",
    "std_dev",
]
