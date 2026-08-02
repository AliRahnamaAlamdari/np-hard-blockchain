"""Runnable experiments.  Each maps to a table, figure, or claim in the paper."""

from .ablation_components import ComponentAblation
from .ablation_vrf_grinding import VRFGrindingAblation
from .adversarial import AdversarialExperiment
from .base import Experiment, ExperimentOutput
from .baselines_compare import BaselineComparisonExperiment
from .consensus import ConsensusExperiment
from .convergence import ConvergenceExperiment
from .precomputation_scale import PrecomputationScaleExperiment
from .problem_classes import ProblemClassExperiment
from .reseeding import ReseedingBiasExperiment
from .validation_scale import ValidationScaleExperiment
from .sensitivity import (
    HashRateSensitivity,
    RepositorySizeSensitivity,
    SolverAdvantageSensitivity,
)
from .table3_time_allocation import TimeAllocationExperiment
from .table4_problem_size import ProblemSizeExperiment
from .table5_fairness import FairnessExperiment
from .table6_validation_cost import ValidationCostExperiment

REGISTRY: dict[str, type[Experiment]] = {
    experiment.name: experiment
    for experiment in (
        TimeAllocationExperiment,
        ProblemSizeExperiment,
        FairnessExperiment,
        ValidationCostExperiment,
        ConvergenceExperiment,
        VRFGrindingAblation,
        ComponentAblation,
        RepositorySizeSensitivity,
        HashRateSensitivity,
        SolverAdvantageSensitivity,
        # Reviewer-response experiments
        ProblemClassExperiment,
        ConsensusExperiment,
        AdversarialExperiment,
        BaselineComparisonExperiment,
        # Reviewer-2 experiments
        ReseedingBiasExperiment,
        PrecomputationScaleExperiment,
        ValidationScaleExperiment,
    )
}

# Which paper artifact each experiment regenerates.
PAPER_MAP = {
    "table3_time_allocation": "Table 3 (tab:timeallocation)",
    "table4_problem_size": "Table 4 (tab:problemsize)",
    "table5_fairness": "Table 5 (tab:fairness)",
    "table6_validation_cost": "Table 6 (tab:validationcost)",
    "convergence": "Section 'Convergence and Correctness' (prose claim, L483)",
    "ablation_vrf_grinding": "New: tests the grinding bound argued at L375",
    "ablation_components": "New: measures the claims argued in Table 1 (tab:designrationale)",
    "sensitivity_repository_size": "New: varies N from Equation (3)",
    "sensitivity_hash_rate": "New: fairness beyond equal-power miners",
    "sensitivity_solver_advantage": "New: mining edge from a better exact solver",
    "problem_classes": "Reviewer C9: framework across TSP/knapsack/coloring/SAT",
    "consensus": "Reviewer C3: fork rate, convergence, liveness, difficulty adjustment",
    "adversarial": "Reviewer C2: grinding/precompute/replay/bias/steering/majority",
    "baselines": "Reviewer C5: measured baseline comparison",
    "reseeding_bias": "Reviewer-2 C6/C8: reseeding bias vs hashing target, safe-target boundary",
    "precomputation_scale": "Reviewer-2 C3: static-repo precompute cost + freshness break-even",
    "validation_scale": "Reviewer-2 C7/C10: feasibility-vs-optimality + communication cost",
}

__all__ = [
    "PAPER_MAP",
    "REGISTRY",
    "AdversarialExperiment",
    "BaselineComparisonExperiment",
    "ComponentAblation",
    "ConsensusExperiment",
    "ConvergenceExperiment",
    "Experiment",
    "ExperimentOutput",
    "FairnessExperiment",
    "HashRateSensitivity",
    "ProblemClassExperiment",
    "ProblemSizeExperiment",
    "RepositorySizeSensitivity",
    "SolverAdvantageSensitivity",
    "TimeAllocationExperiment",
    "VRFGrindingAblation",
    "ValidationCostExperiment",
]
