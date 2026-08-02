"""Problem-class abstraction so the useful-work task is a pluggable module.

Reviewer comment 9 asks that the framework be separable from the TSP case study. A
``ProblemClass`` bundles the four operations the framework needs from any NP-hard task:

* ``generate`` -- produce a deterministic instance from a seed;
* ``solve_exact`` -- the miner's solver (returns a solution and an operation count);
* ``solve_reference`` -- an independent exact solver, used to check optimality and to
  quantify the solver-choice advantage of comment 8's analysis;
* ``verify`` -- the validator's cheap check ``V(I, w)``.

Every class produces a generic :class:`ProblemInstance` and :class:`Solution` carrying a
``problem_class`` tag, so the same repository, VRF selection, Merkle commitment, and
block-validation path serve all of them without change. The registry at the bottom lets
configs name a class by string.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProblemInstance:
    """A problem-class-agnostic instance.

    ``data`` holds the class-specific payload (city coordinates, item weights, a CNF,
    ...). ``size`` is the natural hardness parameter of the class (cities, items,
    vertices, variables), reported in every scaling table.
    """

    instance_id: str
    problem_class: str
    size: int
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Canonical dict form -- this is what a Merkle leaf commits to."""
        return {
            "instance_id": self.instance_id,
            "problem_class": self.problem_class,
            "size": self.size,
            "data": self.data,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ProblemInstance":
        return cls(
            instance_id=payload["instance_id"],
            problem_class=payload["problem_class"],
            size=int(payload["size"]),
            data=payload["data"],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class Solution:
    """A candidate solution and its objective/quality value.

    ``witness`` is the class-specific certificate (a tour, a chosen-item set, a colour
    assignment, a truth assignment). ``cost`` is the objective a validator recomputes;
    ``feasible`` records whether the witness satisfies the class's hard constraints.
    """

    problem_class: str
    witness: Any
    cost: float
    feasible: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "problem_class": self.problem_class,
            "witness": self.witness,
            "cost": float(self.cost),
            "feasible": self.feasible,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Solution":
        return cls(
            problem_class=payload["problem_class"],
            witness=payload["witness"],
            cost=float(payload["cost"]),
            feasible=bool(payload.get("feasible", True)),
        )


class ProblemClass(abc.ABC):
    """Interface every useful-work task must implement to plug into the framework."""

    #: short stable name used in configs and the ``problem_class`` tag
    name: str = "abstract"
    #: human-readable asymptotic cost of ``solve_exact`` (for the complexity draft)
    exact_complexity: str = "unknown"
    #: asymptotic cost of ``verify`` in the witness size
    verify_complexity: str = "unknown"
    #: whether this is a decision (SAT-style) or optimization problem
    is_decision: bool = False
    #: True when a *random* witness is essentially always rejected by feasibility alone
    #: (hard-constraint classes: colouring, SAT). False for pure optimization classes
    #: (TSP, knapsack), where a random witness is often feasible and a quality threshold
    #: is needed to reject a feasible-but-suboptimal solution. This is the crux of the
    #: framework-vs-case-study distinction for comment 9.
    constraint_rejects_random: bool = False

    @abc.abstractmethod
    def generate(self, instance_id: str, size: int, seed: int, **params: Any) -> ProblemInstance:
        """Deterministically generate one instance."""

    @abc.abstractmethod
    def solve_exact(self, instance: ProblemInstance) -> tuple[Solution, int]:
        """The miner's exact solver. Returns ``(solution, operation_count)``."""

    @abc.abstractmethod
    def solve_reference(self, instance: ProblemInstance) -> tuple[Solution, int]:
        """An independent exact solver for optimality checking and solver-advantage."""

    @abc.abstractmethod
    def verify(
        self,
        instance: ProblemInstance,
        solution: Solution,
        quality_threshold: float | None = None,
        reference_cost: float | None = None,
        cost_tolerance: float = 1e-9,
    ) -> tuple[bool, str]:
        """Validator-side check ``V(I, w)``. Returns ``(accepted, reason)``."""

    def random_witness(self, instance: ProblemInstance, rng: Any) -> Solution:
        """A cheap, usually-invalid witness, used by adversarial-submission tests."""
        raise NotImplementedError


_REGISTRY: dict[str, ProblemClass] = {}


def register(problem: ProblemClass) -> ProblemClass:
    _REGISTRY[problem.name] = problem
    return problem


def get_problem_class(name: str) -> ProblemClass:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown problem class {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def available_classes() -> list[str]:
    return sorted(_REGISTRY)
