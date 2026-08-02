"""A problem-class-agnostic Merkle-committed repository.

Mirrors :class:`pouw.tasks.repository.ProblemRepository` but stores generic
:class:`ProblemInstance` objects, so a single repository can hold instances of any
registered class. The commitment, proof, and verification logic are unchanged -- this is
the point of comment 9: the framework machinery does not depend on the task type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ...crypto.merkle import MerkleProof, MerkleTree, verify_membership
from .base import ProblemInstance, get_problem_class


@dataclass(frozen=True)
class GenericRepositorySpec:
    problem_class: str
    num_instances: int
    instance_size: int
    seed: int
    params: dict[str, Any]


class GenericRepository:
    """A set of generic problem instances plus their Merkle commitment."""

    def __init__(self, instances: list[ProblemInstance]) -> None:
        if not instances:
            raise ValueError("repository must contain at least one instance")
        self._instances = list(instances)
        self._tree = MerkleTree([i.to_payload() for i in self._instances])

    @classmethod
    def from_spec(cls, spec: GenericRepositorySpec) -> "GenericRepository":
        problem = get_problem_class(spec.problem_class)
        instances = [
            problem.generate(
                instance_id=f"{spec.problem_class}-{spec.instance_size}-{idx:04d}",
                size=spec.instance_size,
                seed=spec.seed * 1_000_003 + idx,
                **spec.params,
            )
            for idx in range(spec.num_instances)
        ]
        return cls(instances)

    @property
    def root(self) -> bytes:
        return self._tree.root

    @property
    def root_hex(self) -> str:
        return self._tree.root.hex()

    @property
    def depth(self) -> int:
        return self._tree.depth

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self) -> Iterator[ProblemInstance]:
        return iter(self._instances)

    def instance(self, index: int) -> ProblemInstance:
        return self._instances[index]

    def get_proof(self, index: int) -> tuple[ProblemInstance, MerkleProof]:
        return self._instances[index], self._tree.prove(index)

    def verify(self, instance: ProblemInstance, proof: MerkleProof) -> bool:
        return verify_membership(self.root, instance.to_payload(), proof)
