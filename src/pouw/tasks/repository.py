"""The Merkle-authenticated problem repository of paper §3.2.

The repository holds ``N`` TSP instances committed by a single Merkle root ``R``.
Instances live off-chain (here: a JSON file); only ``R`` is consensus-critical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..crypto.merkle import MerkleProof, MerkleTree, verify_membership
from .tsp import TSPInstance, generate_instance, solve


@dataclass(frozen=True)
class RepositorySpec:
    """Declarative description of a repository, sufficient to rebuild it exactly."""

    num_instances: int
    instance_size: int
    seed: int
    coordinate_range: tuple[float, float]
    difficulty_class: str | None = None
    id_prefix: str = "I"
    # Solver the repository publisher runs once per instance to establish the
    # reference cost that the validators' quality threshold is measured against.
    # ``None`` publishes no bound, and the threshold is then unavailable.
    reference_solver: str | None = None


class ProblemRepository:
    """A set of instances plus the Merkle commitment over them."""

    def __init__(self, instances: list[TSPInstance]) -> None:
        if not instances:
            raise ValueError("repository must contain at least one instance")
        self._instances = list(instances)
        self._tree = MerkleTree([instance.to_payload() for instance in self._instances])

    # -- construction --------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: RepositorySpec) -> "ProblemRepository":
        """Build deterministically from a spec.  Same spec => same Merkle root.

        When ``spec.reference_solver`` is set, the publisher solves each instance
        once and writes the resulting cost into the instance metadata.  Metadata is
        part of the Merkle leaf, so the bound is committed by the repository root
        exactly as the instance data is: a miner cannot raise the bound to make its
        own tour acceptable without breaking the membership proof.
        """
        instances = [
            generate_instance(
                instance_id=f"{spec.id_prefix}-{spec.instance_size}-{index:04d}",
                size=spec.instance_size,
                seed=spec.seed * 1_000_003 + index,
                coordinate_range=spec.coordinate_range,
                difficulty_class=spec.difficulty_class,
            )
            for index in range(spec.num_instances)
        ]
        if spec.reference_solver is not None:
            instances = [
                _with_reference_cost(instance, spec.reference_solver)
                for instance in instances
            ]
        return cls(instances)

    @classmethod
    def load(cls, path: Path) -> "ProblemRepository":
        payload = json.loads(Path(path).read_text())
        return cls([TSPInstance.from_payload(entry) for entry in payload["instances"]])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "root": self.root.hex(),
                    "num_instances": len(self._instances),
                    "instances": [
                        instance.to_payload() for instance in self._instances
                    ],
                },
                indent=2,
            )
        )

    # -- access --------------------------------------------------------------

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

    def __iter__(self) -> Iterator[TSPInstance]:
        return iter(self._instances)

    def instance(self, index: int) -> TSPInstance:
        return self._instances[index]

    def get_proof(self, index: int) -> tuple[TSPInstance, MerkleProof]:
        """``GetProof(i)`` of Algorithm 1, line 9."""
        return self._instances[index], self._tree.prove(index)

    def verify(self, instance: TSPInstance, proof: MerkleProof) -> bool:
        """``VerifyMerkle`` of Algorithm 2, line 6."""
        return verify_membership(self.root, instance.to_payload(), proof)

    def summary(self) -> dict[str, Any]:
        sizes = sorted({instance.size for instance in self._instances})
        return {
            "num_instances": len(self._instances),
            "instance_sizes": sizes,
            "merkle_root": self.root_hex,
            "merkle_depth": self.depth,
        }


def _with_reference_cost(instance: TSPInstance, solver: str) -> TSPInstance:
    """Return ``instance`` with the publisher's reference cost in its metadata."""
    solution, _ = solve(instance, solver)
    metadata = dict(instance.metadata)
    metadata["reference_cost"] = float(solution.cost)
    metadata["reference_solver"] = solver
    return TSPInstance(
        instance_id=instance.instance_id,
        coordinates=instance.coordinates,
        metadata=metadata,
    )


def reference_cost(instance: TSPInstance) -> float | None:
    """The repository-published bound for ``instance``, if it carries one.

    Reading it here rather than at each call site keeps one definition of where the
    bound comes from: the Merkle-committed leaf, never the block.
    """
    value = instance.metadata.get("reference_cost")
    return float(value) if value is not None else None


def index_from_vrf_output(vrf_output: bytes, num_instances: int) -> int:
    """Equation (3): ``i = Integer(y) mod N``.

    Implemented over the full 256- or 512-bit VRF output.  Modulo bias is then
    bounded by ``num_instances / 2**bitlen`` and is cryptographically negligible; the
    paper never states the output width, so a truncated implementation could be
    measurably biased (see ``PAPER_SPEC.md``, closing note).
    """
    if num_instances <= 0:
        raise ValueError("num_instances must be positive")
    return int.from_bytes(vrf_output, "big") % num_instances


def modulo_bias(output_bits: int, num_instances: int) -> float:
    """Max relative over-selection probability of Equation (3), for the ablation.

    Returns 0.0 when ``num_instances`` divides ``2**output_bits`` exactly.
    """
    total = 1 << output_bits
    remainder = total % num_instances
    if remainder == 0:
        return 0.0
    favoured = total // num_instances + 1
    uniform = total / num_instances
    return favoured / uniform - 1.0
