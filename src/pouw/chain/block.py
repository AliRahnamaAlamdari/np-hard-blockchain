"""Block header and block structure.

The block carries every component of the paper's Equation (4)

    B = (header, n, s, pk_m, y, pi_vrf, i, I_i, pi_i, w, c)

and every field of the paper's Table 2.  Note that Table 2 omits ``header``, ``n``,
``pk_m`` and ``I_i`` (``PAPER_SPEC.md`` §11-I5); ``pk_m`` in particular is required for
``VRF.Verify``, so validation would be impossible without it and it is included here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..crypto.hashing import canonical_json, sha256
from ..crypto.merkle import MerkleProof
from ..tasks.tsp import TSPInstance, TSPSolution


@dataclass(frozen=True)
class BlockHeader:
    """Everything hashed by the Stage-1 gate.

    Equation (1) writes ``H(header || n)`` while Algorithm 1 writes ``H(header)`` with
    the nonce folded in by ``BuildHeader`` (``PAPER_SPEC.md`` §11-I4).  The two are
    reconciled here by making ``nonce`` a header field, so a single canonical encoding
    serves both and a validator can recompute the digest from the block alone.
    """

    height: int
    previous_hash: str
    tx_root: str
    miner_id: int
    miner_public_key: str
    nonce: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "previous_hash": self.previous_hash,
            "tx_root": self.tx_root,
            "miner_id": self.miner_id,
            "miner_public_key": self.miner_public_key,
            "nonce": self.nonce,
        }

    def encode(self) -> bytes:
        return canonical_json(self.to_payload())

    def digest(self) -> bytes:
        """``s = H(header)`` -- the PoW output and the VRF seed."""
        return sha256(self.encode())

    def with_nonce(self, nonce: int) -> "BlockHeader":
        return BlockHeader(
            height=self.height,
            previous_hash=self.previous_hash,
            tx_root=self.tx_root,
            miner_id=self.miner_id,
            miner_public_key=self.miner_public_key,
            nonce=nonce,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BlockHeader":
        return cls(**payload)


@dataclass(frozen=True)
class Block:
    """A candidate block carrying all four independently verifiable artifacts."""

    header: BlockHeader
    pow_hash: str
    vrf_output: str
    vrf_proof: str
    instance_id: int
    instance: TSPInstance
    merkle_proof: MerkleProof
    solution: TSPSolution
    solution_cost: float
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.header.height

    @property
    def miner_id(self) -> int:
        return self.header.miner_id

    def block_hash(self) -> str:
        """Commitment over every consensus-relevant field.

        Included so that "the block cannot be modified without invalidating the
        cryptographic linkage" (paper §Problem Statement, condition 5) is a property
        the implementation actually enforces rather than an assertion.
        """
        return sha256(
            canonical_json(
                {
                    "header": self.header.to_payload(),
                    "pow_hash": self.pow_hash,
                    "vrf_output": self.vrf_output,
                    "vrf_proof": self.vrf_proof,
                    "instance_id": self.instance_id,
                    "instance": self.instance.to_payload(),
                    "merkle_leaf": self.merkle_proof.leaf.hex(),
                    "solution": self.solution.to_payload(),
                    "solution_cost": self.solution_cost,
                }
            )
        ).hex()

    def summary(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "miner_id": self.miner_id,
            "nonce": self.header.nonce,
            "pow_hash": self.pow_hash,
            "instance_id": self.instance_id,
            "instance_size": self.instance.size,
            "solution_cost": self.solution_cost,
            "block_hash": self.block_hash(),
            "merkle_proof_depth": self.merkle_proof.depth,
            "merkle_proof_bytes": self.merkle_proof.size_bytes(),
        }
