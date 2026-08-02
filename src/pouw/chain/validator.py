"""Validator: Algorithm 2 of the paper, with each of the four checks timed.

The paper elides every argument in Algorithm 2 (``PAPER_SPEC.md`` §11-U1).  The
binding chosen here is the sound one, and it matters:

* ``s`` is **recomputed** from ``B.header`` rather than read from the block, so a miner
  cannot supply a seed unrelated to its PoW;
* ``VRF.Verify`` is bound to the public key **in the header**, and that key is also
  hashed into the header, so the PoW commits to the key that must produce the VRF;
* the Merkle proof is checked against the claimed instance payload, not merely
  recomputed to a root, so a valid path for a *different* instance is rejected.

Any of these three loosened turns validation into a check that a real adversary
passes trivially.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..crypto.hashing import meets_target
from ..crypto.vrf import VRFBackend
from ..tasks.repository import ProblemRepository, index_from_vrf_output
from ..tasks.tsp import verify_solution
from .block import Block


class RejectReason(str, Enum):
    """Why a block was rejected, in the order Algorithm 2 applies its checks."""

    POW_FAILED = "pow_failed"
    POW_HASH_MISMATCH = "pow_hash_mismatch"
    VRF_INVALID = "vrf_invalid"
    VRF_OUTPUT_MISMATCH = "vrf_output_mismatch"
    INDEX_MISMATCH = "index_mismatch"
    MERKLE_INVALID = "merkle_invalid"
    SOLUTION_INVALID = "solution_invalid"
    BAD_HEIGHT = "bad_height"
    BAD_PARENT = "bad_parent"


@dataclass
class ValidationResult:
    """Accept/reject plus the per-step latencies that feed the validation-cost table."""

    accepted: bool
    reason: RejectReason | None = None
    step_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(self.step_seconds.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason.value if self.reason else None,
            "step_seconds": dict(self.step_seconds),
            "total_seconds": self.total_seconds,
        }


@dataclass
class Validator:
    """A validating node holding the chain state and the repository commitment."""

    validator_id: int
    vrf: VRFBackend
    repository: ProblemRepository
    target: int
    quality_threshold: float | None = None
    cost_tolerance: float = 1e-9

    def validate(
        self,
        block: Block,
        expected_height: int | None = None,
        expected_parent: str | None = None,
    ) -> ValidationResult:
        """Run Algorithm 2.  Returns on the first failing check, as a real node would."""
        steps: dict[str, float] = {}

        if expected_height is not None and block.height != expected_height:
            return ValidationResult(False, RejectReason.BAD_HEIGHT, steps)
        if expected_parent is not None and block.header.previous_hash != expected_parent:
            return ValidationResult(False, RejectReason.BAD_PARENT, steps)

        # Step 1 -- recompute the PoW digest and test it against the target.
        start = time.perf_counter()
        digest = block.header.digest()
        pow_ok = meets_target(digest, self.target)
        hash_matches = digest.hex() == block.pow_hash
        steps["pow_check"] = time.perf_counter() - start
        if not pow_ok:
            return ValidationResult(False, RejectReason.POW_FAILED, steps)
        if not hash_matches:
            return ValidationResult(False, RejectReason.POW_HASH_MISMATCH, steps)

        # Step 2 -- verify the VRF proof against the header's public key and this seed.
        start = time.perf_counter()
        public_key = bytes.fromhex(block.header.miner_public_key)
        beta = self.vrf.verify(public_key, digest, bytes.fromhex(block.vrf_proof))
        steps["vrf_verify"] = time.perf_counter() - start
        if beta is None:
            return ValidationResult(False, RejectReason.VRF_INVALID, steps)
        if beta.hex() != block.vrf_output:
            return ValidationResult(False, RejectReason.VRF_OUTPUT_MISMATCH, steps)

        # Step 3 -- reconstruct the task index from the VRF output.
        start = time.perf_counter()
        expected_index = index_from_vrf_output(beta, len(self.repository))
        index_ok = expected_index == block.instance_id
        steps["index_check"] = time.perf_counter() - start
        if not index_ok:
            return ValidationResult(False, RejectReason.INDEX_MISMATCH, steps)

        # Step 4 -- verify repository membership of the claimed instance.
        start = time.perf_counter()
        merkle_ok = self.repository.verify(block.instance, block.merkle_proof)
        steps["merkle_verify"] = time.perf_counter() - start
        if not merkle_ok:
            return ValidationResult(False, RejectReason.MERKLE_INVALID, steps)

        # Step 5 -- check the useful-work solution: feasibility and claimed cost.
        start = time.perf_counter()
        solution_ok, _ = verify_solution(
            block.instance,
            block.solution,
            quality_threshold=self.quality_threshold,
            reference_cost=block.solution_cost if self.quality_threshold is not None else None,
            cost_tolerance=self.cost_tolerance,
        )
        steps["solution_check"] = time.perf_counter() - start
        if not solution_ok:
            return ValidationResult(False, RejectReason.SOLUTION_INVALID, steps)

        return ValidationResult(True, None, steps)
