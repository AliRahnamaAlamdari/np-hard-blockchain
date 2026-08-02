"""Class-agnostic single-block framework cycle: PoW -> VRF -> select -> solve -> validate.

This runs the *entire* protocol for any registered problem class, reusing the same
crypto (hash gate, VRF, Merkle) as the TSP path. It exists so reviewer comment 9's claim
--- that the framework is independent of the task type --- is demonstrated by executing
the full mine-and-validate cycle for TSP, knapsack, graph colouring, and SAT through one
code path, rather than only for TSP.

It also returns per-stage timings and supports validating tampered blocks, which the
adversarial and multi-class experiments use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..crypto.hashing import difficulty_to_target, meets_target
from ..crypto.merkle import MerkleProof
from ..crypto.vrf import VRFBackend, build_vrf
from ..tasks.problems.base import ProblemInstance, Solution, get_problem_class
from ..tasks.problems.repository import GenericRepository
from ..tasks.repository import index_from_vrf_output
from .block import BlockHeader


@dataclass
class GenericBlock:
    header: BlockHeader
    pow_hash: str
    vrf_output: str
    vrf_proof: str
    instance_id: int
    instance: ProblemInstance
    merkle_proof: MerkleProof
    solution: Solution
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class GenericRunner:
    """Runs one full framework cycle for a chosen problem class."""

    repository: GenericRepository
    vrf: VRFBackend
    difficulty_bits: int
    secret: bytes
    public: bytes
    miner_id: int = 0
    quality_threshold: float | None = None

    @property
    def target(self) -> int:
        return difficulty_to_target(self.difficulty_bits)

    def mine(self, height: int, previous_hash: str, tx_root: str, max_attempts: int = 10_000_000) -> GenericBlock:
        # Stage 1: PoW gate
        template = BlockHeader(height, previous_hash, tx_root, self.miner_id, self.public.hex(), 0)
        start = time.perf_counter()
        nonce = 0
        digest = template.digest()
        while not meets_target(digest, self.target):
            nonce += 1
            if nonce >= max_attempts:
                raise RuntimeError("PoW search exhausted")
            template = template.with_nonce(nonce)
            digest = template.digest()
        pow_seconds = time.perf_counter() - start

        # Stage 2: VRF task assignment
        t = time.perf_counter()
        beta, proof = self.vrf.prove(self.secret, digest)
        vrf_seconds = time.perf_counter() - t
        index = index_from_vrf_output(beta, len(self.repository))

        instance, merkle_proof = self.repository.get_proof(index)

        # Useful work
        problem = get_problem_class(instance.problem_class)
        t = time.perf_counter()
        solution, _ops = problem.solve_exact(instance)
        solve_seconds = time.perf_counter() - t

        return GenericBlock(
            header=template,
            pow_hash=digest.hex(),
            vrf_output=beta.hex(),
            vrf_proof=proof.hex(),
            instance_id=index,
            instance=instance,
            merkle_proof=merkle_proof,
            solution=solution,
            timings={"pow_s": pow_seconds, "vrf_prove_s": vrf_seconds, "solve_s": solve_seconds},
        )

    def validate(self, block: GenericBlock) -> tuple[bool, str, dict[str, float]]:
        """Full validation with per-step timings; mirrors Algorithm 2 for any class."""
        steps: dict[str, float] = {}

        t = time.perf_counter()
        digest = block.header.digest()
        pow_ok = meets_target(digest, self.target) and digest.hex() == block.pow_hash
        steps["pow_check"] = time.perf_counter() - t
        if not pow_ok:
            return False, "pow_failed", steps

        t = time.perf_counter()
        beta = self.vrf.verify(bytes.fromhex(block.header.miner_public_key), digest, bytes.fromhex(block.vrf_proof))
        steps["vrf_verify"] = time.perf_counter() - t
        if beta is None or beta.hex() != block.vrf_output:
            return False, "vrf_invalid", steps

        t = time.perf_counter()
        expected_index = index_from_vrf_output(beta, len(self.repository))
        steps["index_check"] = time.perf_counter() - t
        if expected_index != block.instance_id:
            return False, "index_mismatch", steps

        t = time.perf_counter()
        merkle_ok = self.repository.verify(block.instance, block.merkle_proof)
        steps["merkle_verify"] = time.perf_counter() - t
        if not merkle_ok:
            return False, "merkle_invalid", steps

        t = time.perf_counter()
        problem = get_problem_class(block.instance.problem_class)
        ok, reason = problem.verify(block.instance, block.solution, quality_threshold=self.quality_threshold)
        steps["solution_check"] = time.perf_counter() - t
        if not ok:
            return False, f"solution_invalid:{reason}", steps

        return True, "ok", steps


def build_runner(
    problem_class: str,
    repository: GenericRepository,
    difficulty_bits: int,
    seed_material: bytes,
    vrf_backend: str = "ecvrf",
    vrf_kwargs: dict[str, Any] | None = None,
    quality_threshold: float | None = None,
) -> GenericRunner:
    vrf = build_vrf(vrf_backend, **(vrf_kwargs or {}))
    keys = vrf.keygen(seed_material)
    return GenericRunner(
        repository=repository,
        vrf=vrf,
        difficulty_bits=difficulty_bits,
        secret=keys.secret,
        public=keys.public,
        quality_threshold=quality_threshold,
    )
