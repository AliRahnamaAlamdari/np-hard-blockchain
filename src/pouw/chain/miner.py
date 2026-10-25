"""Miner: Algorithm 1 of the paper, with each stage timed separately.

Deviations from the printed pseudocode, all logged in ``PAPER_SPEC.md``:

* §11-I2 -- Algorithm 1 increments the nonce *after* computing the digest and *before*
  the ``until`` test, so the returned nonce is one past the winning value and no
  honest block would validate.  The loop here returns the nonce that actually
  satisfied the target.
* §11-I3 -- Algorithm 1 has no branch for ``VerifySolution`` returning false.  A
  failure here indicates a solver bug rather than an adversary, so it raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..crypto.hashing import meets_target
from ..crypto.vrf import KeyPair, VRFBackend
from ..logging_utils import Timer
from ..tasks.repository import (
    ProblemRepository,
    index_from_vrf_output,
    reference_cost as published_reference_cost,
)
from ..tasks.tsp import solve, verify_solution
from .block import Block, BlockHeader


class MiningError(RuntimeError):
    """Raised when mining cannot complete (exhausted nonces, or an invalid solve)."""


@dataclass
class PoWResult:
    """Outcome of the Stage-1 nonce search."""

    header: BlockHeader
    digest: bytes
    attempts: int
    seconds: float


@dataclass
class Miner:
    """One mining node: a keypair, a hash rate, and the two-stage procedure."""

    miner_id: int
    keys: KeyPair
    vrf: VRFBackend
    hash_rate: float = 1.0

    @property
    def public_key_hex(self) -> str:
        return self.keys.public.hex()

    # -- Stage 1 -------------------------------------------------------------

    def search_pow(
        self,
        height: int,
        previous_hash: str,
        tx_root: str,
        target: int,
        max_attempts: int,
        timer: Timer | None = None,
        start_nonce: int = 0,
    ) -> PoWResult:
        """Equation (1): search for a nonce whose header digest is below ``target``.

        Every miner searches its own space because ``miner_id`` and the miner's public
        key are header fields, so two miners never collide on the same digest sequence.
        """
        import time

        template = BlockHeader(
            height=height,
            previous_hash=previous_hash,
            tx_root=tx_root,
            miner_id=self.miner_id,
            miner_public_key=self.public_key_hex,
            nonce=start_nonce,
        )
        start = time.perf_counter()
        attempts = 0
        nonce = start_nonce
        while attempts < max_attempts:
            header = template.with_nonce(nonce)
            digest = header.digest()
            attempts += 1
            if meets_target(digest, target):
                elapsed = time.perf_counter() - start
                if timer is not None:
                    timer.record("pow_search", elapsed)
                return PoWResult(
                    header=header, digest=digest, attempts=attempts, seconds=elapsed
                )
            nonce += 1
        raise MiningError(
            f"miner {self.miner_id} exhausted {max_attempts} nonces at height {height}"
        )

    def search_pow_batch(
        self,
        template: BlockHeader,
        target: int,
        start_nonce: int,
        batch_size: int,
    ) -> tuple[PoWResult | None, int]:
        """Try ``batch_size`` nonces from ``start_nonce``; return a hit or None.

        Used by the network race so every miner advances in lockstep and all of them
        stop when the first one succeeds -- which is what a real parallel race does.
        Searching each miner to completion instead would inflate total network work by
        a factor of the miner count.
        """
        import time

        start = time.perf_counter()
        nonce = start_nonce
        for _ in range(batch_size):
            header = template.with_nonce(nonce)
            digest = header.digest()
            nonce += 1
            if meets_target(digest, target):
                return (
                    PoWResult(
                        header=header,
                        digest=digest,
                        attempts=nonce - start_nonce,
                        seconds=time.perf_counter() - start,
                    ),
                    nonce,
                )
        return None, nonce

    def header_template(
        self, height: int, previous_hash: str, tx_root: str
    ) -> BlockHeader:
        """Header with this miner's identity bound in, so search spaces are disjoint."""
        return BlockHeader(
            height=height,
            previous_hash=previous_hash,
            tx_root=tx_root,
            miner_id=self.miner_id,
            miner_public_key=self.public_key_hex,
            nonce=0,
        )

    # -- Stage 2 -------------------------------------------------------------

    def assign_task(
        self,
        seed: bytes,
        repository: ProblemRepository,
        timer: Timer | None = None,
    ) -> tuple[bytes, bytes, int]:
        """Equations (2) and (3): prove the VRF and derive the instance index."""
        import time

        start = time.perf_counter()
        beta, proof = self.vrf.prove(self.keys.secret, seed)
        if timer is not None:
            timer.record("vrf_prove", time.perf_counter() - start)
        index = index_from_vrf_output(beta, len(repository))
        return beta, proof, index

    # -- Full procedure ------------------------------------------------------

    def mine(
        self,
        height: int,
        previous_hash: str,
        tx_root: str,
        target: int,
        repository: ProblemRepository,
        solver: str,
        max_attempts: int,
        quality_threshold: float | None = None,
        cost_tolerance: float = 1e-9,
        timer: Timer | None = None,
        pow_result: PoWResult | None = None,
    ) -> tuple[Block, dict[str, Any]]:
        """Run Algorithm 1 end to end and return the block plus per-stage timings.

        ``pow_result`` may be supplied when the Stage-1 race has already been run, so
        the winning miner is not asked to redo it.
        """
        import time

        timer = timer or Timer()
        if pow_result is None:
            pow_result = self.search_pow(
                height=height,
                previous_hash=previous_hash,
                tx_root=tx_root,
                target=target,
                max_attempts=max_attempts,
                timer=timer,
            )

        seed = pow_result.digest
        vrf_start = time.perf_counter()
        beta, proof, index = self.assign_task(seed, repository, timer=None)
        vrf_seconds = time.perf_counter() - vrf_start

        proof_start = time.perf_counter()
        instance, merkle_proof = repository.get_proof(index)
        proof_seconds = time.perf_counter() - proof_start

        solve_start = time.perf_counter()
        solution, operations = solve(instance, solver)
        solve_seconds = time.perf_counter() - solve_start

        # Two separate questions, deliberately not merged.  Feasibility and cost
        # consistency failing means the solver is broken, so it raises.  Missing the
        # repository's published quality bound is a legitimate protocol outcome --
        # it is what a weak solver is supposed to produce -- so it is recorded and
        # left for the validator to act on.
        ok, reason = verify_solution(
            instance, solution, quality_threshold=None, cost_tolerance=cost_tolerance
        )
        if not ok:
            raise MiningError(
                f"miner {self.miner_id} produced an invalid solution at height "
                f"{height}: {reason}"
            )
        bound = published_reference_cost(instance)
        meets_quality = True
        if quality_threshold is not None:
            meets_quality, _ = verify_solution(
                instance,
                solution,
                quality_threshold=quality_threshold,
                reference_cost=bound,
                cost_tolerance=cost_tolerance,
            )

        package_start = time.perf_counter()
        block = Block(
            header=pow_result.header,
            pow_hash=seed.hex(),
            vrf_output=beta.hex(),
            vrf_proof=proof.hex(),
            instance_id=index,
            instance=instance,
            merkle_proof=merkle_proof,
            solution=solution,
            solution_cost=solution.cost,
        )
        block_hash = block.block_hash()
        package_seconds = time.perf_counter() - package_start

        timings = {
            "pow_s": pow_result.seconds,
            "vrf_prove_s": vrf_seconds,
            "merkle_proof_s": proof_seconds,
            "solve_s": solve_seconds,
            "package_s": package_seconds,
            "pow_attempts": pow_result.attempts,
            "solver_operations": operations,
            "block_hash": block_hash,
            "reference_cost": bound,
            "meets_quality_threshold": meets_quality,
        }
        for label in ("pow_s", "vrf_prove_s", "merkle_proof_s", "solve_s", "package_s"):
            timer.record(label.removesuffix("_s"), timings[label])
        return block, timings
