"""Adversarial harnesses that empirically corroborate the formal security claims.

Reviewer comment 2 asks that the security properties be backed by adversarial
experiments, not only prose. Each function here mounts a concrete attack against the
real protocol machinery (real VRF, real hash gate, real Merkle repository) and returns
measured outcomes that the theorems in ``draft_security_model.tex`` predict.

The attacks:

* :func:`grinding_advantage` -- distinct tasks reachable per paid PoW seed (task grinding).
* :func:`precomputation_exposure` -- probability an assigned task was pre-solved, vs the
  fraction of the repository the adversary pre-solved and vs repository size.
* :func:`replay_rejection` -- reuse of a VRF proof/seed at a new header/height.
* :func:`assignment_bias` -- empirical uniformity of VRF task selection (χ² test).
* :func:`adaptive_steering_cost` -- expected PoW seeds an adversary must burn to land a
  task in a favoured subset (adaptive manipulation).

Malicious-majority behaviour is measured separately through the consensus simulator,
because it is a chain-level property; see :mod:`pouw.chain.consensus_sim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..crypto.hashing import difficulty_to_target, expected_attempts, meets_target
from ..crypto.vrf import VRFBackend
from ..metrics.fairness import chi_square_uniformity
from ..tasks.repository import index_from_vrf_output


def _mine_seed(vrf_public_header: str, secret: bytes, target: int, seed_material: bytes, max_attempts: int) -> bytes:
    """Return a hash digest below ``target`` by varying a counter (a lightweight PoW)."""
    import hashlib

    for nonce in range(max_attempts):
        digest = hashlib.sha256(seed_material + nonce.to_bytes(8, "big")).digest()
        if meets_target(digest, target):
            return digest
    raise RuntimeError("PoW exhausted in adversary harness")


@dataclass
class GrindingResult:
    vrf_backend: str
    unique_output: bool
    distinct_tasks_per_seed: float
    all_proofs_valid: bool


def grinding_advantage(
    vrf: VRFBackend, secret: bytes, public: bytes, repo_size: int, evaluations: int
) -> GrindingResult:
    """How many distinct tasks one *fixed* PoW seed can be ground into.

    A unique-output VRF pins this at 1. A malleable primitive lets the adversary
    re-evaluate on the same seed and reach many tasks (its ``prove`` accepts an
    ``attempt`` argument). Every produced proof is checked to verify, so the count
    reflects *usable* tasks, not rejected garbage.
    """
    seed = b"fixed-pow-seed-for-grinding"
    distinct: set[int] = set()
    all_valid = True
    supports_attempt = vrf.name == "ecdsa_pseudo"
    for attempt in range(evaluations):
        if supports_attempt:
            beta, proof = vrf.prove(secret, seed, attempt=attempt)  # type: ignore[call-arg]
        else:
            beta, proof = vrf.prove(secret, seed)
        if vrf.verify(public, seed, proof) != beta:
            all_valid = False
        distinct.add(index_from_vrf_output(beta, repo_size))
    return GrindingResult(
        vrf_backend=vrf.name,
        unique_output=vrf.is_unique_output(),
        distinct_tasks_per_seed=float(len(distinct)),
        all_proofs_valid=all_valid,
    )


@dataclass
class PrecomputationResult:
    repo_size: int
    precomputed_fraction: float
    empirical_hit_rate: float
    theoretical_hit_rate: float
    trials: int


def precomputation_exposure(
    vrf: VRFBackend, secret: bytes, repo_size: int, precomputed_fraction: float, trials: int, rng
) -> PrecomputationResult:
    """Probability the VRF-assigned task falls in a pre-solved subset.

    Because assignment is uniform and independent of the adversary's choice of which
    instances to pre-solve, the hit rate equals the pre-solved fraction regardless of
    how the adversary chooses the subset -- so maintaining a fixed coverage as the
    repository grows costs work linear in the repository size. The experiment confirms
    the hit rate matches the fraction and does not exceed it.
    """
    precomputed = set(rng.sample(range(repo_size), int(repo_size * precomputed_fraction)))
    hits = 0
    for t in range(trials):
        seed = f"precomp-seed-{t}".encode()
        beta, _ = vrf.prove(secret, seed)
        index = index_from_vrf_output(beta, repo_size)
        if index in precomputed:
            hits += 1
    return PrecomputationResult(
        repo_size=repo_size,
        precomputed_fraction=len(precomputed) / repo_size,
        empirical_hit_rate=hits / trials,
        theoretical_hit_rate=len(precomputed) / repo_size,
        trials=trials,
    )


@dataclass
class ReplayResult:
    seeds_differ_across_headers: bool
    cross_header_proof_rejected: bool
    replayed_index_inconsistent: bool
    detail: str


def replay_rejection(vrf: VRFBackend, secret: bytes, public: bytes, repo_size: int) -> ReplayResult:
    """A VRF proof is bound to the header (hence height/parent) that produced its seed.

    We build two real headers differing only in height/parent, derive each one's PoW
    seed, and show: (i) the seeds differ, so (ii) a proof produced for header A does not
    verify under header B's seed, and (iii) even if an attacker copies A's ``vrf_output``
    into a block at height B, the index a validator reconstructs from B's freshly derived
    seed disagrees with the copied output. Replay across heights therefore fails at the
    validator with no additional anti-replay bookkeeping.
    """
    from .block import BlockHeader

    header_a = BlockHeader(10, "parent-A", "tx", 0, public.hex(), 0)
    header_b = BlockHeader(11, "parent-B", "tx", 0, public.hex(), 0)
    seed_a = header_a.digest()
    seed_b = header_b.digest()

    beta_a, proof_a = vrf.prove(secret, seed_a)
    cross = vrf.verify(public, seed_b, proof_a)
    cross_rejected = cross is None or cross != beta_a

    # Attacker copies beta_a into a block at header_b; validator derives its own output
    # from seed_b and compares the implied index.
    beta_b, _ = vrf.prove(secret, seed_b)
    index_from_replayed = index_from_vrf_output(beta_a, repo_size)
    index_from_fresh = index_from_vrf_output(beta_b, repo_size)

    return ReplayResult(
        seeds_differ_across_headers=seed_a != seed_b,
        cross_header_proof_rejected=cross_rejected,
        replayed_index_inconsistent=index_from_replayed != index_from_fresh,
        detail=(
            "the PoW seed is a function of the header; a proof for one height does not "
            "verify at another, and a copied VRF output implies a different task index "
            "than the one a validator derives from the new header's seed"
        ),
    )


@dataclass
class AssignmentBiasResult:
    repo_size: int
    samples: int
    chi_square: float
    dof: int
    p_value: float
    uniform_at_5pct: bool


def assignment_bias(vrf: VRFBackend, secret: bytes, repo_size: int, samples: int) -> AssignmentBiasResult:
    """Empirical uniformity of VRF task assignment over many seeds (χ² test)."""
    from collections import Counter

    counts: Counter[int] = Counter()
    for s in range(samples):
        beta, _ = vrf.prove(secret, f"assign-{s}".encode())
        counts[index_from_vrf_output(beta, repo_size)] += 1
    observed = [counts.get(i, 0) for i in range(repo_size)]
    chi, dof, p = chi_square_uniformity(observed)
    return AssignmentBiasResult(repo_size, samples, chi, dof, p, p > 0.05)


@dataclass
class AdaptiveSteeringResult:
    favoured_fraction: float
    mean_pow_seeds_to_hit: float
    theoretical_expectation: float
    trials: int


def adaptive_steering_cost(
    vrf: VRFBackend, secret: bytes, difficulty_bits: int, repo_size: int, favoured_fraction: float, trials: int
) -> AdaptiveSteeringResult:
    """Expected number of *paid* PoW seeds to land a task in a favoured subset.

    Each re-mine requires a fresh PoW (unique-output VRF gives one task per seed), so
    steering toward a favoured fraction φ is a geometric process with success φ per paid
    seed: E[seeds] = 1/φ. The adversary cannot do better without breaking VRF
    uniqueness. We measure the empirical mean against the 1/φ prediction.
    """
    target = difficulty_to_target(difficulty_bits)
    favoured = set(range(int(repo_size * favoured_fraction)))
    seeds_needed: list[int] = []
    for t in range(trials):
        count = 0
        while True:
            count += 1
            digest = _mine_seed("", secret, target, f"adaptive-{t}-{count}".encode(), 10_000_000)
            beta, _ = vrf.prove(secret, digest)
            if index_from_vrf_output(beta, repo_size) in favoured:
                break
        seeds_needed.append(count)
    mean = sum(seeds_needed) / len(seeds_needed)
    return AdaptiveSteeringResult(
        favoured_fraction=favoured_fraction,
        mean_pow_seeds_to_hit=mean,
        theoretical_expectation=1.0 / favoured_fraction,
        trials=trials,
    )


@dataclass
class ReseedingResult:
    favoured_fraction: float
    reseed_budget: int
    achieved_favoured_rate: float
    theoretical_favoured_rate: float
    mean_reseeds_used: float


def reseeding_bias(
    vrf: VRFBackend,
    secret: bytes,
    repo_size: int,
    favoured_fraction: float,
    reseed_budget: int,
    trials: int,
    rng,
) -> ReseedingResult:
    """A miner that discards unfavourable seeds to bias task assignment (comments 6 and 8).

    The seed is the PoW digest; a miner can mine a fresh seed and re-derive its task index.
    With a budget of ``reseed_budget`` extra seeds it accepts the first favoured index it
    sees, so the achieved favoured rate is ``1-(1-phi)^(budget+1)``. This function measures
    that rate empirically; the *cost* of each reseed (a full PoW) and the economic boundary
    are computed by the experiment as a function of the difficulty target.

    Crucially, the reseeding does not break VRF uniqueness (each seed still yields exactly
    one task); it only exploits a low PoW target that makes discarding a seed cheap. The
    bias is therefore bounded by how many PoW attempts the adversary can afford, which the
    difficulty target controls.
    """
    favoured = set(range(int(repo_size * favoured_fraction)))
    hits = 0
    reseeds_used: list[int] = []
    for t in range(trials):
        got = False
        for attempt in range(reseed_budget + 1):
            beta, _ = vrf.prove(secret, f"reseed-{t}-{attempt}".encode())
            if index_from_vrf_output(beta, repo_size) in favoured:
                got = True
                reseeds_used.append(attempt)
                break
        if got:
            hits += 1
        else:
            reseeds_used.append(reseed_budget)
    return ReseedingResult(
        favoured_fraction=favoured_fraction,
        reseed_budget=reseed_budget,
        achieved_favoured_rate=hits / trials,
        theoretical_favoured_rate=1.0 - (1.0 - favoured_fraction) ** (reseed_budget + 1),
        mean_reseeds_used=sum(reseeds_used) / len(reseeds_used),
    )
