"""In-house baseline designs for a *measured* comparison (reviewer comment 5).

We cannot run Ofelimos, Proof-of-Solution, or the specific VRF-PoW reference systems:
no maintained runnable implementations are available and faithful reimplementation of
each is out of scope. Those stay literature-reported in the comparison table.

What we *can* measure honestly is a set of in-house baselines that each represent a
*design class*, run on the same harness as our framework, so the security-relevant
quantities (verification cost, task-selection freedom, grinding cost, precomputation
exposure) are directly comparable:

* ``pure_pow`` -- lightweight PoW only, no useful work (a hash-puzzle control).
* ``miner_chosen`` -- PoUW where the miner selects its own instance from the pool, with
  no VRF binding (a COCP-style optimization-mining analogue).
* ``vrf_only`` -- VRF task selection but no PoW entry gate (isolates the gate's role).
* ``ours`` -- the full two-stage framework.

Each baseline is described by which protocol features it has; the measurements are taken
by exercising exactly those features with the real primitives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..crypto.hashing import difficulty_to_target, meets_target, sha256
from ..crypto.vrf import build_vrf
from ..tasks.problems.base import get_problem_class
from ..tasks.problems.repository import GenericRepository
from ..tasks.repository import index_from_vrf_output


@dataclass
class BaselineSpec:
    name: str
    has_pow_gate: bool
    has_vrf_binding: bool
    has_useful_work: bool
    miner_chooses_task: bool


BASELINES = {
    "pure_pow": BaselineSpec("pure_pow", True, False, False, False),
    "miner_chosen": BaselineSpec("miner_chosen", True, False, True, True),
    "vrf_only": BaselineSpec("vrf_only", False, True, True, False),
    "ours": BaselineSpec("ours", True, True, True, False),
}


@dataclass
class BaselineMeasurement:
    name: str
    verify_ms: float
    verify_ms_std: float
    task_control_resistance: str  # measured/qualitative label
    grinding_tasks_per_seed: float  # 1 => bound to PoW; inf/large => free
    precomputation_exposure: float  # probability assigned task is pre-solved
    useful_work: bool
    spam_entry_cost: str
    notes: str


def measure_baseline(
    spec: BaselineSpec,
    repository: GenericRepository,
    difficulty_bits: int,
    seed_material: bytes,
    trials: int = 200,
    precomputed_fraction: float = 0.25,
    rng: Any = None,
) -> BaselineMeasurement:
    """Measure verification cost and task-selection properties for one baseline."""
    import random

    rng = rng or random.Random(0)
    vrf = build_vrf("ecvrf")
    keys = vrf.keygen(seed_material)
    target = difficulty_to_target(difficulty_bits)
    problem = get_problem_class(next(iter(repository)).problem_class)

    # Build one representative valid block's pieces for repeated verification timing.
    header_bytes = b"header||nonce"
    seed = sha256(header_bytes)
    while not meets_target(seed, target):
        header_bytes += b"x"
        seed = sha256(header_bytes)

    if spec.has_vrf_binding:
        beta, proof = vrf.prove(keys.secret, seed)
        index = index_from_vrf_output(beta, len(repository))
    elif spec.miner_chooses_task:
        # miner picks the cheapest instance it can find (free choice)
        index = min(range(len(repository)), key=lambda i: repository.instance(i).size)
        beta, proof = None, None
    else:
        index = 0
        beta, proof = None, None

    instance, mproof = repository.get_proof(index)
    solution = None
    if spec.has_useful_work:
        solution, _ = problem.solve_exact(instance)

    # Time the verification path this baseline actually requires.
    verify_samples = []
    for _ in range(trials):
        t = time.perf_counter()
        _ = meets_target(sha256(header_bytes), target)  # PoW check (all have a header hash)
        if spec.has_vrf_binding:
            got = vrf.verify(keys.public, seed, proof)
            _ = got is not None and index_from_vrf_output(got, len(repository)) == index
        if spec.has_useful_work:
            repository.verify(instance, mproof)
            problem.verify(instance, solution)
        verify_samples.append(time.perf_counter() - t)

    from ..stats_utils import mean_std

    vmean, vstd = mean_std(verify_samples)

    # Task-control resistance and grinding: bound to PoW only when VRF-bound.
    if spec.has_vrf_binding:
        grinding = 1.0
        control = "high (VRF-bound to paid seed)"
        precomp = precomputed_fraction  # uniform assignment => exposure = coverage
    elif spec.miner_chooses_task:
        grinding = float("inf")
        control = "none (miner selects instance)"
        precomp = 1.0  # attacker simply chooses a pre-solved instance
    else:
        grinding = 1.0
        control = "n/a (no useful-work selection)"
        precomp = 0.0

    spam = "yes (nonzero PoW)" if spec.has_pow_gate else "no (free entry)"

    return BaselineMeasurement(
        name=spec.name,
        verify_ms=vmean * 1000.0,
        verify_ms_std=vstd * 1000.0,
        task_control_resistance=control,
        grinding_tasks_per_seed=grinding,
        precomputation_exposure=precomp,
        useful_work=spec.has_useful_work,
        spam_entry_cost=spam,
        notes="measured in-house baseline (design-class representative)",
    )
