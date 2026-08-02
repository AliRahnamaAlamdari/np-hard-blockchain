"""Deterministic seeding.

Every stochastic component draws from a seed derived from a single run seed and a
textual label, so components are independent of one another and of call order.  Two
runs with the same run seed produce bit-identical results; two components within a run
never share a stream.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

MASK_64 = (1 << 64) - 1


def derive_seed(run_seed: int, label: str) -> int:
    """Derive a stable 64-bit sub-seed for ``label`` under ``run_seed``."""
    digest = hashlib.sha256(f"{run_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & MASK_64


@dataclass
class SeedBundle:
    """The random sources for one run, all derived from one integer."""

    run_seed: int

    def python_random(self, label: str) -> random.Random:
        return random.Random(derive_seed(self.run_seed, label))

    def numpy_generator(self, label: str) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.run_seed, label))

    def bytes_for(self, label: str, num_bytes: int = 32) -> bytes:
        """Deterministic key material -- used for miner key generation."""
        return hashlib.sha256(
            f"{self.run_seed}:{label}".encode("utf-8")
        ).digest()[:num_bytes]

    def integer(self, label: str) -> int:
        return derive_seed(self.run_seed, label)


def seed_everything(run_seed: int) -> SeedBundle:
    """Seed the global interpreter RNGs and return the per-run bundle.

    Globals are seeded so that any third-party code picking up ``random`` or
    ``numpy.random`` is deterministic too; first-party code always uses the bundle.
    """
    random.seed(derive_seed(run_seed, "global:python"))
    np.random.seed(derive_seed(run_seed, "global:numpy") % (2**32))
    return SeedBundle(run_seed=run_seed)
