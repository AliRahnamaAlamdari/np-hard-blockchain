"""Shared pytest fixtures and the ``src`` path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pouw.crypto.vrf import build_vrf  # noqa: E402
from pouw.seeding import seed_everything  # noqa: E402
from pouw.tasks.repository import ProblemRepository, RepositorySpec  # noqa: E402
from pouw.tasks.tsp import generate_instance  # noqa: E402


@pytest.fixture
def bundle():
    return seed_everything(1234)


@pytest.fixture
def small_repository() -> ProblemRepository:
    return ProblemRepository.from_spec(
        RepositorySpec(
            num_instances=8,
            instance_size=5,
            seed=7,
            coordinate_range=(0.0, 100.0),
        )
    )


@pytest.fixture
def tiny_instance():
    return generate_instance("T-5", size=5, seed=11)


@pytest.fixture(params=["ecvrf", "ecdsa_pseudo", "hash_stub"])
def any_vrf(request):
    return build_vrf(request.param)


@pytest.fixture
def real_vrf():
    return build_vrf("ecvrf")
