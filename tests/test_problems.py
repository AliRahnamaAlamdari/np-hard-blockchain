"""Tests for the pluggable problem classes and the class-agnostic framework runner."""

from __future__ import annotations

import copy
import random

import pytest

from pouw.chain.generic_runner import build_runner
from pouw.tasks.problems import available_classes, get_problem_class
from pouw.tasks.problems.base import ProblemInstance, Solution
from pouw.tasks.problems.repository import GenericRepository, GenericRepositorySpec

CLASSES = ["tsp", "knapsack", "graph_coloring", "sat"]
PARAMS = {
    "tsp": {"coordinate_range": (0.0, 100.0)},
    "knapsack": {},
    "graph_coloring": {"edge_probability": 0.5},
    "sat": {"clause_ratio": 4.0},
}


def test_all_expected_classes_registered():
    assert set(available_classes()) == set(CLASSES)


@pytest.mark.parametrize("name", CLASSES)
class TestProblemClass:
    def test_generation_is_deterministic(self, name):
        pc = get_problem_class(name)
        a = pc.generate("x", 8, 42, **PARAMS[name])
        b = pc.generate("x", 8, 42, **PARAMS[name])
        assert a.to_payload() == b.to_payload()

    def test_exact_and_reference_agree_on_optimum(self, name):
        pc = get_problem_class(name)
        for seed in range(4):
            inst = pc.generate(f"{name}-{seed}", 8, seed, **PARAMS[name])
            exact, _ = pc.solve_exact(inst)
            ref, _ = pc.solve_reference(inst)
            assert exact.cost == pytest.approx(ref.cost, abs=1e-6), (
                f"{name} seed {seed}: exact {exact.cost} != reference {ref.cost}"
            )

    def test_honest_solution_verifies(self, name):
        pc = get_problem_class(name)
        inst = pc.generate(f"{name}-v", 9, 1, **PARAMS[name])
        sol, _ = pc.solve_exact(inst)
        ok, reason = pc.verify(inst, sol)
        assert ok, f"{name}: honest solution rejected ({reason})"

    def test_payload_roundtrip(self, name):
        pc = get_problem_class(name)
        inst = pc.generate(f"{name}-r", 8, 3, **PARAMS[name])
        sol, _ = pc.solve_exact(inst)
        inst2 = ProblemInstance.from_payload(inst.to_payload())
        sol2 = Solution.from_payload(sol.to_payload())
        ok, _ = pc.verify(inst2, sol2)
        assert ok

    def test_understated_cost_is_rejected(self, name):
        pc = get_problem_class(name)
        inst = pc.generate(f"{name}-c", 8, 5, **PARAMS[name])
        sol, _ = pc.solve_exact(inst)
        tampered = Solution(sol.problem_class, sol.witness, sol.cost - 5.0, sol.feasible)
        ok, _ = pc.verify(inst, tampered)
        assert not ok


class TestConstraintRejection:
    """Hard-constraint classes reject an infeasible witness on feasibility alone;
    optimization classes need a quality threshold. This is the framework-vs-case-study
    distinction, so it is tested as a property, not assumed."""

    @pytest.mark.parametrize("name", ["graph_coloring", "sat"])
    def test_hard_constraint_class_rejects_random_witness(self, name):
        pc = get_problem_class(name)
        assert pc.constraint_rejects_random
        inst = pc.generate(f"{name}-rw", 9, 7, **PARAMS[name])
        rng = random.Random(0)
        rejected = sum(not pc.verify(inst, pc.random_witness(inst, rng))[0] for _ in range(10))
        assert rejected >= 8, f"{name}: random witnesses too often accepted"

    @pytest.mark.parametrize("name", ["tsp", "knapsack"])
    def test_optimization_class_needs_threshold_to_reject_suboptimal(self, name):
        pc = get_problem_class(name)
        assert not pc.constraint_rejects_random
        inst = pc.generate(f"{name}-opt", 9, 7, **PARAMS[name])
        exact, _ = pc.solve_exact(inst)
        # Without a threshold, a feasible random witness is accepted; with a strict
        # threshold, a suboptimal one is rejected.
        rng = random.Random(1)
        rw = pc.random_witness(inst, rng)
        if pc.verify(inst, rw)[0] and abs(rw.cost - exact.cost) > 1e-6:
            ok_thr, _ = pc.verify(inst, rw, quality_threshold=0.0, reference_cost=exact.cost)
            assert not ok_thr, f"{name}: threshold failed to reject a suboptimal witness"


class TestGenericRepository:
    @pytest.mark.parametrize("name", CLASSES)
    def test_membership_proofs_verify(self, name):
        repo = GenericRepository.from_spec(
            GenericRepositorySpec(name, num_instances=8, instance_size=7, seed=1, params=PARAMS[name])
        )
        for i in range(len(repo)):
            inst, proof = repo.get_proof(i)
            assert repo.verify(inst, proof)

    def test_same_spec_same_root(self):
        spec = GenericRepositorySpec("knapsack", 8, 7, 1, {})
        assert GenericRepository.from_spec(spec).root == GenericRepository.from_spec(spec).root


class TestGenericRunner:
    @pytest.mark.parametrize("name", CLASSES)
    def test_full_cycle_accepts_honest_block(self, name):
        repo = GenericRepository.from_spec(
            GenericRepositorySpec(name, num_instances=16, instance_size=8, seed=2, params=PARAMS[name])
        )
        runner = build_runner(name, repo, difficulty_bits=6, seed_material=b"m")
        block = runner.mine(1, "genesis", "tx")
        ok, reason, steps = runner.validate(block)
        assert ok, f"{name}: honest block rejected ({reason})"
        assert set(steps) == {"pow_check", "vrf_verify", "index_check", "merkle_verify", "solution_check"}

    @pytest.mark.parametrize("name", ["graph_coloring", "sat"])
    def test_full_cycle_rejects_tampered_solution(self, name):
        repo = GenericRepository.from_spec(
            GenericRepositorySpec(name, num_instances=16, instance_size=8, seed=2, params=PARAMS[name])
        )
        runner = build_runner(name, repo, difficulty_bits=6, seed_material=b"m")
        block = runner.mine(1, "genesis", "tx")
        bad = copy.copy(block)
        bad.solution = get_problem_class(name).random_witness(block.instance, random.Random(3))
        ok, reason, _ = runner.validate(bad)
        assert not ok and "solution_invalid" in reason

    @pytest.mark.parametrize("name", CLASSES)
    def test_wrong_vrf_key_is_rejected(self, name):
        repo = GenericRepository.from_spec(
            GenericRepositorySpec(name, num_instances=16, instance_size=8, seed=2, params=PARAMS[name])
        )
        runner = build_runner(name, repo, difficulty_bits=6, seed_material=b"m")
        block = runner.mine(1, "genesis", "tx")
        # forge the public key in the header
        bad = copy.copy(block)
        bad.header = block.header.__class__(
            block.header.height, block.header.previous_hash, block.header.tx_root,
            block.header.miner_id, "00" * 32, block.header.nonce,
        )
        ok, reason, _ = runner.validate(bad)
        assert not ok
