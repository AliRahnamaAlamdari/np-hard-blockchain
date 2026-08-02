"""Tests for TSP instance generation, the three solvers, and solution verification."""

from __future__ import annotations

import math

import pytest

from pouw.tasks.repository import (
    ProblemRepository,
    RepositorySpec,
    index_from_vrf_output,
    modulo_bias,
)
from pouw.tasks.tsp import (
    TSPInstance,
    TSPSolution,
    brute_force,
    count_permutations,
    generate_instance,
    held_karp,
    is_feasible,
    nearest_neighbor_2opt,
    solve,
    tour_cost,
    verify_solution,
)

# A unit square: the optimal tour is the perimeter, cost exactly 4.
UNIT_SQUARE = TSPInstance(
    instance_id="square",
    coordinates=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
)

# A 3-4-5 right triangle: the only tour has cost 3 + 4 + 5 = 12.
TRIANGLE = TSPInstance(
    instance_id="triangle",
    coordinates=((0.0, 0.0), (3.0, 0.0), (0.0, 4.0)),
)


class TestTourCost:
    def test_square_perimeter_is_four(self):
        assert tour_cost(UNIT_SQUARE, (0, 1, 2, 3)) == pytest.approx(4.0)

    def test_crossing_tour_of_the_square_uses_two_diagonals(self):
        # 0 -> 2 -> 1 -> 3 -> 0 : sqrt2 + 1 + sqrt2 + 1
        assert tour_cost(UNIT_SQUARE, (0, 2, 1, 3)) == pytest.approx(2 + 2 * math.sqrt(2))

    def test_triangle_perimeter_is_twelve(self):
        assert tour_cost(TRIANGLE, (0, 1, 2)) == pytest.approx(12.0)

    def test_cost_is_rotation_invariant(self):
        assert tour_cost(UNIT_SQUARE, (0, 1, 2, 3)) == pytest.approx(
            tour_cost(UNIT_SQUARE, (2, 3, 0, 1))
        )

    def test_cost_is_reversal_invariant(self):
        assert tour_cost(UNIT_SQUARE, (0, 1, 2, 3)) == pytest.approx(
            tour_cost(UNIT_SQUARE, (3, 2, 1, 0))
        )


class TestFeasibility:
    def test_a_permutation_is_feasible(self):
        assert is_feasible(UNIT_SQUARE, (0, 1, 2, 3))

    def test_a_repeated_city_is_infeasible(self):
        assert not is_feasible(UNIT_SQUARE, (0, 1, 1, 3))

    def test_a_short_tour_is_infeasible(self):
        assert not is_feasible(UNIT_SQUARE, (0, 1, 2))

    def test_an_out_of_range_city_is_infeasible(self):
        assert not is_feasible(UNIT_SQUARE, (0, 1, 2, 9))


class TestBruteForce:
    def test_finds_the_known_square_optimum(self):
        solution, evaluated = brute_force(UNIT_SQUARE)
        assert solution.cost == pytest.approx(4.0)
        assert evaluated == 6  # (4-1)! = 6

    def test_finds_the_known_triangle_optimum(self):
        solution, evaluated = brute_force(TRIANGLE)
        assert solution.cost == pytest.approx(12.0)
        assert evaluated == 2  # (3-1)! = 2

    @pytest.mark.parametrize("size", [4, 5, 6, 7])
    def test_evaluates_exactly_n_minus_one_factorial_tours(self, size):
        # The paper's stated solver enumerates (N-1)! permutations (L408).
        instance = generate_instance("gen", size=size, seed=1)
        _, evaluated = brute_force(instance)
        assert evaluated == count_permutations(size)
        assert evaluated == math.factorial(size - 1)

    def test_returns_a_feasible_tour_starting_at_city_zero(self):
        instance = generate_instance("gen", size=6, seed=2)
        solution, _ = brute_force(instance)
        assert is_feasible(instance, solution.tour)
        assert solution.tour[0] == 0

    def test_reported_cost_matches_the_recomputed_cost(self):
        instance = generate_instance("gen", size=6, seed=3)
        solution, _ = brute_force(instance)
        assert solution.cost == pytest.approx(tour_cost(instance, solution.tour))


class TestHeldKarp:
    def test_finds_the_known_square_optimum(self):
        solution, _ = held_karp(UNIT_SQUARE)
        assert solution.cost == pytest.approx(4.0)

    @pytest.mark.parametrize("size", [4, 5, 6, 7, 8])
    def test_agrees_with_brute_force(self, size):
        # Both are exact, so any disagreement is a bug in one of them.
        instance = generate_instance("gen", size=size, seed=size)
        exact, _ = brute_force(instance)
        dynamic, _ = held_karp(instance)
        assert dynamic.cost == pytest.approx(exact.cost, rel=1e-12)

    def test_returns_a_feasible_tour(self):
        instance = generate_instance("gen", size=7, seed=4)
        solution, _ = held_karp(instance)
        assert is_feasible(instance, solution.tour)

    def test_uses_far_fewer_operations_than_brute_force(self):
        instance = generate_instance("gen", size=9, seed=5)
        _, brute_ops = brute_force(instance)
        _, dynamic_ops = held_karp(instance)
        assert dynamic_ops < brute_ops / 10


class TestHeuristic:
    def test_returns_a_feasible_tour(self):
        instance = generate_instance("gen", size=10, seed=6)
        solution, _ = nearest_neighbor_2opt(instance)
        assert is_feasible(instance, solution.tour)

    def test_is_never_better_than_the_exact_optimum(self):
        instance = generate_instance("gen", size=8, seed=7)
        exact, _ = brute_force(instance)
        heuristic, _ = nearest_neighbor_2opt(instance)
        assert heuristic.cost >= exact.cost - 1e-9

    def test_reported_cost_matches_the_recomputed_cost(self):
        instance = generate_instance("gen", size=9, seed=8)
        solution, _ = nearest_neighbor_2opt(instance)
        assert solution.cost == pytest.approx(tour_cost(instance, solution.tour))


class TestGeneration:
    def test_generation_is_deterministic(self):
        assert generate_instance("a", 6, 42).coordinates == generate_instance("a", 6, 42).coordinates

    def test_different_seeds_give_different_instances(self):
        assert generate_instance("a", 6, 1).coordinates != generate_instance("a", 6, 2).coordinates

    def test_coordinates_respect_the_configured_range(self):
        instance = generate_instance("a", 20, 3, coordinate_range=(-5.0, 5.0))
        for x, y in instance.coordinates:
            assert -5.0 <= x <= 5.0 and -5.0 <= y <= 5.0

    def test_payload_roundtrip_preserves_the_instance(self):
        original = generate_instance("a", 6, 9)
        assert TSPInstance.from_payload(original.to_payload()) == original

    def test_sizes_below_three_are_rejected(self):
        with pytest.raises(ValueError):
            generate_instance("a", 2, 1)


class TestVerifySolution:
    def test_accepts_a_correct_solution(self):
        solution, _ = brute_force(UNIT_SQUARE)
        assert verify_solution(UNIT_SQUARE, solution) == (True, "ok")

    def test_rejects_an_infeasible_tour(self):
        bad = TSPSolution(tour=(0, 1, 1, 3), cost=4.0)
        assert verify_solution(UNIT_SQUARE, bad)[1] == "infeasible_tour"

    def test_rejects_an_understated_cost(self):
        # A miner claiming a better objective than its tour achieves.
        bad = TSPSolution(tour=(0, 2, 1, 3), cost=4.0)
        assert verify_solution(UNIT_SQUARE, bad)[1] == "cost_mismatch"

    def test_quality_threshold_accepts_within_bound(self):
        solution = TSPSolution(tour=(0, 2, 1, 3), cost=tour_cost(UNIT_SQUARE, (0, 2, 1, 3)))
        ok, _ = verify_solution(
            UNIT_SQUARE, solution, quality_threshold=0.5, reference_cost=4.0
        )
        assert ok  # 4.83 <= 4.0 * 1.5

    def test_quality_threshold_rejects_outside_bound(self):
        solution = TSPSolution(tour=(0, 2, 1, 3), cost=tour_cost(UNIT_SQUARE, (0, 2, 1, 3)))
        ok, reason = verify_solution(
            UNIT_SQUARE, solution, quality_threshold=0.01, reference_cost=4.0
        )
        assert not ok and reason == "quality_threshold_exceeded"

    def test_threshold_without_reference_is_rejected(self):
        solution, _ = brute_force(UNIT_SQUARE)
        ok, reason = verify_solution(UNIT_SQUARE, solution, quality_threshold=0.1)
        assert not ok and reason == "missing_reference_cost"


class TestRepository:
    def test_same_spec_gives_the_same_merkle_root(self):
        spec = RepositorySpec(num_instances=8, instance_size=5, seed=1, coordinate_range=(0.0, 100.0))
        assert ProblemRepository.from_spec(spec).root == ProblemRepository.from_spec(spec).root

    def test_different_seed_gives_a_different_root(self):
        base = RepositorySpec(num_instances=8, instance_size=5, seed=1, coordinate_range=(0.0, 100.0))
        other = RepositorySpec(num_instances=8, instance_size=5, seed=2, coordinate_range=(0.0, 100.0))
        assert ProblemRepository.from_spec(base).root != ProblemRepository.from_spec(other).root

    def test_every_instance_proves_membership(self, small_repository):
        for index in range(len(small_repository)):
            instance, proof = small_repository.get_proof(index)
            assert small_repository.verify(instance, proof)

    def test_a_foreign_instance_fails_membership(self, small_repository):
        _, proof = small_repository.get_proof(0)
        forged = generate_instance("FORGED", 5, 999)
        assert not small_repository.verify(forged, proof)

    def test_save_and_load_roundtrip(self, small_repository, tmp_path):
        path = tmp_path / "repo.json"
        small_repository.save(path)
        assert ProblemRepository.load(path).root == small_repository.root


class TestIndexDerivation:
    def test_index_is_within_range(self):
        for value in range(64):
            index = index_from_vrf_output(value.to_bytes(32, "big"), 10)
            assert 0 <= index < 10

    def test_index_matches_the_equation(self):
        # Equation (3): i = Integer(y) mod N.
        assert index_from_vrf_output((1234).to_bytes(32, "big"), 100) == 34

    def test_zero_repository_size_is_rejected(self):
        with pytest.raises(ValueError):
            index_from_vrf_output(b"\x00" * 32, 0)

    def test_power_of_two_repository_has_no_modulo_bias(self):
        assert modulo_bias(256, 256) == 0.0
        assert modulo_bias(32, 1024) == 0.0

    def test_non_power_of_two_bias_is_negligible_at_256_bits(self):
        assert modulo_bias(256, 1000) < 1e-70

    def test_bias_becomes_measurable_with_a_truncated_output(self):
        # The paper never states the VRF output width; a narrow one would bias
        # Equation (3) measurably.
        assert modulo_bias(16, 50000) > 0.01


def test_solver_dispatch_covers_every_registered_solver():
    instance = generate_instance("gen", 6, 12)
    for name in ("brute_force", "held_karp", "nearest_neighbor_2opt"):
        solution, _ = solve(instance, name)
        assert is_feasible(instance, solution.tour)


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError):
        solve(UNIT_SQUARE, "no_such_solver")
