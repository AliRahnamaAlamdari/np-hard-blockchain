"""Tests for the solver backends, the growth fits, and the committed quality bound.

These cover the three things the extended problem-size sweep rests on: that every
backend is interchangeable behind one interface without changing block validity,
that the fits used for extrapolation recover a growth law they are given, and that
the quality threshold is enforced against a bound the miner cannot choose.
"""

from __future__ import annotations

import math

import pytest

from pouw.chain.miner import Miner
from pouw.chain.validator import RejectReason, Validator
from pouw.config import ExperimentConfig, RepositoryConfig, SolverConfig, validate
from pouw.crypto.hashing import difficulty_to_target
from pouw.crypto.vrf import build_vrf
from pouw.metrics.scaling import (
    FitError,
    crossover_size,
    fit_growth,
    fit_linear,
)
from pouw.seeding import seed_everything
from pouw.tasks.repository import (
    ProblemRepository,
    RepositorySpec,
    reference_cost,
)
from pouw.tasks.tsp import (
    EXACT_SOLVERS,
    SOLVERS,
    TSPInstance,
    TSPSolution,
    brute_force,
    generate_instance,
    held_karp,
    is_feasible,
    nearest_neighbor,
    or_opt,
    solve,
    tour_cost,
    verify_solution,
)

HEURISTICS = ["nearest_neighbor", "nearest_neighbor_2opt", "or_opt"]


class TestExactSolversAgree:
    """The claim the extended sweep depends on: both exact solvers, one optimum."""

    @pytest.mark.parametrize("size", [4, 5, 6, 7, 8, 9, 10])
    def test_held_karp_matches_exhaustive_on_shared_instances(self, size):
        instance = generate_instance("shared", size=size, seed=1000 + size)
        exhaustive, _ = brute_force(instance)
        dynamic, _ = held_karp(instance)
        assert dynamic.cost == pytest.approx(exhaustive.cost, rel=1e-12)

    def test_they_agree_on_every_instance_of_a_repository(self, small_repository):
        # The sweep draws its instances through the VRF, so agreement has to hold
        # across the repository rather than on one hand-picked instance.
        for instance in small_repository:
            exhaustive, _ = brute_force(instance)
            dynamic, _ = held_karp(instance)
            assert dynamic.cost == pytest.approx(exhaustive.cost, rel=1e-12)

    def test_both_are_registered_as_exact(self):
        assert EXACT_SOLVERS == {"brute_force", "held_karp"}


class TestHeuristicLadder:
    @pytest.mark.parametrize("name", HEURISTICS)
    def test_returns_a_feasible_tour(self, name):
        instance = generate_instance("h", size=30, seed=3)
        solution, _ = solve(instance, name)
        assert is_feasible(instance, solution.tour)

    @pytest.mark.parametrize("name", HEURISTICS)
    def test_reported_cost_matches_the_recomputed_cost(self, name):
        instance = generate_instance("h", size=25, seed=4)
        solution, _ = solve(instance, name)
        assert solution.cost == pytest.approx(tour_cost(instance, solution.tour))

    @pytest.mark.parametrize("name", HEURISTICS)
    def test_is_never_better_than_the_exact_optimum(self, name):
        instance = generate_instance("h", size=9, seed=5)
        exact, _ = held_karp(instance)
        heuristic, _ = solve(instance, name)
        assert heuristic.cost >= exact.cost - 1e-9

    @pytest.mark.parametrize("name", HEURISTICS)
    def test_starts_at_city_zero_like_every_other_solver(self, name):
        instance = generate_instance("h", size=20, seed=6)
        solution, _ = solve(instance, name)
        assert solution.tour[0] == 0

    @pytest.mark.parametrize("name", HEURISTICS)
    def test_is_deterministic(self, name):
        instance = generate_instance("h", size=40, seed=7)
        first, _ = solve(instance, name)
        second, _ = solve(instance, name)
        assert first.tour == second.tour
        assert first.cost == second.cost

    def test_quality_increases_up_the_ladder(self):
        # Not guaranteed instance by instance, but the ordering has to hold on
        # average or the ladder does not exercise the threshold as intended.
        totals = {name: 0.0 for name in HEURISTICS}
        for seed in range(8):
            instance = generate_instance("ladder", size=60, seed=seed)
            for name in HEURISTICS:
                totals[name] += solve(instance, name)[0].cost
        assert totals["or_opt"] <= totals["nearest_neighbor_2opt"]
        assert totals["nearest_neighbor_2opt"] < totals["nearest_neighbor"]

    def test_or_opt_reaches_sizes_no_exact_solver_can(self):
        instance = generate_instance("big", size=300, seed=8)
        solution, _ = or_opt(instance)
        assert is_feasible(instance, solution.tour)


def test_solver_registry_covers_every_backend():
    assert set(SOLVERS) == {
        "brute_force",
        "held_karp",
        "nearest_neighbor",
        "nearest_neighbor_2opt",
        "or_opt",
    }


class TestGrowthFit:
    def test_recovers_a_synthetic_growth_law(self):
        # t = 3e-9 * (N-1)! exactly, so the fit must return exponent 1 and R^2 1.
        sizes = [8, 9, 10, 11, 12]
        times = [3e-9 * math.factorial(size - 1) for size in sizes]
        fit = fit_growth("brute_force", sizes, times, min_seconds=0.0)
        assert fit.exponent == pytest.approx(1.0, abs=1e-9)
        assert fit.coefficient == pytest.approx(3e-9, rel=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
        assert fit.max_abs_percent_error < 1e-6

    def test_predicts_a_point_it_was_not_given(self):
        sizes = [8, 9, 10, 11]
        times = [2e-9 * math.factorial(size - 1) for size in sizes]
        fit = fit_growth("brute_force", sizes, times, min_seconds=0.0)
        assert fit.predict(13) == pytest.approx(2e-9 * math.factorial(12), rel=1e-6)

    def test_names_the_points_the_floor_excluded(self):
        sizes = [4, 6, 8, 10, 11]
        times = [1e-6, 1e-5, 2e-3, 1.5e-1, 1.6]
        fit = fit_growth("brute_force", sizes, times, min_seconds=1e-3)
        assert fit.sizes_excluded == [4, 6]
        assert fit.sizes_fitted == [8, 10, 11]

    def test_residuals_cover_every_fitted_point(self):
        sizes = [10, 12, 14, 16]
        times = [1e-8 * (2**n) * n * n for n in sizes]
        fit = fit_growth("held_karp", sizes, times, min_seconds=0.0)
        assert [r.size for r in fit.residuals] == sizes

    def test_too_few_points_is_an_error_not_a_guess(self):
        with pytest.raises(FitError):
            fit_growth("brute_force", [10, 11], [0.1, 1.0], min_seconds=0.0)

    def test_unknown_solver_has_no_growth_model(self):
        with pytest.raises(FitError):
            fit_growth("nonesuch", [8, 9, 10], [1.0, 2.0, 3.0])


class TestVerificationModel:
    def test_linear_fit_recovers_intercept_and_slope(self):
        sizes = [10, 50, 100, 200]
        seconds = [7e-3 + 1e-6 * size for size in sizes]
        fit = fit_linear(sizes, seconds)
        assert fit.intercept == pytest.approx(7e-3, rel=1e-9)
        assert fit.slope == pytest.approx(1e-6, rel=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-12)

    def test_one_distinct_size_is_an_error(self):
        with pytest.raises(FitError):
            fit_linear([10, 10], [1.0, 1.0])


class TestCrossover:
    def test_measured_crossover_is_the_first_size_where_solving_wins(self):
        sizes = [8, 9, 10, 11]
        solve_seconds = [0.002, 0.016, 0.16, 1.7]
        verify_seconds = [0.007] * 4
        fit = fit_growth("brute_force", sizes, solve_seconds, min_seconds=1e-3)
        crossover = crossover_size(
            fit, 0.007, sizes=sizes, mean_solve_seconds=solve_seconds,
            mean_verify_seconds=verify_seconds,
        )
        # N=9 is the first measured size whose solve time reaches the 7 ms verify.
        assert crossover.measured_size == 9
        assert crossover.measured_bracket == (8, 9)

    def test_a_size_dependent_verification_cost_moves_the_crossover(self):
        sizes = [50, 100, 200, 400]
        solve_seconds = [1e-7 * size * size for size in sizes]
        flat = crossover_size(
            fit_growth("nearest_neighbor_2opt", sizes, solve_seconds, min_seconds=0.0),
            0.007,
        )
        rising = crossover_size(
            fit_growth("nearest_neighbor_2opt", sizes, solve_seconds, min_seconds=0.0),
            fit_linear(sizes, [0.007 + 5e-5 * size for size in sizes]),
        )
        assert rising.fitted_size > flat.fitted_size


class TestCommittedQualityBound:
    @pytest.fixture
    def bounded_repository(self):
        return ProblemRepository.from_spec(
            RepositorySpec(
                num_instances=4,
                instance_size=30,
                seed=99,
                coordinate_range=(0.0, 100.0),
                reference_solver="or_opt",
            )
        )

    def test_the_bound_is_published_in_the_instance_metadata(self, bounded_repository):
        instance = bounded_repository.instance(0)
        assert reference_cost(instance) == pytest.approx(
            or_opt(instance)[0].cost
        )
        assert instance.metadata["reference_solver"] == "or_opt"

    def test_a_repository_without_a_reference_solver_publishes_no_bound(
        self, small_repository
    ):
        assert reference_cost(small_repository.instance(0)) is None

    def test_raising_the_bound_breaks_the_membership_proof(self, bounded_repository):
        instance, proof = bounded_repository.get_proof(0)
        forged = TSPInstance(
            instance_id=instance.instance_id,
            coordinates=instance.coordinates,
            metadata={**instance.metadata, "reference_cost": 1e12},
        )
        assert bounded_repository.verify(instance, proof)
        assert not bounded_repository.verify(forged, proof)

    def test_the_threshold_rejects_a_tour_above_the_bound(self, bounded_repository):
        instance = bounded_repository.instance(0)
        bound = reference_cost(instance)
        weak, _ = nearest_neighbor(instance)
        assert weak.cost > bound  # the ladder has to bind for this test to mean anything
        ok, reason = verify_solution(
            instance, weak, quality_threshold=0.0, reference_cost=bound
        )
        assert not ok
        assert reason == "quality_threshold_exceeded"

    def test_the_threshold_accepts_the_same_tour_when_widened(self, bounded_repository):
        instance = bounded_repository.instance(0)
        bound = reference_cost(instance)
        weak, _ = nearest_neighbor(instance)
        widened = (weak.cost / bound - 1.0) + 0.01
        ok, reason = verify_solution(
            instance, weak, quality_threshold=widened, reference_cost=bound
        )
        assert ok and reason == "ok"


class TestValidityIsSolverIndependent:
    """A block is valid on witness and cost alone, whichever solver produced it."""

    def _mine_and_validate(self, solver, quality_threshold, reference_solver):
        bundle = seed_everything(3)
        repository = ProblemRepository.from_spec(
            RepositorySpec(
                num_instances=8,
                instance_size=9,
                seed=5,
                coordinate_range=(0.0, 100.0),
                reference_solver=reference_solver,
            )
        )
        vrf = build_vrf("ecvrf")
        target = difficulty_to_target(6)
        miner = Miner(miner_id=0, keys=vrf.keygen(bundle.bytes_for("m")), vrf=vrf)
        block, _ = miner.mine(
            height=1,
            previous_hash="parent",
            tx_root="tx",
            target=target,
            repository=repository,
            solver=solver,
            max_attempts=1_000_000,
            quality_threshold=quality_threshold,
        )
        validator = Validator(
            validator_id=0,
            vrf=vrf,
            repository=repository,
            target=target,
            quality_threshold=quality_threshold,
        )
        return block, validator.validate(block)

    @pytest.mark.parametrize("solver", sorted(SOLVERS))
    def test_every_solver_produces_a_valid_block_without_a_threshold(self, solver):
        _, result = self._mine_and_validate(solver, None, None)
        assert result.accepted, result.reason

    @pytest.mark.parametrize("solver", ["brute_force", "held_karp", "or_opt"])
    def test_both_exact_solvers_meet_a_zero_threshold(self, solver):
        # The bound is the exact optimum here, so a zero tolerance accepts exactly
        # the solvers that reach it -- and or_opt does on this instance size.
        _, result = self._mine_and_validate(solver, 0.0, "held_karp")
        assert result.accepted, result.reason

    def test_a_weak_solver_is_rejected_against_a_zero_threshold(self):
        block, result = self._mine_and_validate(
            "nearest_neighbor", 0.0, "held_karp"
        )
        optimum = held_karp(block.instance)[0].cost
        if block.solution.cost <= optimum + 1e-9:
            pytest.skip("nearest neighbour happened to be optimal on this instance")
        assert not result.accepted
        assert result.reason is RejectReason.SOLUTION_INVALID

    def test_the_miner_reports_whether_it_met_the_bound(self):
        bundle = seed_everything(3)
        repository = ProblemRepository.from_spec(
            RepositorySpec(
                num_instances=8,
                instance_size=9,
                seed=5,
                coordinate_range=(0.0, 100.0),
                reference_solver="held_karp",
            )
        )
        vrf = build_vrf("ecvrf")
        miner = Miner(miner_id=0, keys=vrf.keygen(bundle.bytes_for("m")), vrf=vrf)
        _, timings = miner.mine(
            height=1,
            previous_hash="parent",
            tx_root="tx",
            target=difficulty_to_target(6),
            repository=repository,
            solver="held_karp",
            max_attempts=1_000_000,
            quality_threshold=0.0,
        )
        # Missing the bound is a protocol outcome, not a crash: the miner records it.
        assert timings["meets_quality_threshold"] is True
        assert timings["reference_cost"] is not None


class TestConfigGuardsTheBound:
    def test_a_threshold_without_a_published_bound_is_rejected(self):
        config = ExperimentConfig(
            solver=SolverConfig(quality_threshold=0.1),
            repository=RepositoryConfig(reference_solver=None),
        )
        with pytest.raises(ValueError, match="reference_solver"):
            validate(config)

    def test_an_unknown_reference_solver_is_rejected(self):
        config = ExperimentConfig(repository=RepositoryConfig(reference_solver="magic"))
        with pytest.raises(ValueError, match="reference_solver"):
            validate(config)

    def test_a_threshold_with_a_published_bound_is_accepted(self):
        config = ExperimentConfig(
            solver=SolverConfig(quality_threshold=0.1),
            repository=RepositoryConfig(reference_solver="or_opt"),
        )
        validate(config)


class TestUnresolvedVerificationSlope:
    """A slope below the noise floor must not become an extrapolation model."""

    def test_a_negative_slope_falls_back_to_the_flat_mean(self):
        # Verification cost that is flat plus noise: the least-squares slope can
        # come out negative, which would predict negative time at large N.
        sizes = [10, 50, 100, 400, 800]
        seconds = [0.0079, 0.0081, 0.0078, 0.0079, 0.0077]
        fit = fit_linear(sizes, seconds)
        assert not fit.slope_is_resolved
        assert fit.predict(5000) == pytest.approx(fit.mean_seconds)
        assert fit.predict(5000) > 0.0
        assert fit.to_dict()["model_used"] == "flat_mean"

    def test_a_resolved_slope_is_used(self):
        sizes = [10, 50, 100, 400, 800]
        seconds = [0.007 + 1e-5 * size for size in sizes]
        fit = fit_linear(sizes, seconds)
        assert fit.slope_is_resolved
        assert fit.predict(1000) == pytest.approx(0.007 + 1e-5 * 1000, rel=1e-9)
        assert fit.to_dict()["model_used"] == "linear"
