"""Metric tests against hand-computed values and against the paper's own numbers.

The fairness block from the paper's Table 5 is fully self-consistent: its reported
standard deviation, max/min ratio and Jain index are exactly recoverable from the
counts.  Those reported values are used here as ground truth, which pins our
implementations to the paper's definitions rather than to our own reading of them.
"""

from __future__ import annotations

import math

import pytest

from pouw.metrics.convergence import analyse_convergence, analyse_correctness
from pouw.metrics.fairness import (
    analyse,
    chi_square_uniformity,
    gini,
    jain_index,
    max_min_ratio,
    std_dev,
)
from pouw.metrics.timing import compute_asymmetry, compute_shares

# Paper Table 5 (tab:fairness), L497-501.
PAPER_COUNTS = [101, 92, 105, 91, 111]
PAPER_STD_DEV = 7.6420
PAPER_MAX_MIN = 1.2198
PAPER_JAIN = 0.9942


class TestFairnessAgainstPaper:
    """Every statistic the paper reports for Table 5, recomputed."""

    def test_counts_sum_to_reported_total(self):
        assert sum(PAPER_COUNTS) == 500

    def test_std_dev_matches_paper(self):
        # Population std: deviations (1, -8, 5, -9, 11), sum of squares 292,
        # variance 292/5 = 58.4, sqrt(58.4) = 7.641989...
        assert std_dev(PAPER_COUNTS, ddof=0) == pytest.approx(PAPER_STD_DEV, abs=5e-5)

    def test_sample_std_dev_is_not_what_the_paper_reported(self):
        # Documents the convention: ddof=1 gives 8.5440, which is not the paper's value.
        assert std_dev(PAPER_COUNTS, ddof=1) == pytest.approx(8.5440, abs=5e-5)

    def test_max_min_ratio_matches_paper(self):
        assert max_min_ratio(PAPER_COUNTS) == pytest.approx(PAPER_MAX_MIN, abs=5e-5)
        assert max_min_ratio(PAPER_COUNTS) == pytest.approx(111 / 91)

    def test_jain_index_matches_paper(self):
        # (500)^2 / (5 * 50292) = 250000 / 251460
        assert jain_index(PAPER_COUNTS) == pytest.approx(PAPER_JAIN, abs=5e-5)
        assert jain_index(PAPER_COUNTS) == pytest.approx(250000 / 251460)

    def test_chi_square_is_hand_computable(self):
        # sum (O-E)^2 / E = 292 / 100 = 2.92, dof = 4
        statistic, dof, p_value = chi_square_uniformity(PAPER_COUNTS)
        assert statistic == pytest.approx(2.92)
        assert dof == 4
        assert p_value == pytest.approx(0.5712, abs=1e-3)

    def test_analyse_reproduces_all_reported_statistics(self):
        report = analyse(PAPER_COUNTS)
        assert report.total_blocks == 500
        assert report.expected_per_miner == pytest.approx(100.0)
        assert report.std_dev == pytest.approx(PAPER_STD_DEV, abs=5e-5)
        assert report.max_min_ratio == pytest.approx(PAPER_MAX_MIN, abs=5e-5)
        assert report.jain_index == pytest.approx(PAPER_JAIN, abs=5e-5)


class TestFairnessEdgeCases:
    def test_perfectly_equal_gives_jain_one(self):
        assert jain_index([100, 100, 100, 100]) == pytest.approx(1.0)

    def test_total_concentration_gives_jain_one_over_n(self):
        assert jain_index([500, 0, 0, 0, 0]) == pytest.approx(1 / 5)

    def test_max_min_ratio_is_infinite_when_a_miner_wins_nothing(self):
        assert max_min_ratio([10, 0, 5]) == math.inf

    def test_gini_is_zero_for_equal_shares(self):
        assert gini([25, 25, 25, 25]) == pytest.approx(0.0, abs=1e-12)

    def test_gini_approaches_one_for_full_concentration(self):
        assert gini([100, 0, 0, 0, 0]) == pytest.approx(0.8)

    def test_chi_square_rejects_a_clearly_biased_distribution(self):
        _, _, p_value = chi_square_uniformity([300, 50, 50, 50, 50])
        assert p_value < 0.001


class TestTimingShares:
    def test_shares_match_hand_computation(self):
        # 1 s PoW and 9 s useful => 10% / 90%.
        report = compute_shares([1.0], [9.0])
        assert report.useful_share_percent == pytest.approx(90.0)
        assert report.pow_share_percent == pytest.approx(10.0)

    def test_shares_reproduce_the_papers_low_tier_row(self):
        # Paper Table 3 Low: PoW 0.1660 s, useful 1.6009 s -> 9.40 / 90.60.
        # Exact value is 9.39499%, which the paper rounds to 9.40 at two decimals.
        report = compute_shares([0.1660], [1.6009])
        assert report.pow_share_percent == pytest.approx(9.395, abs=1e-3)
        assert round(report.pow_share_percent, 2) == pytest.approx(9.40, abs=0.011)
        assert report.useful_share_percent == pytest.approx(90.605, abs=1e-3)

    def test_shares_reproduce_the_papers_high_tier_row(self):
        report = compute_shares([67.4985], [1.5932])
        assert report.pow_share_percent == pytest.approx(97.69, abs=0.005)
        assert report.useful_share_percent == pytest.approx(2.31, abs=0.005)

    def test_ratio_of_means_and_mean_of_ratios_differ(self):
        # The paper does not say which it uses (PAPER_SPEC.md 11-U12).
        # Ratio of sums: 100 useful / 201 total = 49.751%.
        # Mean of per-block ratios: (99% + ~0.99%) / 2 = 49.995%.
        report = compute_shares([1.0, 100.0], [99.0, 1.0])
        assert report.useful_share_percent == pytest.approx(100 / 201 * 100)
        assert report.useful_share_percent_mean_of_ratios == pytest.approx(49.995, abs=1e-3)
        assert report.useful_share_percent != report.useful_share_percent_mean_of_ratios

    def test_overhead_lowers_the_useful_share(self):
        report = compute_shares([1.0], [9.0], [10.0])
        assert report.useful_share_percent == pytest.approx(90.0)
        assert report.useful_share_percent_with_overhead == pytest.approx(45.0)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            compute_shares([1.0, 2.0], [1.0])

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            compute_shares([], [])


class TestAsymmetry:
    def test_ratio_matches_hand_computation(self):
        report = compute_asymmetry(
            solve_seconds=[1.0],
            step_seconds={"a": [0.25], "b": [0.25]},
        )
        assert report.mean_verify_seconds == pytest.approx(0.5)
        assert report.asymmetry_ratio == pytest.approx(2.0)

    def test_reproduces_the_papers_asymmetry_arithmetic(self):
        # Paper: 785.08 ms solve / 0.005069 ms verify = 154,878x, reported as ~155,000.
        report = compute_asymmetry(
            solve_seconds=[0.78508],
            step_seconds={
                "pow_check": [0.000000458],
                "vrf_verify": [0.000001958],
                "merkle_verify": [0.000002435],
                "solution_check": [0.000000218],
            },
        )
        assert report.total_verify_milliseconds == pytest.approx(0.005069, abs=1e-9)
        assert report.asymmetry_ratio == pytest.approx(154878, rel=1e-3)


class TestConvergence:
    def test_agreement_on_one_tip_is_convergence(self):
        report = analyse_convergence({0: "abc", 1: "abc", 2: "abc"}, [True, True])
        assert report.converged
        assert report.distinct_tips == 1
        assert report.acceptance_rate == pytest.approx(1.0)

    def test_disagreement_is_detected(self):
        report = analyse_convergence({0: "abc", 1: "def"}, [True])
        assert not report.converged
        assert report.distinct_tips == 2
        assert report.tip_hash is None

    def test_rejected_blocks_lower_the_acceptance_rate(self):
        report = analyse_convergence({0: "a"}, [True, False, True, True])
        assert report.blocks_rejected == 1
        assert report.acceptance_rate == pytest.approx(0.75)


class TestCorrectness:
    def test_exact_match_is_fully_optimal(self):
        report = analyse_correctness([10.0, 20.0], [10.0, 20.0])
        assert report.optimality_rate == pytest.approx(1.0)
        assert report.max_relative_gap == pytest.approx(0.0)

    def test_gap_is_measured_relative_to_the_optimum(self):
        report = analyse_correctness([11.0], [10.0])
        assert report.optimality_rate == pytest.approx(0.0)
        assert report.max_relative_gap == pytest.approx(0.1)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            analyse_correctness([1.0], [1.0, 2.0])
