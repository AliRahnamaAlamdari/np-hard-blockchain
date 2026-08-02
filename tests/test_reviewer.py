"""Tests for the consensus simulator, adversary harnesses, and CI statistics."""

from __future__ import annotations

import math

import pytest

from pouw.chain import adversary as adv
from pouw.chain.consensus_sim import ConsensusConfig, ConsensusSim
from pouw.crypto.vrf import build_vrf
from pouw.seeding import seed_everything
from pouw.stats_utils import ci_half_width, confidence_interval


class TestConfidenceInterval:
    def test_zero_variance_gives_zero_width(self):
        low, mean, high = confidence_interval([5.0, 5.0, 5.0])
        assert mean == pytest.approx(5.0)
        assert low == pytest.approx(5.0) and high == pytest.approx(5.0)

    def test_single_value_has_no_interval(self):
        low, mean, high = confidence_interval([7.0])
        assert low == mean == high == 7.0

    def test_known_interval(self):
        # For [1,2,3,4,5]: mean 3, sample sd ~1.5811, se ~0.7071, t(0.975,4)=2.776
        # half-width ~ 1.963
        assert ci_half_width([1, 2, 3, 4, 5]) == pytest.approx(1.9626, abs=1e-3)

    def test_interval_brackets_mean(self):
        low, mean, high = confidence_interval([10, 12, 11, 13, 9, 14])
        assert low < mean < high

    def test_empty_is_nan(self):
        low, mean, high = confidence_interval([])
        assert math.isnan(mean)


class TestConsensusSim:
    def _fork_rate(self, delay, seed=0, miners=10, blocks=200):
        cfg = ConsensusConfig(num_miners=miners, num_blocks=blocks, base_block_rate=1.0,
                              propagation_delay=delay)
        return ConsensusSim(cfg, seed_everything(seed)).run().fork_rate

    def test_zero_delay_has_no_forks(self):
        assert self._fork_rate(0.0) == 0.0

    def test_fork_rate_increases_with_delay(self):
        low = self._fork_rate(0.05)
        high = self._fork_rate(1.0)
        assert high > low > 0.0

    def test_determinism(self):
        cfg = ConsensusConfig(num_miners=10, num_blocks=200, propagation_delay=0.2)
        a = ConsensusSim(cfg, seed_everything(3)).run()
        b = ConsensusSim(cfg, seed_everything(3)).run()
        assert a.fork_rate == b.fork_rate
        assert a.blocks_produced == b.blocks_produced

    def test_k_deep_convergence_holds(self):
        cfg = ConsensusConfig(num_miners=50, num_blocks=200, propagation_delay=0.2)
        res = ConsensusSim(cfg, seed_everything(0)).run()
        assert res.converged_k_deep

    def test_scales_to_many_miners(self):
        cfg = ConsensusConfig(num_miners=1000, num_blocks=100, propagation_delay=0.2)
        res = ConsensusSim(cfg, seed_everything(0)).run()
        assert res.blocks_produced >= 100
        assert res.converged_k_deep

    def test_partition_causes_reorg_then_heals(self):
        cfg = ConsensusConfig(num_miners=20, num_blocks=300, propagation_delay=0.1,
                              partition_start=100, partition_end=150)
        res = ConsensusSim(cfg, seed_everything(0)).run()
        assert res.max_reorg_depth > 1  # the minority branch is reorged out
        assert res.converged_k_deep  # heals afterward

    def test_heterogeneous_solve_inflates_interval(self):
        base = ConsensusSim(ConsensusConfig(num_miners=10, num_blocks=300, propagation_delay=0.1,
                            mean_solve_time=1.0, solve_time_std=0.0), seed_everything(0)).run()
        assert base.mean_interval > 1.0  # solve time adds to the interval

    def test_difficulty_retarget_moves_interval_toward_target(self):
        cfg = ConsensusConfig(num_miners=10, num_blocks=400, base_block_rate=5.0,
                              propagation_delay=0.05, retarget_window=20, target_interval=2.0)
        res = ConsensusSim(cfg, seed_everything(0)).run()
        # without retargeting the interval would be ~1/5; retargeting pushes it up toward 2
        assert res.mean_interval > 0.5
        assert len(res.difficulty_trace) > 0

    def test_majority_miner_wins_majority_of_main_chain(self):
        # miner 0 holds ~80% of hash rate
        rates = [0.8 * 9 / 0.2] + [1.0] * 9
        cfg = ConsensusConfig(num_miners=10, hash_rates=rates, num_blocks=300, propagation_delay=0.1)
        res = ConsensusSim(cfg, seed_everything(0)).run()
        total = sum(res.main_chain_miner_counts.values())
        assert res.main_chain_miner_counts.get(0, 0) / total > 0.6


class TestAdversary:
    def test_grinding_unique_vs_malleable(self):
        ec = build_vrf("ecvrf")
        kp = ec.keygen(b"k")
        g = adv.grinding_advantage(ec, kp.secret, kp.public, 256, 40)
        assert g.distinct_tasks_per_seed == 1.0 and g.unique_output

        mal = build_vrf("ecdsa_pseudo")
        kp2 = mal.keygen(b"k")
        g2 = adv.grinding_advantage(mal, kp2.secret, kp2.public, 256, 40)
        assert g2.distinct_tasks_per_seed > 1.0 and not g2.unique_output
        assert g2.all_proofs_valid  # the ground proofs actually verify

    def test_precomputation_hit_matches_coverage(self):
        import random
        vrf = build_vrf("ecvrf")
        kp = vrf.keygen(b"k")
        r = adv.precomputation_exposure(vrf, kp.secret, 200, 0.3, 1500, random.Random(0))
        assert abs(r.empirical_hit_rate - 0.3) < 0.05

    def test_replay_is_rejected(self):
        vrf = build_vrf("ecvrf")
        kp = vrf.keygen(b"k")
        r = adv.replay_rejection(vrf, kp.secret, kp.public, 256)
        assert r.seeds_differ_across_headers
        assert r.cross_header_proof_rejected
        assert r.replayed_index_inconsistent

    def test_assignment_is_uniform(self):
        vrf = build_vrf("ecvrf")
        kp = vrf.keygen(b"k")
        r = adv.assignment_bias(vrf, kp.secret, 16, 1600)
        assert r.uniform_at_5pct

    def test_adaptive_steering_matches_inverse_phi(self):
        vrf = build_vrf("ecvrf")
        kp = vrf.keygen(b"k")
        r = adv.adaptive_steering_cost(vrf, kp.secret, 4, 100, 0.5, 80)
        # E[seeds] = 1/phi = 2; allow sampling slack
        assert 1.5 < r.mean_pow_seeds_to_hit < 2.8
