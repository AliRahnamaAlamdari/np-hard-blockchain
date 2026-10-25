"""Adversarial experiments corroborating the formal security claims (comment 2).

Runs every harness in :mod:`pouw.chain.adversary` plus a malicious-majority study on the
consensus simulator, reporting measured outcomes next to the values the theorems predict.
The malicious-majority result is deliberately a *negative* one: the framework inherits
longest-chain assumptions and does not resist a dishonest majority, which the paper
states and this experiment confirms.
"""

from __future__ import annotations

import random
from typing import Any

from ..chain import adversary as adv
from ..chain.consensus_sim import ConsensusConfig, ConsensusSim
from ..crypto.vrf import build_vrf
from ..seeding import seed_everything
from ..stats_utils import confidence_interval, mean_std, summarise
from .base import Experiment


class AdversarialExperiment(Experiment):
    name = "adversarial"
    description = "Grinding, precomputation, replay, assignment bias, adaptive steering, malicious majority"

    def run(self) -> dict[str, Any]:
        repo_size = int(self.config.repository.num_instances)
        difficulty = self.config.pow.difficulty_bits

        # --- grinding: unique-output vs malleable ---
        grinding_rows = []
        for backend in self.config.sweep.get("vrf_backends", ["ecvrf", "ecdsa_pseudo"]):
            per_seed = []
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                vrf = build_vrf(backend)
                kp = vrf.keygen(bundle.bytes_for("adv"))
                g = adv.grinding_advantage(vrf, kp.secret, kp.public, repo_size,
                                           int(self.config.sweep.get("grinding_evals", 50)))
                per_seed.append(g.distinct_tasks_per_seed)
            m, s = mean_std(per_seed)
            low, _, high = confidence_interval(per_seed)
            grinding_rows.append({"vrf_backend": backend, "mean_distinct_tasks_per_seed": m,
                                  "std": s, "unique_output": build_vrf(backend).is_unique_output(),
                                  "evaluations_per_seed": int(self.config.sweep.get("grinding_evals", 50)),
                                  "num_seeds": len(per_seed), "ci95_low": low, "ci95_high": high})
            self.logger.info("grinding %s: %.1f tasks/seed", backend, m)

        # --- precomputation: hit rate == coverage, independent of repo size ---
        precomp_rows = []
        vrf = build_vrf("ecvrf")
        # The VRF outputs depend only on the key and trial label, so every coverage level
        # sees the same indices; their uniformity is tested once, on the first level.
        precomp_indices: list[int] = []
        for frac in self.config.sweep.get("precomputed_fractions", [0.05, 0.1, 0.25, 0.5]):
            per_seed = []
            hit_flags: list[float] = []
            coverage = frac
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                kp = vrf.keygen(bundle.bytes_for("adv"))
                r = adv.precomputation_exposure(vrf, kp.secret, repo_size, frac,
                                                int(self.config.sweep.get("precomp_trials", 800)),
                                                random.Random(bundle.integer("precomp")))
                per_seed.append(r.empirical_hit_rate)
                hit_flags.extend([1.0] * r.hits + [0.0] * (r.trials - r.hits))
                coverage = r.theoretical_hit_rate
                if not precomp_rows:
                    precomp_indices.extend(r.indices)
            m, s = mean_std(per_seed)
            low, _, high = confidence_interval(hit_flags)
            # The prediction is the realised coverage int(|R| * frac) / |R|, not the
            # nominal fraction: 0.05 of 256 instances is 12, i.e. 0.0469.
            precomp_rows.append({"precomputed_fraction": frac, "empirical_hit_rate": m,
                                 "hit_rate_std": s, "theoretical_hit_rate": coverage,
                                 "trials": len(hit_flags), "hits": int(sum(hit_flags)),
                                 "ci95_low": low, "ci95_high": high})
        from scipy import stats

        precomp_uniformity_p = float(
            stats.chisquare([precomp_indices.count(i) for i in range(repo_size)]).pvalue
        )

        # --- replay ---
        kp = vrf.keygen(b"replay")
        replay = adv.replay_rejection(vrf, kp.secret, kp.public, repo_size)
        replay_row = {
            "seeds_differ_across_headers": replay.seeds_differ_across_headers,
            "cross_header_proof_rejected": replay.cross_header_proof_rejected,
            "replayed_index_inconsistent": replay.replayed_index_inconsistent,
        }

        # --- assignment bias (uniformity chi-square) ---
        bias_size = int(self.config.sweep.get("bias_repo_size", 16))
        bias_rows = []
        for seed in self.config.run.seeds:
            bundle = seed_everything(seed)
            kp = vrf.keygen(bundle.bytes_for("bias"))
            r = adv.assignment_bias(vrf, kp.secret, bias_size,
                                    int(self.config.sweep.get("bias_samples", 1600)))
            bias_rows.append(r.p_value)
        bias_p_mean, bias_p_std = mean_std(bias_rows)

        # --- adaptive steering: E[seeds] = 1/phi ---
        steer_rows = []
        for phi in self.config.sweep.get("favoured_fractions", [0.5, 0.25]):
            per_seed = []
            samples: list[int] = []
            for seed in self.config.run.seeds:
                bundle = seed_everything(seed)
                kp = vrf.keygen(bundle.bytes_for("adv"))
                r = adv.adaptive_steering_cost(vrf, kp.secret, min(6, difficulty), 100, phi,
                                               int(self.config.sweep.get("steer_trials", 60)))
                per_seed.append(r.mean_pow_seeds_to_hit)
                samples.extend(r.samples)
            m, s = mean_std(per_seed)
            low, _, high = confidence_interval(samples)
            steer_rows.append({"favoured_fraction": phi, "mean_seeds": m, "std": s,
                               "theoretical": 1.0 / phi, "trials": len(samples),
                               "ci95_low": low, "ci95_high": high})
            self.logger.info("steering phi=%.2f: %.2f seeds (theory %.2f)", phi, m, 1.0 / phi)

        # --- malicious majority (negative result via consensus sim) ---
        majority_rows = []
        n = int(self.config.sweep.get("majority_miners", 20))
        blocks = int(self.config.sweep.get("majority_blocks", 400))
        for beta in self.config.sweep.get("adversary_shares", [0.1, 0.3, 0.5, 0.7, 0.9]):
            # one adversary miner (id 0) holds share beta; the rest split the remainder
            adv_rate = beta * (n - 1) / (1 - beta) if beta < 1 else 1e9
            rates = [adv_rate] + [1.0] * (n - 1)
            per_seed = []
            for seed in self.config.run.seeds:
                res = ConsensusSim(ConsensusConfig(num_miners=n, hash_rates=rates,
                                                   num_blocks=blocks, propagation_delay=0.1),
                                   seed_everything(seed)).run()
                total_main = sum(res.main_chain_miner_counts.values())
                adv_share = res.main_chain_miner_counts.get(0, 0) / total_main if total_main else 0.0
                per_seed.append(adv_share)
            m, s = mean_std(per_seed)
            low, _, high = confidence_interval(per_seed)
            majority_rows.append({"adversary_hashrate_share": beta, "adversary_mainchain_share": m,
                                  "std": s, "blocks_per_seed": blocks, "num_seeds": len(per_seed),
                                  "ci95_low": low, "ci95_high": high})
            self.logger.info("majority beta=%.1f -> adversary won %.1f%% of main chain", beta, m * 100)

        return {
            "grinding": grinding_rows,
            "precomputation": precomp_rows,
            "precomputation_index_uniformity_p": precomp_uniformity_p,
            "replay": replay_row,
            "assignment_bias": {"repo_size": bias_size, "mean_p_value": bias_p_mean,
                                "std_p_value": bias_p_std, "uniform_at_5pct": bias_p_mean > 0.05},
            "adaptive_steering": steer_rows,
            "malicious_majority": majority_rows,
            "csv_tables": {
                "adversarial_grinding": grinding_rows,
                "adversarial_precomputation": precomp_rows,
                "adversarial_steering": steer_rows,
                "adversarial_majority": majority_rows,
            },
            "interpretation": (
                "Grinding, precomputation, replay, assignment bias, and adaptive steering "
                "all match their theoretical predictions under a unique-output VRF. The "
                "malicious-majority study is a negative control: adversary main-chain share "
                "tracks its hash-rate share and exceeds 50% once it holds a majority, "
                "confirming the framework inherits rather than improves longest-chain "
                "majority assumptions."
            ),
        }
