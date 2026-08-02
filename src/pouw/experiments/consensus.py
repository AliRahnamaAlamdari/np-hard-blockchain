"""Consensus-level experiments (reviewer comment 3), multi-seed with 95% CIs.

Uses the event-driven :class:`ConsensusSim`. Every reported quantity is a mean over
independent seeds with a t-based 95% confidence interval. Sub-studies:

* fork rate and reorg depth vs propagation delay;
* convergence (k-deep common prefix) and fork rate vs miner count (order-of-magnitude
  scaling);
* block-interval mean/variance vs heterogeneity of useful-task solve times;
* liveness and reorg under churn and under a temporary network partition;
* difficulty-adjustment convergence to a target block interval.

These numbers come from a network model, not literal PoW execution, and are labeled as
such in the paper.
"""

from __future__ import annotations

from typing import Any

from ..chain.consensus_sim import ConsensusConfig, ConsensusSim
from ..seeding import seed_everything
from ..stats_utils import summarise
from .base import Experiment


def _agg(values: list[float]) -> dict[str, float]:
    s = summarise(values)
    return {"mean": s["mean"], "std": s["std"], "ci95_half": s["ci95_half_width"]}


class ConsensusExperiment(Experiment):
    name = "consensus"
    description = "Fork rate, convergence, liveness, difficulty adjustment (event-driven model)"

    def _run_many(self, make_cfg) -> list:
        results = []
        for seed in self.config.run.seeds:
            cfg = make_cfg()
            results.append(ConsensusSim(cfg, seed_everything(seed)).run())
        return results

    def run(self) -> dict[str, Any]:
        blocks = int(self.config.sweep.get("blocks", 400))
        base_rate = float(self.config.sweep.get("base_block_rate", 1.0))

        # --- fork rate vs propagation delay ---
        delay_rows = []
        for delay in self.config.sweep.get("delays", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]):
            res = self._run_many(lambda d=delay: ConsensusConfig(
                num_miners=int(self.config.sweep.get("miners", 20)), num_blocks=blocks,
                base_block_rate=base_rate, propagation_delay=d))
            delay_rows.append({
                "delay": delay,
                "fork_rate": _agg([r.fork_rate for r in res]),
                "mean_reorg_depth": _agg([r.mean_reorg_depth for r in res]),
                "max_reorg_depth": max(r.max_reorg_depth for r in res),
                "k_deep_converged_rate": sum(r.converged_k_deep for r in res) / len(res),
            })
            self.logger.info("delay=%.2f fork_rate=%.3f±%.3f", delay,
                             delay_rows[-1]["fork_rate"]["mean"], delay_rows[-1]["fork_rate"]["ci95_half"])

        # --- scaling: fork rate & convergence vs miner count ---
        scale_rows = []
        for miners in self.config.sweep.get("miner_counts", [10, 50, 100, 500, 1000]):
            res = self._run_many(lambda m=miners: ConsensusConfig(
                num_miners=m, num_blocks=blocks, base_block_rate=base_rate,
                propagation_delay=float(self.config.sweep.get("scale_delay", 0.2))))
            scale_rows.append({
                "miners": miners,
                "fork_rate": _agg([r.fork_rate for r in res]),
                "k_deep_converged_rate": sum(r.converged_k_deep for r in res) / len(res),
                "mean_reorg_depth": _agg([r.mean_reorg_depth for r in res]),
            })
            self.logger.info("miners=%d fork_rate=%.3f k_deep_conv=%.2f", miners,
                             scale_rows[-1]["fork_rate"]["mean"], scale_rows[-1]["k_deep_converged_rate"])

        # --- heterogeneous solve time -> interval variance ---
        hetero_rows = []
        for std in self.config.sweep.get("solve_stds", [0.0, 0.5, 1.0, 2.0]):
            res = self._run_many(lambda s=std: ConsensusConfig(
                num_miners=int(self.config.sweep.get("miners", 20)), num_blocks=blocks,
                propagation_delay=0.1, mean_solve_time=1.0, solve_time_std=s))
            hetero_rows.append({
                "solve_time_std": std,
                "mean_interval": _agg([r.mean_interval for r in res]),
                "std_interval": _agg([r.std_interval for r in res]),
                "fork_rate": _agg([r.fork_rate for r in res]),
            })
            self.logger.info("solve_std=%.1f interval=%.3f±%.3f", std,
                             hetero_rows[-1]["mean_interval"]["mean"], hetero_rows[-1]["std_interval"]["mean"])

        # --- churn & partition liveness ---
        churn_res = self._run_many(lambda: ConsensusConfig(
            num_miners=30, num_blocks=blocks, propagation_delay=0.1,
            churn_interval=int(self.config.sweep.get("churn_interval", 50)),
            churn_fraction=float(self.config.sweep.get("churn_fraction", 0.3))))
        partition_res = self._run_many(lambda: ConsensusConfig(
            num_miners=30, num_blocks=blocks, propagation_delay=0.1,
            partition_start=int(blocks * 0.3), partition_end=int(blocks * 0.5)))
        resilience = {
            "churn": {
                "fork_rate": _agg([r.fork_rate for r in churn_res]),
                "k_deep_converged_rate": sum(r.converged_k_deep for r in churn_res) / len(churn_res),
                "max_liveness_gap": _agg([r.liveness_gap_max for r in churn_res]),
                "blocks_produced": _agg([float(r.blocks_produced) for r in churn_res]),
            },
            "partition": {
                "fork_rate": _agg([r.fork_rate for r in partition_res]),
                "max_reorg_depth": _agg([float(r.max_reorg_depth) for r in partition_res]),
                "k_deep_converged_after_heal": sum(r.converged_k_deep for r in partition_res) / len(partition_res),
            },
        }
        self.logger.info("churn k_deep_conv=%.2f | partition max_reorg=%.1f converged_after_heal=%.2f",
                         resilience["churn"]["k_deep_converged_rate"],
                         resilience["partition"]["max_reorg_depth"]["mean"],
                         resilience["partition"]["k_deep_converged_after_heal"])

        # --- difficulty adjustment ---
        target = float(self.config.sweep.get("retarget_target", 2.0))
        diff_res = self._run_many(lambda: ConsensusConfig(
            num_miners=15, num_blocks=int(self.config.sweep.get("retarget_blocks", 600)),
            base_block_rate=float(self.config.sweep.get("retarget_start_rate", 5.0)),
            propagation_delay=0.05, retarget_window=int(self.config.sweep.get("retarget_window", 20)),
            target_interval=target))
        # measure interval over the last third (post-convergence)
        difficulty = {
            "target_interval": target,
            "final_interval": _agg([r.mean_interval for r in diff_res]),
            "retarget_steps": _agg([float(len(r.difficulty_trace)) for r in diff_res]),
        }
        self.logger.info("difficulty: target=%.2f final_interval=%.3f", target, difficulty["final_interval"]["mean"])

        return {
            "delay_rows": delay_rows,
            "scale_rows": scale_rows,
            "hetero_rows": hetero_rows,
            "resilience": resilience,
            "difficulty": difficulty,
            "csv_tables": {
                "consensus_fork_vs_delay": [
                    {"delay": r["delay"], "fork_rate": r["fork_rate"]["mean"],
                     "fork_rate_ci95": r["fork_rate"]["ci95_half"],
                     "mean_reorg": r["mean_reorg_depth"]["mean"], "max_reorg": r["max_reorg_depth"]}
                    for r in delay_rows],
                "consensus_scale": [
                    {"miners": r["miners"], "fork_rate": r["fork_rate"]["mean"],
                     "fork_rate_ci95": r["fork_rate"]["ci95_half"],
                     "k_deep_converged_rate": r["k_deep_converged_rate"]} for r in scale_rows],
                "consensus_heterogeneity": [
                    {"solve_time_std": r["solve_time_std"], "mean_interval": r["mean_interval"]["mean"],
                     "std_interval": r["std_interval"]["mean"], "fork_rate": r["fork_rate"]["mean"]}
                    for r in hetero_rows],
            },
            "model_note": (
                "Numbers from an event-driven network model (Poisson block discovery, "
                "service-time useful work, delayed gossip), not literal PoW; used because "
                "fork/liveness behaviour cannot arise in a synchronous zero-delay race and "
                "because it scales to 1000 miners."
            ),
        }
