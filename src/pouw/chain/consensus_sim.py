"""Event-driven consensus simulator for fork / liveness / convergence analysis.

Reviewer comment 3 asks for fork probability, chain convergence, liveness under adverse
conditions, difficulty adjustment, and the effect of heterogeneous useful-task execution
times on the block interval. The literal-hash simulator in :mod:`pouw.chain.network`
cannot produce forks --- it is a synchronous, zero-delay race --- so this module models
the network as a continuous-time process instead.

Model (standard for blockchain fork analysis, e.g. Bitcoin propagation studies):

* Block production is a race. Miner ``i`` completes a candidate block at rate
  ``lambda_i`` proportional to its hash rate; the time to the next completion is
  exponential. This abstracts the Poisson nature of PoW search.
* After finding the PoW, the miner spends a *useful-work solve time* before the block is
  ready to broadcast. Solve times may be heterogeneous across miners (comment 3's
  "heterogeneous execution times"), which inflates the effective block interval and its
  variance.
* A finished block propagates to every other node after a per-link delay. A fork occurs
  when a miner builds on a tip that is already stale because a competing block has been
  produced but not yet delivered.
* Nodes follow the longest-chain rule, ties broken by first-seen. Switching to a longer
  competing branch is a reorg; its depth is recorded.
* Difficulty is retargeted every window to steer the mean block interval toward a goal.

This is a model, not literal execution, and is labeled as such wherever its numbers are
reported. It scales to thousands of miners, which literal PoW cannot.
"""

from __future__ import annotations

import heapq
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from ..seeding import SeedBundle


@dataclass
class SimBlock:
    block_id: int
    height: int
    parent_id: int
    miner_id: int
    produced_at: float


@dataclass
class ConsensusConfig:
    num_miners: int = 10
    hash_rates: list[float] | None = None  # None => equal
    base_block_rate: float = 1.0  # expected PoW completions per unit time, whole network
    mean_solve_time: float = 0.0  # added service time per block (useful work)
    solve_time_std: float = 0.0  # heterogeneity of solve time across blocks
    propagation_delay: float = 0.1  # mean link delay
    delay_model: str = "exponential"  # "constant" | "exponential"
    num_blocks: int = 500
    # difficulty adjustment
    retarget_window: int = 0  # 0 disables
    target_interval: float = 1.0
    # churn: fraction of miners that leave/rejoin, applied every churn_interval blocks
    churn_interval: int = 0
    churn_fraction: float = 0.0
    # partition: split network for [partition_start, partition_end] block-heights
    partition_start: int = 0
    partition_end: int = 0
    partition_fraction: float = 0.5


@dataclass
class ConsensusResult:
    blocks_produced: int = 0
    main_chain_length: int = 0
    orphaned_blocks: int = 0
    fork_rate: float = 0.0
    reorg_depths: list[int] = field(default_factory=list)
    max_reorg_depth: int = 0
    mean_reorg_depth: float = 0.0
    block_intervals: list[float] = field(default_factory=list)
    mean_interval: float = 0.0
    std_interval: float = 0.0
    converged: bool = False
    converged_k_deep: bool = False
    common_prefix_depth: int = 0
    distinct_final_tips: int = 0
    blocks_won: dict[int, int] = field(default_factory=dict)
    main_chain_miner_counts: dict[int, int] = field(default_factory=dict)
    difficulty_trace: list[float] = field(default_factory=list)
    liveness_gap_max: float = 0.0
    partition_reorg_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["reorg_depths"] = list(self.reorg_depths)
        d["block_intervals"] = None  # too large to serialize; summary stats kept
        d["blocks_won"] = {str(k): v for k, v in self.blocks_won.items()}
        return d


class ConsensusSim:
    """Continuous-time discrete-event simulation of the mining network."""

    def __init__(self, config: ConsensusConfig, seeds: SeedBundle) -> None:
        self.config = config
        self.rng = seeds.python_random("consensus")
        rates = config.hash_rates or [1.0] * config.num_miners
        if len(rates) != config.num_miners:
            raise ValueError("hash_rates length must equal num_miners")
        self.base_rates = list(rates)
        self.total_rate = sum(rates)
        self.blocks: dict[int, SimBlock] = {}
        self._next_id = 1
        self.difficulty_scale = 1.0  # multiplies base_block_rate (lower => slower)

    def _delay(self) -> float:
        d = self.config.propagation_delay
        if d <= 0:
            return 0.0
        if self.config.delay_model == "constant":
            return d
        return self.rng.expovariate(1.0 / d)

    def _solve_time(self) -> float:
        m = self.config.mean_solve_time
        if m <= 0:
            return 0.0
        if self.config.solve_time_std <= 0:
            return m
        # lognormal keeps solve time positive with the requested mean/spread
        sigma = math.sqrt(math.log(1 + (self.config.solve_time_std / m) ** 2))
        mu = math.log(m) - sigma * sigma / 2
        return self.rng.lognormvariate(mu, sigma)

    def run(self) -> ConsensusResult:
        cfg = self.config
        genesis = SimBlock(0, 0, -1, -1, 0.0)
        self.blocks[0] = genesis

        # Each node's local canonical tip (block_id) and its height.
        node_tip = {i: 0 for i in range(cfg.num_miners)}
        node_height = {i: 0 for i in range(cfg.num_miners)}
        active = set(range(cfg.num_miners))
        blocks_won = {i: 0 for i in range(cfg.num_miners)}
        reorg_depths: list[int] = []
        produced_times: list[float] = []
        difficulty_trace: list[float] = []

        # Event queue: (time, seq, kind, payload). seq breaks time ties deterministically.
        queue: list[tuple[float, int, str, Any]] = []
        seq = 0

        def push(t: float, kind: str, payload: Any) -> None:
            nonlocal seq
            heapq.heappush(queue, (t, seq, kind, payload))
            seq += 1

        def schedule_completion(miner: int, now: float) -> None:
            rate = self.base_rates[miner] / self.total_rate * cfg.base_block_rate * self.difficulty_scale
            if rate <= 0:
                return
            pow_time = self.rng.expovariate(rate)
            ready = now + pow_time + self._solve_time()
            push(ready, "complete", miner)

        for miner in active:
            schedule_completion(miner, 0.0)

        def partition_group(miner: int) -> int:
            # deterministic split by miner id
            return 0 if miner < int(cfg.num_miners * cfg.partition_fraction) else 1

        produced = 0
        last_progress_time = 0.0
        liveness_gap_max = 0.0
        height_reached = 0

        while queue and produced < cfg.num_blocks:
            t, _, kind, payload = heapq.heappop(queue)

            if kind == "complete":
                miner = payload
                if miner not in active:
                    continue
                parent = node_tip[miner]
                parent_block = self.blocks[parent]
                block = SimBlock(self._next_id, parent_block.height + 1, parent, miner, t)
                self.blocks[self._next_id] = block
                self._next_id += 1
                produced += 1
                blocks_won[miner] += 1
                produced_times.append(t)

                # liveness: time since the main chain last advanced in height
                if block.height > height_reached:
                    gap = t - last_progress_time
                    liveness_gap_max = max(liveness_gap_max, gap)
                    last_progress_time = t
                    height_reached = block.height

                # local adoption for the producer
                node_tip[miner] = block.block_id
                node_height[miner] = block.height

                # propagate to peers (respecting partition window)
                in_partition = (
                    cfg.partition_end > cfg.partition_start
                    and cfg.partition_start <= height_reached <= cfg.partition_end
                )
                for peer in range(cfg.num_miners):
                    if peer == miner or peer not in active:
                        continue
                    if in_partition and partition_group(peer) != partition_group(miner):
                        continue  # link severed during partition
                    push(t + self._delay(), "deliver", (peer, block.block_id))

                # difficulty retarget
                if cfg.retarget_window and produced % cfg.retarget_window == 0 and len(produced_times) > cfg.retarget_window:
                    window = produced_times[-cfg.retarget_window:]
                    observed = (window[-1] - window[0]) / (cfg.retarget_window - 1)
                    if observed > 0:
                        # scale rate to push observed interval toward target
                        adjust = observed / cfg.target_interval
                        self.difficulty_scale *= max(0.25, min(4.0, adjust))
                        difficulty_trace.append(self.difficulty_scale)

                # churn
                if cfg.churn_interval and produced % cfg.churn_interval == 0 and cfg.churn_fraction > 0:
                    k = max(1, int(cfg.num_miners * cfg.churn_fraction))
                    leaving = self.rng.sample(sorted(active), min(k, len(active) - 1)) if len(active) > 1 else []
                    for m in leaving:
                        active.discard(m)
                    # rejoin previously-left miners
                    absent = set(range(cfg.num_miners)) - active
                    for m in list(absent)[:k]:
                        active.add(m)
                        node_tip[m] = node_tip[miner]  # sync to a current peer on rejoin
                        node_height[m] = node_height[miner]
                        schedule_completion(m, t)

                schedule_completion(miner, t)

            elif kind == "deliver":
                peer, block_id = payload
                if peer not in active:
                    continue
                block = self.blocks[block_id]
                if block.height > node_height[peer]:
                    # adopt longer chain; measure reorg depth = how far back the fork is
                    depth = self._reorg_depth(node_tip[peer], block, node_height[peer])
                    if depth > 0:
                        reorg_depths.append(depth)
                    node_tip[peer] = block_id
                    node_height[peer] = block.height

        # Final convergence: drain remaining deliveries so honest nodes settle.
        while queue:
            t, _, kind, payload = heapq.heappop(queue)
            if kind == "deliver":
                peer, block_id = payload
                if peer not in active:
                    continue
                block = self.blocks[block_id]
                if block.height > node_height[peer]:
                    node_tip[peer] = block_id
                    node_height[peer] = block.height

        return self._finalize(node_tip, blocks_won, reorg_depths, produced_times, difficulty_trace, liveness_gap_max, active)

    def _reorg_depth(self, old_tip_id: int, new_block: SimBlock, old_height: int) -> int:
        """Depth of the reorg when switching from ``old_tip`` to ``new_block``'s chain."""
        # Walk the new chain back to old_height, then find common ancestor.
        new_ancestors = set()
        cur = new_block
        while cur.height >= 0:
            new_ancestors.add(cur.block_id)
            if cur.parent_id < 0:
                break
            cur = self.blocks[cur.parent_id]
        depth = 0
        cur_id = old_tip_id
        while cur_id not in new_ancestors and cur_id >= 0:
            depth += 1
            parent = self.blocks[cur_id].parent_id
            if parent < 0:
                break
            cur_id = parent
        return depth

    def _finalize(self, node_tip, blocks_won, reorg_depths, produced_times, difficulty_trace, liveness_gap_max, active) -> ConsensusResult:
        # Determine the canonical chain as the longest among active node tips.
        active_tips = [node_tip[i] for i in active] or list(node_tip.values())
        best_tip = max(active_tips, key=lambda bid: self.blocks[bid].height)
        main_chain = set()
        main_chain_miner_counts: dict[int, int] = {}
        cur = self.blocks[best_tip]
        while cur.height >= 0:
            main_chain.add(cur.block_id)
            if cur.miner_id >= 0:
                main_chain_miner_counts[cur.miner_id] = main_chain_miner_counts.get(cur.miner_id, 0) + 1
            if cur.parent_id < 0:
                break
            cur = self.blocks[cur.parent_id]

        total_blocks = len(self.blocks) - 1  # exclude genesis
        main_len = self.blocks[best_tip].height
        orphans = total_blocks - main_len

        intervals = [b - a for a, b in zip(produced_times, produced_times[1:])]
        final_tips = {node_tip[i] for i in active} if active else set(node_tip.values())

        # k-deep common prefix: the standard convergence guarantee. Nakamoto consensus
        # agrees on all but the most recent few blocks, not necessarily the exact tip.
        confirm_depth = max(1, int(0.02 * main_len) + 1)  # ~2% of chain, min 1
        tip_nodes = list(active) if active else list(node_tip.keys())
        ancestors_at_depth = []
        for node in tip_nodes:
            cur = self.blocks[node_tip[node]]
            steps = confirm_depth
            while steps > 0 and cur.parent_id >= 0:
                cur = self.blocks[cur.parent_id]
                steps -= 1
            ancestors_at_depth.append(cur.block_id)
        converged_k_deep = len(set(ancestors_at_depth)) == 1

        res = ConsensusResult()
        res.blocks_produced = total_blocks
        res.main_chain_length = main_len
        res.orphaned_blocks = orphans
        res.fork_rate = orphans / total_blocks if total_blocks else 0.0
        res.reorg_depths = reorg_depths
        res.max_reorg_depth = max(reorg_depths) if reorg_depths else 0
        res.mean_reorg_depth = statistics.fmean(reorg_depths) if reorg_depths else 0.0
        res.block_intervals = intervals
        res.mean_interval = statistics.fmean(intervals) if intervals else 0.0
        res.std_interval = statistics.pstdev(intervals) if len(intervals) > 1 else 0.0
        res.converged = len(final_tips) == 1
        res.converged_k_deep = converged_k_deep
        res.common_prefix_depth = confirm_depth
        res.distinct_final_tips = len(final_tips)
        res.blocks_won = blocks_won
        res.main_chain_miner_counts = main_chain_miner_counts
        res.difficulty_trace = difficulty_trace
        res.liveness_gap_max = liveness_gap_max
        return res
