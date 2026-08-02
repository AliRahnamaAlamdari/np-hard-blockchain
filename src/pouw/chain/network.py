"""Discrete-event simulation of the mining network (paper §Prototype Implementation).

At each height every miner races for the Stage-1 gate.  Because PoW attempts are
independent Bernoulli trials, the miner that needs the fewest *simulated seconds*
(attempts divided by its hash rate) is the one that would have won a real parallel
race, so the winner distribution is exact.  All miners' searches are actually executed
rather than sampled analytically, so the reported PoW wall-clock times are measured
work, not a model.

The winner then runs Stage 2 and every node independently validates the block, which
is what the chain-convergence metric observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..crypto.hashing import difficulty_to_target, sha256_hex
from ..crypto.vrf import build_vrf
from ..logging_utils import Timer, get_logger
from ..seeding import SeedBundle
from ..tasks.repository import ProblemRepository
from .block import Block
from .miner import Miner, MiningError, PoWResult
from .validator import Validator

LOGGER = get_logger("pouw.network")

GENESIS_HASH = sha256_hex(b"pouw-genesis")


@dataclass
class BlockRecord:
    """Everything measured about one produced block."""

    height: int
    winner_id: int
    pow_seconds: float
    pow_attempts: int
    network_attempts: int
    simulated_seconds: float
    vrf_prove_seconds: float
    merkle_proof_seconds: float
    solve_seconds: float
    package_seconds: float
    solver_operations: int
    instance_id: int
    instance_size: int
    solution_cost: float
    block_hash: str
    validation_seconds: dict[int, float]
    validation_steps: dict[str, float]
    all_accepted: bool

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "height": self.height,
            "winner_id": self.winner_id,
            "pow_seconds": self.pow_seconds,
            "pow_attempts": self.pow_attempts,
            "network_attempts": self.network_attempts,
            "simulated_seconds": self.simulated_seconds,
            "vrf_prove_seconds": self.vrf_prove_seconds,
            "merkle_proof_seconds": self.merkle_proof_seconds,
            "solve_seconds": self.solve_seconds,
            "package_seconds": self.package_seconds,
            "solver_operations": self.solver_operations,
            "instance_id": self.instance_id,
            "instance_size": self.instance_size,
            "solution_cost": self.solution_cost,
            "block_hash": self.block_hash,
            "all_accepted": self.all_accepted,
        }
        payload.update(
            {f"validation_step_{k}": v for k, v in self.validation_steps.items()}
        )
        return payload


@dataclass
class SimulationResult:
    """Aggregate outcome of a full simulation run."""

    blocks: list[BlockRecord] = field(default_factory=list)
    blocks_won: dict[int, int] = field(default_factory=dict)
    node_tips: dict[int, str] = field(default_factory=dict)
    converged: bool = True
    rejected: int = 0
    timer: Timer = field(default_factory=Timer)
    repository_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def total_pow_seconds(self) -> float:
        return sum(record.pow_seconds for record in self.blocks)

    def total_solve_seconds(self) -> float:
        return sum(record.solve_seconds for record in self.blocks)


class Network:
    """A set of miners and validators sharing one repository and one chain."""

    def __init__(
        self,
        num_miners: int,
        repository: ProblemRepository,
        difficulty_bits: int,
        vrf_backend: str,
        seeds: SeedBundle,
        solver: str = "brute_force",
        hash_rates: list[float] | None = None,
        max_attempts: int = 100_000_000,
        quality_threshold: float | None = None,
        cost_tolerance: float = 1e-9,
        tie_break: str = "lowest_index",
        vrf_kwargs: dict[str, Any] | None = None,
        batch_size: int = 64,
    ) -> None:
        self.repository = repository
        self.target = difficulty_to_target(difficulty_bits)
        self.difficulty_bits = difficulty_bits
        self.solver = solver
        self.max_attempts = max_attempts
        self.tie_break = tie_break
        self.seeds = seeds
        self.batch_size = batch_size
        vrf_kwargs = vrf_kwargs or {}

        self.vrf = build_vrf(vrf_backend, **vrf_kwargs)
        rates = hash_rates or [1.0] * num_miners
        self.miners = [
            Miner(
                miner_id=index,
                keys=self.vrf.keygen(seeds.bytes_for(f"miner:{index}")),
                vrf=self.vrf,
                hash_rate=rates[index],
            )
            for index in range(num_miners)
        ]
        self.validators = [
            Validator(
                validator_id=index,
                vrf=self.vrf,
                repository=repository,
                target=self.target,
                quality_threshold=quality_threshold,
                cost_tolerance=cost_tolerance,
            )
            for index in range(num_miners)
        ]

    # -- one height ----------------------------------------------------------

    def _race(
        self, height: int, previous_hash: str, tx_root: str, timer: Timer
    ) -> tuple[Miner, PoWResult, float, float, int]:
        """Race every miner in lockstep until one satisfies the target.

        Miners advance in rounds, each doing a number of attempts proportional to its
        hash rate, and all of them stop as soon as any one succeeds.  This is the
        behaviour of a real parallel race: total network work per block is about
        ``2**difficulty_bits`` attempts regardless of how many miners participate.

        Returned wall time is the elapsed race, which is the network's Stage-1 cost
        for this block.
        """
        import time

        templates = {
            miner.miner_id: miner.header_template(height, previous_hash, tx_root)
            for miner in self.miners
        }
        cursors = {miner.miner_id: 0 for miner in self.miners}
        batches = {
            miner.miner_id: max(1, int(round(self.batch_size * miner.hash_rate)))
            for miner in self.miners
        }
        total_attempts = 0
        start = time.perf_counter()

        while total_attempts < self.max_attempts:
            # (miner, result, absolute_attempts, simulated_finish_time)
            round_winners: list[tuple[Miner, PoWResult, int, float]] = []
            for miner in self.miners:
                start_cursor = cursors[miner.miner_id]
                result, next_cursor = miner.search_pow_batch(
                    template=templates[miner.miner_id],
                    target=self.target,
                    start_nonce=start_cursor,
                    batch_size=batches[miner.miner_id],
                )
                total_attempts += next_cursor - start_cursor
                cursors[miner.miner_id] = next_cursor
                if result is not None:
                    absolute = start_cursor + result.attempts
                    # A miner with k times the hash rate performs k times as many
                    # attempts in the same wall-clock time, so simulated finish time
                    # is attempts divided by rate. Comparing raw attempt counts here
                    # would systematically penalise fast miners.
                    round_winners.append(
                        (miner, result, absolute, absolute / miner.hash_rate)
                    )
            if round_winners:
                if self.tie_break == "lowest_index":
                    winner, result, attempts, simulated = min(
                        round_winners, key=lambda entry: (entry[3], entry[0].miner_id)
                    )
                else:
                    winner, result, attempts, simulated = round_winners[0]
                elapsed = time.perf_counter() - start
                timer.record("pow_race_wall", elapsed)
                return (
                    winner,
                    PoWResult(
                        header=result.header,
                        digest=result.digest,
                        attempts=attempts,
                        seconds=elapsed,
                    ),
                    simulated,
                    elapsed,
                    total_attempts,
                )

        raise MiningError(
            f"no miner satisfied the target within {self.max_attempts} attempts at "
            f"height {height}"
        )

    def mine_block(
        self, height: int, previous_hash: str, timer: Timer
    ) -> tuple[Block, BlockRecord]:
        """Produce one block and have every node validate it."""
        tx_root = sha256_hex(f"tx-root:{height}".encode())
        winner, pow_result, simulated_seconds, _, network_attempts = self._race(
            height, previous_hash, tx_root, timer
        )
        block, timings = winner.mine(
            height=height,
            previous_hash=previous_hash,
            tx_root=tx_root,
            target=self.target,
            repository=self.repository,
            solver=self.solver,
            max_attempts=self.max_attempts,
            quality_threshold=self.validators[0].quality_threshold,
            cost_tolerance=self.validators[0].cost_tolerance,
            timer=timer,
            pow_result=pow_result,
        )

        validation_seconds: dict[int, float] = {}
        step_totals: dict[str, float] = {}
        all_accepted = True
        for validator in self.validators:
            result = validator.validate(
                block, expected_height=height, expected_parent=previous_hash
            )
            validation_seconds[validator.validator_id] = result.total_seconds
            for label, value in result.step_seconds.items():
                step_totals[label] = step_totals.get(label, 0.0) + value
            if not result.accepted:
                all_accepted = False
                LOGGER.error(
                    "block at height %d rejected by validator %d: %s",
                    height,
                    validator.validator_id,
                    result.reason,
                )
        num_validators = len(self.validators)
        step_means = {label: total / num_validators for label, total in step_totals.items()}

        record = BlockRecord(
            height=height,
            winner_id=winner.miner_id,
            pow_seconds=timings["pow_s"],
            pow_attempts=int(timings["pow_attempts"]),
            network_attempts=network_attempts,
            simulated_seconds=simulated_seconds,
            vrf_prove_seconds=timings["vrf_prove_s"],
            merkle_proof_seconds=timings["merkle_proof_s"],
            solve_seconds=timings["solve_s"],
            package_seconds=timings["package_s"],
            solver_operations=int(timings["solver_operations"]),
            instance_id=block.instance_id,
            instance_size=block.instance.size,
            solution_cost=block.solution_cost,
            block_hash=block.block_hash(),
            validation_seconds=validation_seconds,
            validation_steps=step_means,
            all_accepted=all_accepted,
        )
        return block, record

    # -- full run ------------------------------------------------------------

    def run(
        self,
        num_blocks: int,
        log_every: int = 50,
        on_block: Any = None,
    ) -> SimulationResult:
        """Mine ``num_blocks`` blocks, validating each at every node."""
        result = SimulationResult(repository_summary=self.repository.summary())
        result.blocks_won = {miner.miner_id: 0 for miner in self.miners}
        previous_hash = GENESIS_HASH
        node_tips = {validator.validator_id: GENESIS_HASH for validator in self.validators}

        for height in range(1, num_blocks + 1):
            try:
                block, record = self.mine_block(height, previous_hash, result.timer)
            except MiningError as error:
                LOGGER.error("mining failed at height %d: %s", height, error)
                break
            result.blocks.append(record)
            result.blocks_won[record.winner_id] += 1
            if record.all_accepted:
                previous_hash = record.block_hash
                for validator_id in node_tips:
                    node_tips[validator_id] = record.block_hash
            else:
                result.rejected += 1
            if on_block is not None:
                on_block(height, record)
            if log_every and height % log_every == 0:
                LOGGER.info(
                    "height %d/%d | winner=%d | pow=%.4fs solve=%.4fs | won=%s",
                    height,
                    num_blocks,
                    record.winner_id,
                    record.pow_seconds,
                    record.solve_seconds,
                    result.blocks_won,
                )

        result.node_tips = node_tips
        result.converged = len(set(node_tips.values())) == 1
        return result
