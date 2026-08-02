"""Tests for mining (Algorithm 1), validation (Algorithm 2), and the simulator.

The validation tests are adversarial: each one tampers with exactly one field of an
otherwise valid block and asserts the specific check that must catch it.  That is what
makes Algorithm 2's elided arguments (``PAPER_SPEC.md`` §11-U1) concrete.
"""

from __future__ import annotations

import pytest

from pouw.chain.block import Block, BlockHeader
from pouw.chain.miner import Miner, MiningError
from pouw.chain.network import GENESIS_HASH, Network
from pouw.chain.validator import RejectReason, Validator
from pouw.crypto.hashing import difficulty_to_target, meets_target
from pouw.crypto.vrf import build_vrf
from pouw.seeding import seed_everything
from pouw.tasks.tsp import TSPSolution, generate_instance

DIFFICULTY = 8


@pytest.fixture
def rig(small_repository, bundle):
    """A miner, a validator, and one honestly mined block sharing a repository."""
    vrf = build_vrf("ecvrf")
    target = difficulty_to_target(DIFFICULTY)
    miner = Miner(miner_id=0, keys=vrf.keygen(bundle.bytes_for("miner:0")), vrf=vrf)
    validator = Validator(
        validator_id=0, vrf=vrf, repository=small_repository, target=target
    )
    block, timings = miner.mine(
        height=1,
        previous_hash=GENESIS_HASH,
        tx_root="tx",
        target=target,
        repository=small_repository,
        solver="brute_force",
        max_attempts=1_000_000,
    )
    return {
        "vrf": vrf,
        "target": target,
        "miner": miner,
        "validator": validator,
        "block": block,
        "timings": timings,
        "repository": small_repository,
    }


class TestBlockHeader:
    def test_digest_is_deterministic(self):
        header = BlockHeader(1, "p", "t", 0, "k", 5)
        assert header.digest() == header.digest()

    def test_changing_the_nonce_changes_the_digest(self):
        header = BlockHeader(1, "p", "t", 0, "k", 5)
        assert header.with_nonce(6).digest() != header.digest()

    def test_changing_the_miner_key_changes_the_digest(self):
        # The public key is bound into the PoW, so a miner cannot swap keys after
        # finding a nonce.
        a = BlockHeader(1, "p", "t", 0, "key-a", 5)
        b = BlockHeader(1, "p", "t", 0, "key-b", 5)
        assert a.digest() != b.digest()

    def test_payload_roundtrip(self):
        header = BlockHeader(3, "p", "t", 2, "k", 9)
        assert BlockHeader.from_payload(header.to_payload()) == header


class TestMining:
    def test_pow_result_satisfies_the_target(self, rig):
        digest = rig["block"].header.digest()
        assert meets_target(digest, rig["target"])

    def test_returned_nonce_is_the_winning_nonce(self, rig):
        # PAPER_SPEC.md 11-I2: the paper's Algorithm 1 returns nonce+1, which would
        # make every honest block fail validation.
        block = rig["block"]
        recomputed = block.header.digest()
        assert recomputed.hex() == block.pow_hash
        assert meets_target(recomputed, rig["target"])

    def test_instance_index_follows_equation_three(self, rig):
        from pouw.tasks.repository import index_from_vrf_output

        block = rig["block"]
        beta = bytes.fromhex(block.vrf_output)
        assert block.instance_id == index_from_vrf_output(beta, len(rig["repository"]))

    def test_solution_is_feasible_and_costed_correctly(self, rig):
        from pouw.tasks.tsp import is_feasible, tour_cost

        block = rig["block"]
        assert is_feasible(block.instance, block.solution.tour)
        assert block.solution.cost == pytest.approx(
            tour_cost(block.instance, block.solution.tour)
        )

    def test_timings_cover_every_stage(self, rig):
        for key in ("pow_s", "vrf_prove_s", "merkle_proof_s", "solve_s", "package_s"):
            assert rig["timings"][key] >= 0.0

    def test_exhausting_the_nonce_budget_raises(self, small_repository, bundle):
        vrf = build_vrf("hash_stub")
        miner = Miner(miner_id=0, keys=vrf.keygen(b"k"), vrf=vrf)
        with pytest.raises(MiningError):
            miner.search_pow(
                height=1,
                previous_hash="p",
                tx_root="t",
                target=difficulty_to_target(64),
                max_attempts=50,
            )


class TestValidationAcceptsHonestBlocks:
    def test_honest_block_is_accepted(self, rig):
        result = rig["validator"].validate(rig["block"])
        assert result.accepted and result.reason is None

    def test_all_five_steps_are_timed(self, rig):
        result = rig["validator"].validate(rig["block"])
        assert set(result.step_seconds) == {
            "pow_check",
            "vrf_verify",
            "index_check",
            "merkle_verify",
            "solution_check",
        }
        assert result.total_seconds > 0


class TestValidationRejectsTampering:
    """One tampered field per test, each caught by the check responsible for it."""

    def _rebuild(self, block: Block, **changes) -> Block:
        fields = {
            "header": block.header,
            "pow_hash": block.pow_hash,
            "vrf_output": block.vrf_output,
            "vrf_proof": block.vrf_proof,
            "instance_id": block.instance_id,
            "instance": block.instance,
            "merkle_proof": block.merkle_proof,
            "solution": block.solution,
            "solution_cost": block.solution_cost,
        }
        fields.update(changes)
        return Block(**fields)

    def test_insufficient_pow_is_rejected(self, rig):
        harder = Validator(
            validator_id=1,
            vrf=rig["vrf"],
            repository=rig["repository"],
            target=difficulty_to_target(250),
        )
        result = harder.validate(rig["block"])
        assert not result.accepted and result.reason is RejectReason.POW_FAILED

    def test_altered_nonce_is_rejected(self, rig):
        block = rig["block"]
        tampered = self._rebuild(block, header=block.header.with_nonce(block.header.nonce + 1))
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.POW_FAILED

    def test_mismatched_pow_hash_field_is_rejected(self, rig):
        tampered = self._rebuild(rig["block"], pow_hash="00" * 32)
        result = rig["validator"].validate(tampered)
        assert not result.accepted
        assert result.reason in (RejectReason.POW_FAILED, RejectReason.POW_HASH_MISMATCH)

    def test_forged_vrf_proof_is_rejected(self, rig):
        tampered = self._rebuild(rig["block"], vrf_proof="00" * 80)
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.VRF_INVALID

    def test_vrf_proof_from_another_miner_is_rejected(self, rig, bundle):
        # The proof is valid, but not under the public key committed in the header.
        other = Miner(
            miner_id=1, keys=rig["vrf"].keygen(bundle.bytes_for("miner:1")), vrf=rig["vrf"]
        )
        seed = rig["block"].header.digest()
        _, foreign_proof = rig["vrf"].prove(other.keys.secret, seed)
        tampered = self._rebuild(rig["block"], vrf_proof=foreign_proof.hex())
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.VRF_INVALID

    def test_altered_vrf_output_is_rejected(self, rig):
        tampered = self._rebuild(rig["block"], vrf_output="ff" * 64)
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.VRF_OUTPUT_MISMATCH

    def test_wrong_instance_index_is_rejected(self, rig):
        block = rig["block"]
        wrong = (block.instance_id + 1) % len(rig["repository"])
        instance, proof = rig["repository"].get_proof(wrong)
        tampered = self._rebuild(
            block, instance_id=wrong, instance=instance, merkle_proof=proof
        )
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.INDEX_MISMATCH

    def test_substituted_instance_is_rejected_by_membership(self, rig):
        forged = generate_instance("FORGED", size=3, seed=1)
        from pouw.tasks.tsp import brute_force

        solution, _ = brute_force(forged)
        tampered = self._rebuild(
            rig["block"], instance=forged, solution=solution, solution_cost=solution.cost
        )
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.MERKLE_INVALID

    def test_infeasible_solution_is_rejected(self, rig):
        block = rig["block"]
        bad = TSPSolution(tour=(0,) * block.instance.size, cost=block.solution_cost)
        tampered = self._rebuild(block, solution=bad)
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.SOLUTION_INVALID

    def test_understated_cost_is_rejected(self, rig):
        block = rig["block"]
        bad = TSPSolution(tour=block.solution.tour, cost=block.solution.cost * 0.5)
        tampered = self._rebuild(block, solution=bad)
        result = rig["validator"].validate(tampered)
        assert not result.accepted and result.reason is RejectReason.SOLUTION_INVALID

    def test_wrong_height_is_rejected(self, rig):
        result = rig["validator"].validate(rig["block"], expected_height=99)
        assert not result.accepted and result.reason is RejectReason.BAD_HEIGHT

    def test_wrong_parent_is_rejected(self, rig):
        result = rig["validator"].validate(rig["block"], expected_parent="deadbeef")
        assert not result.accepted and result.reason is RejectReason.BAD_PARENT


class TestBlockHash:
    def test_block_hash_is_stable(self, rig):
        assert rig["block"].block_hash() == rig["block"].block_hash()

    def test_any_field_change_changes_the_block_hash(self, rig):
        # Paper Problem Statement condition 5: the block cannot be modified without
        # invalidating the cryptographic linkage.
        block = rig["block"]
        original = block.block_hash()
        mutated = Block(
            header=block.header,
            pow_hash=block.pow_hash,
            vrf_output=block.vrf_output,
            vrf_proof=block.vrf_proof,
            instance_id=block.instance_id,
            instance=block.instance,
            merkle_proof=block.merkle_proof,
            solution=block.solution,
            solution_cost=block.solution_cost + 1.0,
        )
        assert mutated.block_hash() != original


class TestNetwork:
    def test_short_run_produces_accepted_blocks_and_converges(self, small_repository, bundle):
        network = Network(
            num_miners=3,
            repository=small_repository,
            difficulty_bits=DIFFICULTY,
            vrf_backend="ecvrf",
            seeds=bundle,
            solver="brute_force",
        )
        result = network.run(num_blocks=5, log_every=0)
        assert result.num_blocks == 5
        assert result.rejected == 0
        assert result.converged
        assert all(record.all_accepted for record in result.blocks)

    def test_block_counts_sum_to_the_number_of_blocks(self, small_repository, bundle):
        network = Network(
            num_miners=3,
            repository=small_repository,
            difficulty_bits=DIFFICULTY,
            vrf_backend="ecvrf",
            seeds=bundle,
        )
        result = network.run(num_blocks=6, log_every=0)
        assert sum(result.blocks_won.values()) == 6

    def test_chain_is_linked_parent_to_child(self, small_repository, bundle):
        network = Network(
            num_miners=2,
            repository=small_repository,
            difficulty_bits=DIFFICULTY,
            vrf_backend="ecvrf",
            seeds=bundle,
        )
        result = network.run(num_blocks=4, log_every=0)
        heights = [record.height for record in result.blocks]
        assert heights == [1, 2, 3, 4]
        assert len({record.block_hash for record in result.blocks}) == 4

    def test_network_attempts_scale_with_difficulty(self, small_repository, bundle):
        """Total network work per block should be ~2**difficulty_bits, not N times it."""
        easy = Network(
            num_miners=4,
            repository=small_repository,
            difficulty_bits=6,
            vrf_backend="hash_stub",
            seeds=bundle,
        ).run(num_blocks=8, log_every=0)
        hard = Network(
            num_miners=4,
            repository=small_repository,
            difficulty_bits=10,
            vrf_backend="hash_stub",
            seeds=bundle,
        ).run(num_blocks=8, log_every=0)
        easy_mean = sum(r.network_attempts for r in easy.blocks) / len(easy.blocks)
        hard_mean = sum(r.network_attempts for r in hard.blocks) / len(hard.blocks)
        assert hard_mean > easy_mean * 4

    def test_higher_hash_rate_wins_more_blocks(self, small_repository, bundle):
        network = Network(
            num_miners=2,
            repository=small_repository,
            difficulty_bits=6,
            vrf_backend="hash_stub",
            seeds=bundle,
            hash_rates=[1.0, 8.0],
        )
        result = network.run(num_blocks=40, log_every=0)
        assert result.blocks_won[1] > result.blocks_won[0]
