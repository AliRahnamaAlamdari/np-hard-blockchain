"""Tests for the hash gate, the Merkle tree, and the three VRF backends."""

from __future__ import annotations

import hashlib

import pytest

from pouw.crypto import ed25519 as ed
from pouw.crypto.hashing import (
    canonical_json,
    difficulty_to_target,
    expected_attempts,
    meets_target,
    sha256,
    target_to_probability,
)
from pouw.crypto.merkle import (
    MerkleTree,
    hash_internal,
    hash_leaf,
    verify_membership,
    verify_proof,
)
from pouw.crypto.vrf import build_vrf


class TestHashing:
    def test_sha256_matches_hashlib(self):
        assert sha256(b"abc") == hashlib.sha256(b"abc").digest()

    def test_chunks_are_concatenated_not_combined(self):
        assert sha256(b"ab", b"c") == sha256(b"abc")

    def test_canonical_json_is_key_order_independent(self):
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_canonical_json_has_no_incidental_whitespace(self):
        assert canonical_json({"a": 1}) == b'{"a":1}'

    @pytest.mark.parametrize("bits,expected", [(0, 2**256), (1, 2**255), (16, 2**240)])
    def test_difficulty_to_target_is_a_power_of_two(self, bits, expected):
        assert difficulty_to_target(bits) == expected

    def test_probability_halves_per_difficulty_bit(self):
        assert target_to_probability(difficulty_to_target(10)) == pytest.approx(2**-10)

    def test_expected_attempts_is_one_over_p(self):
        # The paper's stated cost of producing one valid seed (L375).
        for bits in (4, 10, 20):
            probability = target_to_probability(difficulty_to_target(bits))
            assert expected_attempts(bits) == pytest.approx(1 / probability)

    def test_difficulty_zero_accepts_every_digest(self):
        target = difficulty_to_target(0)
        assert meets_target(b"\xff" * 32, target)

    def test_high_difficulty_rejects_a_high_digest(self):
        assert not meets_target(b"\xff" * 32, difficulty_to_target(8))

    def test_meets_target_boundary_is_strict(self):
        target = difficulty_to_target(8)
        just_below = (target - 1).to_bytes(32, "big")
        exactly = target.to_bytes(32, "big")
        assert meets_target(just_below, target)
        assert not meets_target(exactly, target)

    def test_out_of_range_difficulty_is_rejected(self):
        with pytest.raises(ValueError):
            difficulty_to_target(257)


class TestMerkle:
    def test_single_leaf_root_is_the_leaf_hash(self):
        tree = MerkleTree(["only"])
        assert tree.root == hash_leaf("only")

    def test_two_leaf_root_is_hand_computable(self):
        tree = MerkleTree(["a", "b"])
        assert tree.root == hash_internal(hash_leaf("a"), hash_leaf("b"))

    def test_leaf_and_internal_hashes_are_domain_separated(self):
        # Without distinct prefixes an internal node could be passed off as a leaf,
        # which is the second-preimage attack the repository must resist.
        left, right = hash_leaf("a"), hash_leaf("b")
        assert hash_internal(left, right) != hash_leaf([left.hex(), right.hex()])
        assert hash_leaf(b"x".hex()) != hash_internal(b"", b"")

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 16, 17])
    def test_every_proof_verifies(self, size):
        payloads = [f"item-{index}" for index in range(size)]
        tree = MerkleTree(payloads)
        for index in range(size):
            proof = tree.prove(index)
            assert verify_proof(tree.root, proof)
            assert verify_membership(tree.root, payloads[index], proof)

    def test_proof_fails_against_a_different_root(self):
        tree = MerkleTree(["a", "b", "c", "d"])
        other = MerkleTree(["a", "b", "c", "e"])
        assert not verify_proof(other.root, tree.prove(3))

    def test_membership_rejects_a_swapped_payload(self):
        payloads = ["a", "b", "c", "d"]
        tree = MerkleTree(payloads)
        proof = tree.prove(1)
        # A valid path, but claimed for the wrong payload -- task substitution.
        assert not verify_membership(tree.root, "different", proof)

    def test_tampered_sibling_is_rejected(self):
        tree = MerkleTree(["a", "b", "c", "d"])
        proof = tree.prove(0)
        broken = proof.__class__(
            index=proof.index,
            leaf=proof.leaf,
            siblings=((b"\x00" * 32, proof.siblings[0][1]),) + proof.siblings[1:],
        )
        assert not verify_proof(tree.root, broken)

    def test_depth_is_logarithmic(self):
        assert MerkleTree([str(i) for i in range(256)]).depth == 8
        assert MerkleTree([str(i) for i in range(1024)]).depth == 10

    def test_empty_repository_is_rejected(self):
        with pytest.raises(ValueError):
            MerkleTree([])

    def test_out_of_range_index_is_rejected(self):
        with pytest.raises(IndexError):
            MerkleTree(["a", "b"]).prove(5)


class TestEd25519:
    def test_base_point_has_the_expected_order(self):
        assert ed.is_identity(ed.scalar_mult(ed.Q, ed.BASE))

    def test_encode_decode_roundtrip(self):
        point = ed.scalar_mult_base(12345)
        decoded = ed.decode_point(ed.encode_point(point))
        assert decoded is not None and ed.equal(decoded, point)

    def test_addition_is_commutative(self):
        a = ed.scalar_mult_base(7)
        b = ed.scalar_mult_base(11)
        assert ed.equal(ed.add(a, b), ed.add(b, a))

    def test_scalar_multiplication_is_additive(self):
        assert ed.equal(
            ed.scalar_mult_base(7 + 11),
            ed.add(ed.scalar_mult_base(7), ed.scalar_mult_base(11)),
        )

    def test_subtracting_a_point_from_itself_gives_identity(self):
        point = ed.scalar_mult_base(99)
        assert ed.is_identity(ed.subtract(point, point))

    def test_invalid_encoding_is_rejected(self):
        assert ed.decode_point(b"\xff" * 32) is None
        assert ed.decode_point(b"\x00" * 31) is None

    def test_base_point_is_not_small_order(self):
        assert not ed.is_small_order(ed.BASE)

    def test_identity_is_small_order(self):
        assert ed.is_small_order(ed.IDENTITY)


class TestVRFBackends:
    def test_prove_verify_roundtrip(self, any_vrf):
        keys = any_vrf.keygen(b"seed")
        beta, proof = any_vrf.prove(keys.secret, b"alpha")
        assert any_vrf.verify(keys.public, b"alpha", proof) == beta

    def test_output_is_deterministic(self, any_vrf):
        keys = any_vrf.keygen(b"seed")
        first, _ = any_vrf.prove(keys.secret, b"alpha")
        second, _ = any_vrf.prove(keys.secret, b"alpha")
        assert first == second

    def test_different_inputs_give_different_outputs(self, any_vrf):
        keys = any_vrf.keygen(b"seed")
        first, _ = any_vrf.prove(keys.secret, b"alpha-1")
        second, _ = any_vrf.prove(keys.secret, b"alpha-2")
        assert first != second

    def test_keygen_is_deterministic(self, any_vrf):
        assert any_vrf.keygen(b"seed").public == any_vrf.keygen(b"seed").public

    def test_distinct_seeds_give_distinct_keys(self, any_vrf):
        assert any_vrf.keygen(b"a").public != any_vrf.keygen(b"b").public


class TestEcVrfSecurity:
    """Properties that only the real VRF has -- this is the load-bearing difference."""

    def test_rejects_a_proof_for_a_different_input(self, real_vrf):
        keys = real_vrf.keygen(b"seed")
        _, proof = real_vrf.prove(keys.secret, b"alpha")
        assert real_vrf.verify(keys.public, b"other-alpha", proof) is None

    def test_rejects_a_proof_under_a_different_key(self, real_vrf):
        alice = real_vrf.keygen(b"alice")
        bob = real_vrf.keygen(b"bob")
        _, proof = real_vrf.prove(alice.secret, b"alpha")
        assert real_vrf.verify(bob.public, b"alpha", proof) is None

    def test_rejects_a_tampered_proof(self, real_vrf):
        keys = real_vrf.keygen(b"seed")
        _, proof = real_vrf.prove(keys.secret, b"alpha")
        tampered = bytes([proof[0] ^ 0x01]) + proof[1:]
        assert real_vrf.verify(keys.public, b"alpha", tampered) is None

    def test_rejects_a_malformed_proof_length(self, real_vrf):
        keys = real_vrf.keygen(b"seed")
        assert real_vrf.verify(keys.public, b"alpha", b"\x00" * 10) is None

    def test_proof_has_the_rfc_9381_length(self, real_vrf):
        keys = real_vrf.keygen(b"seed")
        _, proof = real_vrf.prove(keys.secret, b"alpha")
        assert len(proof) == 32 + 16 + 32  # gamma || c (cLen) || s (qLen)

    def test_beta_has_the_rfc_9381_length(self, real_vrf):
        keys = real_vrf.keygen(b"seed")
        beta, _ = real_vrf.prove(keys.secret, b"alpha")
        assert len(beta) == 64  # SHA-512 output

    def test_output_is_unique(self, real_vrf):
        assert real_vrf.is_unique_output()


class TestMalleableBackends:
    """The paper's stated primitive is not unique-output; these tests pin that."""

    def test_ecdsa_pseudo_vrf_admits_many_outputs_per_input(self):
        vrf = build_vrf("ecdsa_pseudo")
        keys = vrf.keygen(b"seed")
        outputs = set()
        for attempt in range(8):
            beta, proof = vrf.prove(keys.secret, b"alpha", attempt=attempt)
            # Each one is a *valid* proof, which is what makes this an attack.
            assert vrf.verify(keys.public, b"alpha", proof) == beta
            outputs.add(beta)
        assert len(outputs) == 8
        assert not vrf.is_unique_output()

    def test_hash_stub_verifies_without_the_public_key(self):
        vrf = build_vrf("hash_stub")
        keys = vrf.keygen(b"seed")
        _, proof = vrf.prove(keys.secret, b"alpha")
        # Any key and any input verify -- the check is keyless.
        assert vrf.verify(b"any-key-at-all", b"unrelated-input", proof) is not None
        assert not vrf.is_unique_output()


def test_unknown_backend_is_rejected():
    from pouw.crypto.vrf import VRFError

    with pytest.raises(VRFError):
        build_vrf("does_not_exist")
