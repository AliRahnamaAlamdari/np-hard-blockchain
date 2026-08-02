"""Binary Merkle tree used to commit the problem repository.

The paper specifies neither the hash, the leaf encoding, the arity, nor the
odd-node rule (``PAPER_SPEC.md`` §11-U6).  This implementation domain-separates leaf
and internal hashes with distinct prefixes, which is what stops the second-preimage
attack that would otherwise let a "task substitution" adversary swap an internal node
for a leaf -- precisely the attack Table 1 of the paper claims the repository
mitigates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hashing import canonical_json, sha256

LEAF_PREFIX = b"\x00"
INTERNAL_PREFIX = b"\x01"


def hash_leaf(payload: Any) -> bytes:
    """Hash a repository entry via its canonical JSON encoding."""
    return sha256(LEAF_PREFIX, canonical_json(payload))


def hash_internal(left: bytes, right: bytes) -> bytes:
    return sha256(INTERNAL_PREFIX, left, right)


@dataclass(frozen=True)
class MerkleProof:
    """A membership proof: the sibling path from a leaf up to the root."""

    index: int
    leaf: bytes
    siblings: tuple[tuple[bytes, bool], ...]  # (sibling_hash, sibling_is_left)

    @property
    def depth(self) -> int:
        return len(self.siblings)

    def size_bytes(self) -> int:
        return sum(len(sibling) + 1 for sibling, _ in self.siblings)


class MerkleTree:
    """Binary Merkle tree with last-node duplication on odd levels."""

    def __init__(self, payloads: list[Any]) -> None:
        if not payloads:
            raise ValueError("cannot build a Merkle tree over an empty repository")
        self._leaves = [hash_leaf(payload) for payload in payloads]
        self._levels: list[list[bytes]] = [list(self._leaves)]
        while len(self._levels[-1]) > 1:
            current = self._levels[-1]
            if len(current) % 2 == 1:
                current = current + [current[-1]]
            self._levels.append(
                [
                    hash_internal(current[i], current[i + 1])
                    for i in range(0, len(current), 2)
                ]
            )

    @property
    def root(self) -> bytes:
        return self._levels[-1][0]

    @property
    def num_leaves(self) -> int:
        return len(self._leaves)

    @property
    def depth(self) -> int:
        return len(self._levels) - 1

    def leaf(self, index: int) -> bytes:
        return self._leaves[index]

    def prove(self, index: int) -> MerkleProof:
        """Produce the membership proof for the leaf at ``index``."""
        if not 0 <= index < self.num_leaves:
            raise IndexError(f"leaf index {index} out of range [0, {self.num_leaves})")
        siblings: list[tuple[bytes, bool]] = []
        position = index
        for level in self._levels[:-1]:
            padded = level + [level[-1]] if len(level) % 2 == 1 else level
            if position % 2 == 0:
                siblings.append((padded[position + 1], False))
            else:
                siblings.append((padded[position - 1], True))
            position //= 2
        return MerkleProof(index=index, leaf=self._leaves[index], siblings=tuple(siblings))


def verify_proof(root: bytes, proof: MerkleProof) -> bool:
    """Recompute the root from a proof and compare it to the committed root."""
    node = proof.leaf
    for sibling, sibling_is_left in proof.siblings:
        node = (
            hash_internal(sibling, node)
            if sibling_is_left
            else hash_internal(node, sibling)
        )
    return node == root


def verify_membership(root: bytes, payload: Any, proof: MerkleProof) -> bool:
    """Full check: the proof is well-formed *and* its leaf is the claimed payload."""
    return proof.leaf == hash_leaf(payload) and verify_proof(root, proof)
