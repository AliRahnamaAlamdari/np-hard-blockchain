"""Hash helpers and the Stage-1 proof-of-work target encoding.

The paper never names its hash function or its difficulty encoding
(``PAPER_SPEC.md`` §11-A1, §11-U4).  SHA-256 with a leading-zero-*bit* target is the
Bitcoin-conventional choice and is what this module implements; the number of bits is
a config value, never a literal in code.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DIGEST_BITS = 256


def sha256(*chunks: bytes) -> bytes:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.digest()


def sha256_hex(*chunks: bytes) -> str:
    return sha256(*chunks).hex()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used for every hashed structure.

    Sorted keys and no incidental whitespace, so that a digest depends on the value
    rather than on dict insertion order or formatting.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def difficulty_to_target(difficulty_bits: int) -> int:
    """Convert a leading-zero-bit count to the integer target of Equation (1).

    A digest satisfies the gate when ``int(digest) < target``, which happens with
    probability ``2**-difficulty_bits`` for a uniform digest.
    """
    if not 0 <= difficulty_bits <= DIGEST_BITS:
        raise ValueError(f"difficulty_bits must be in [0, {DIGEST_BITS}]")
    return 1 << (DIGEST_BITS - difficulty_bits)


def target_to_probability(target: int) -> float:
    """Success probability ``p`` of a single hash attempt (see paper L375)."""
    return target / (1 << DIGEST_BITS)


def expected_attempts(difficulty_bits: int) -> float:
    """``1/p``: the paper's stated expected cost of producing one valid seed."""
    return float(1 << difficulty_bits)


def meets_target(digest: bytes, target: int) -> bool:
    return int.from_bytes(digest, "big") < target
