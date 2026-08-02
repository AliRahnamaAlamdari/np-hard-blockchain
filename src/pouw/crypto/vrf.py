"""Verifiable Random Function backends used for Stage 2 task assignment.

The paper (L405) states only that the prototype uses "deterministic VRF-like signing
and verification" backed by "an ECDSA library" (L412), and never names a VRF
construction.  Because ``PAPER_SPEC.md`` §11-I6 shows that this distinction decides
whether design goals G3 and G4 hold at all, three backends are provided and selected
by config:

``ecvrf``
    RFC 9381 ECVRF-EDWARDS25519-SHA512-TAI.  A real VRF: the output is *unique* per
    (key, input) pair, so a miner who holds one PoW seed can derive exactly one task.

``ecdsa_pseudo``
    The paper-faithful reading of "VRF-like signing".  ECDSA signatures are malleable
    in the per-signature nonce, so a miner can re-sign the same seed repeatedly and
    obtain a fresh task index each time at no PoW cost.  Grindable by construction.

``hash_stub``
    A keyless hash chain whose verification cost matches the sub-microsecond latency
    reported in the paper's validation-cost table.  Included so the reported timings
    can be reproduced and so their implications can be measured.  Provides no key
    binding whatsoever.

All backends share one interface, so the mining and validation code is identical
across them and the ablation isolates the primitive.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol

from . import ed25519 as ed

SUITE_STRING = b"\x04"  # ECVRF-EDWARDS25519-SHA512-TAI
_C_LEN = 16
_Q_LEN = 32
_PT_LEN = 32


class VRFError(Exception):
    """Raised when a proof is malformed (as opposed to merely invalid)."""


@dataclass(frozen=True)
class KeyPair:
    secret: bytes
    public: bytes


class VRFBackend(Protocol):
    name: str

    def keygen(self, seed: bytes) -> KeyPair: ...

    def prove(self, secret: bytes, alpha: bytes) -> tuple[bytes, bytes]:
        """Return ``(beta, pi)``: the VRF output and its proof."""

    def verify(self, public: bytes, alpha: bytes, pi: bytes) -> bytes | None:
        """Return ``beta`` if the proof is valid for ``alpha`` under ``public``."""

    def is_unique_output(self) -> bool:
        """True when one (key, input) pair admits exactly one acceptable output."""


# ---------------------------------------------------------------------------
# RFC 9381 ECVRF-EDWARDS25519-SHA512-TAI
# ---------------------------------------------------------------------------


def _sha512(*chunks: bytes) -> bytes:
    digest = hashlib.sha512()
    for chunk in chunks:
        digest.update(chunk)
    return digest.digest()


def _expand_secret(secret: bytes) -> tuple[int, bytes, bytes]:
    """Derive the scalar, the nonce-generation half, and the encoded public key."""
    if len(secret) != 32:
        raise VRFError(f"ed25519 secret must be 32 bytes, got {len(secret)}")
    hashed = _sha512(secret)
    scalar_bytes = bytearray(hashed[:32])
    scalar_bytes[0] &= 248
    scalar_bytes[31] &= 127
    scalar_bytes[31] |= 64
    scalar = int.from_bytes(scalar_bytes, "little")
    public = ed.encode_point(ed.scalar_mult_base(scalar))
    return scalar, hashed[32:], public


def _encode_to_curve_tai(salt: bytes, alpha: bytes) -> ed.Point:
    """RFC 9381 section 5.4.1.1 -- try-and-increment hash to curve."""
    for ctr in range(256):
        candidate = _sha512(
            SUITE_STRING, b"\x01", salt, alpha, bytes([ctr]), b"\x00"
        )[:32]
        point = ed.decode_point(candidate)
        if point is None:
            continue
        point = ed.clear_cofactor(point)
        if not ed.is_identity(point):
            return point
    raise VRFError("encode_to_curve failed to find a point in 256 attempts")


def _challenge(points: list[ed.Point]) -> int:
    """RFC 9381 section 5.4.3 -- challenge generation over five points."""
    buffer = [SUITE_STRING, b"\x02"]
    buffer.extend(ed.encode_point(point) for point in points)
    buffer.append(b"\x00")
    return ed.string_to_int(_sha512(*buffer)[:_C_LEN])


def _proof_to_hash(gamma: ed.Point) -> bytes:
    return _sha512(
        SUITE_STRING, b"\x03", ed.encode_point(ed.clear_cofactor(gamma)), b"\x00"
    )


class EcVrfEd25519:
    """RFC 9381 ECVRF-EDWARDS25519-SHA512-TAI.  Unique-output, publicly verifiable."""

    name = "ecvrf"
    beta_length = 64

    def keygen(self, seed: bytes) -> KeyPair:
        secret = hashlib.sha256(b"pouw-ecvrf-key" + seed).digest()
        _, _, public = _expand_secret(secret)
        return KeyPair(secret=secret, public=public)

    def prove(self, secret: bytes, alpha: bytes) -> tuple[bytes, bytes]:
        scalar, nonce_half, public = _expand_secret(secret)
        h_point = _encode_to_curve_tai(public, alpha)
        h_string = ed.encode_point(h_point)
        gamma = ed.scalar_mult(scalar, h_point)
        k = ed.string_to_int(_sha512(nonce_half, h_string)) % ed.Q
        challenge = _challenge(
            [
                ed.decode_point(public),  # type: ignore[list-item]
                h_point,
                gamma,
                ed.scalar_mult_base(k),
                ed.scalar_mult(k, h_point),
            ]
        )
        s = (k + challenge * scalar) % ed.Q
        pi = (
            ed.encode_point(gamma)
            + ed.int_to_string(challenge, _C_LEN)
            + ed.int_to_string(s, _Q_LEN)
        )
        return _proof_to_hash(gamma), pi

    def verify(self, public: bytes, alpha: bytes, pi: bytes) -> bytes | None:
        if len(pi) != _PT_LEN + _C_LEN + _Q_LEN:
            return None
        gamma = ed.decode_point(pi[:_PT_LEN])
        if gamma is None:
            return None
        challenge = ed.string_to_int(pi[_PT_LEN : _PT_LEN + _C_LEN])
        s = ed.string_to_int(pi[_PT_LEN + _C_LEN :])
        if s >= ed.Q:
            return None
        public_point = ed.decode_point(public)
        if public_point is None or ed.is_small_order(public_point):
            return None
        h_point = _encode_to_curve_tai(public, alpha)
        u = ed.subtract(ed.scalar_mult_base(s), ed.scalar_mult(challenge, public_point))
        v = ed.subtract(ed.scalar_mult(s, h_point), ed.scalar_mult(challenge, gamma))
        if _challenge([public_point, h_point, gamma, u, v]) != challenge:
            return None
        return _proof_to_hash(gamma)

    def is_unique_output(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Paper-faithful malleable backend
# ---------------------------------------------------------------------------


class EcdsaPseudoVrf:
    """"VRF-like signing" built on ECDSA, matching the paper's stated primitives.

    ECDSA admits many valid signatures per message because the per-signature nonce is
    free.  ``prove`` therefore accepts an optional ``attempt`` value that varies that
    nonce, which is exactly the capability an adversary has.  Every resulting proof
    verifies, and each yields a different VRF output -- so a single PoW seed can be
    ground into arbitrarily many task assignments.
    """

    name = "ecdsa_pseudo"
    beta_length = 32

    def __init__(self, curve_name: str = "SECP256k1") -> None:
        import ecdsa  # imported lazily so the ecvrf path has no hard dependency

        self._ecdsa = ecdsa
        self._curve = getattr(ecdsa, curve_name)

    def keygen(self, seed: bytes) -> KeyPair:
        material = hashlib.sha256(b"pouw-ecdsa-key" + seed).digest()
        signing_key = self._ecdsa.SigningKey.from_string(material, curve=self._curve)
        return KeyPair(
            secret=signing_key.to_string(),
            public=signing_key.get_verifying_key().to_string(),
        )

    def prove(
        self, secret: bytes, alpha: bytes, attempt: int = 0
    ) -> tuple[bytes, bytes]:
        signing_key = self._ecdsa.SigningKey.from_string(secret, curve=self._curve)
        if attempt == 0:
            # RFC 6979 deterministic nonce: the honest, "deterministic" reading.
            signature = signing_key.sign_digest_deterministic(
                hashlib.sha256(alpha).digest(), hashfunc=hashlib.sha256
            )
        else:
            # Any other nonce is equally valid -- this is the malleability.
            nonce = (
                int.from_bytes(
                    hashlib.sha256(secret + alpha + attempt.to_bytes(8, "big")).digest(),
                    "big",
                )
                % (self._curve.order - 1)
            ) + 1
            signature = signing_key.sign_digest(
                hashlib.sha256(alpha).digest(), k=nonce
            )
        return hashlib.sha256(signature).digest(), signature

    def verify(self, public: bytes, alpha: bytes, pi: bytes) -> bytes | None:
        verifying_key = self._ecdsa.VerifyingKey.from_string(public, curve=self._curve)
        try:
            verifying_key.verify_digest(pi, hashlib.sha256(alpha).digest())
        except self._ecdsa.BadSignatureError:
            return None
        return hashlib.sha256(pi).digest()

    def is_unique_output(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Timing-plausible reconstruction of the paper's reported latencies
# ---------------------------------------------------------------------------


class HashStubVrf:
    """Keyless hash chain, included to reproduce the paper's reported timings.

    The paper reports 0.001958 ms (1.958 us) for "VRF Proof Verification".  No
    elliptic-curve verification runs that fast in CPython by three orders of
    magnitude, so a construction of roughly this shape is what those numbers describe.
    Verification recomputes a hash of the proof and compares it to the claimed output;
    it never touches the public key, so any party can forge any output for any key.
    """

    name = "hash_stub"
    beta_length = 32

    def keygen(self, seed: bytes) -> KeyPair:
        secret = hashlib.sha256(b"pouw-stub-key" + seed).digest()
        return KeyPair(secret=secret, public=hashlib.sha256(secret).digest())

    def prove(self, secret: bytes, alpha: bytes) -> tuple[bytes, bytes]:
        pi = hashlib.sha256(secret + alpha).digest()
        return hashlib.sha256(pi).digest(), pi

    def verify(self, public: bytes, alpha: bytes, pi: bytes) -> bytes | None:
        del public, alpha  # deliberately unused -- that is the point being measured
        return hashlib.sha256(pi).digest()

    def is_unique_output(self) -> bool:
        return False


_BACKENDS = {
    "ecvrf": EcVrfEd25519,
    "ecdsa_pseudo": EcdsaPseudoVrf,
    "hash_stub": HashStubVrf,
}


def build_vrf(name: str, **kwargs: object) -> VRFBackend:
    """Instantiate a VRF backend by config name."""
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise VRFError(
            f"unknown vrf backend {name!r}; expected one of {sorted(_BACKENDS)}"
        ) from None
    return factory(**kwargs)  # type: ignore[arg-type]


def random_seed(num_bytes: int = 32) -> bytes:
    return os.urandom(num_bytes)
