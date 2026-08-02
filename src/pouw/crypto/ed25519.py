"""Pure-Python Edwards25519 group arithmetic.

Written from the RFC 8032 specification so that the ECVRF in :mod:`pouw.crypto.vrf`
depends on no compiled extension and stays auditable alongside the paper.  Speed is
adequate for the simulator (a scalar multiplication costs on the order of a
millisecond); this is deliberately *not* a constant-time implementation and must not
be used to protect real keys.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Curve parameters (RFC 8032 section 5.1)
# ---------------------------------------------------------------------------

P = 2**255 - 19
Q = 2**252 + 27742317777372353535851937790883648493
COFACTOR = 8
_D = (-121665 * pow(121666, P - 2, P)) % P
_SQRT_M1 = pow(2, (P - 1) // 4, P)

# Extended homogeneous coordinates: (X, Y, Z, T) with x = X/Z, y = Y/Z, xy = T/Z.
Point = tuple[int, int, int, int]

IDENTITY: Point = (0, 1, 1, 0)


def _recover_x(y: int, sign: int) -> int | None:
    """Recover the x coordinate matching ``y`` with low bit ``sign``, or None."""
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, P - 2, P) % P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * _SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if x & 1 != sign:
        x = P - x
    return x


_BASE_Y = 4 * pow(5, P - 2, P) % P
_BASE_X = _recover_x(_BASE_Y, 0)
assert _BASE_X is not None
BASE: Point = (_BASE_X, _BASE_Y, 1, _BASE_X * _BASE_Y % P)


def add(p: Point, q: Point) -> Point:
    """Add two points using the unified extended-coordinate formulas."""
    a_x, a_y, a_z, a_t = p
    b_x, b_y, b_z, b_t = q
    a = (a_y - a_x) * (b_y - b_x) % P
    b = (a_y + a_x) * (b_y + b_x) % P
    c = 2 * a_t * b_t * _D % P
    d = 2 * a_z * b_z % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def double(p: Point) -> Point:
    return add(p, p)


def scalar_mult(k: int, p: Point) -> Point:
    """Compute ``k * p`` by double-and-add (variable time)."""
    result = IDENTITY
    addend = p
    k %= Q * COFACTOR if k >= 0 else Q * COFACTOR
    while k > 0:
        if k & 1:
            result = add(result, addend)
        addend = double(addend)
        k >>= 1
    return result


def scalar_mult_base(k: int) -> Point:
    return scalar_mult(k, BASE)


def negate(p: Point) -> Point:
    x, y, z, t = p
    return ((-x) % P, y, z, (-t) % P)


def subtract(p: Point, q: Point) -> Point:
    return add(p, negate(q))


def equal(p: Point, q: Point) -> bool:
    """Compare two points projectively."""
    a_x, a_y, a_z, _ = p
    b_x, b_y, b_z, _ = q
    return (a_x * b_z - b_x * a_z) % P == 0 and (a_y * b_z - b_y * a_z) % P == 0


def is_identity(p: Point) -> bool:
    return equal(p, IDENTITY)


def encode_point(p: Point) -> bytes:
    """Serialise to the 32-byte RFC 8032 compressed form."""
    x, y, z, _ = p
    z_inv = pow(z, P - 2, P)
    x = x * z_inv % P
    y = y * z_inv % P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def decode_point(data: bytes) -> Point | None:
    """Parse a 32-byte compressed point, returning None if it is not on the curve."""
    if len(data) != 32:
        return None
    value = int.from_bytes(data, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % P)


def is_small_order(p: Point) -> bool:
    """True if the point lies in the order-8 torsion subgroup."""
    return is_identity(scalar_mult(COFACTOR, p))


def clear_cofactor(p: Point) -> Point:
    return scalar_mult(COFACTOR, p)


def string_to_int(data: bytes) -> int:
    """RFC 9381 string_to_int: little-endian for the edwards25519 suites."""
    return int.from_bytes(data, "little")


def int_to_string(value: int, length: int) -> bytes:
    return int.to_bytes(value % (1 << (8 * length)), length, "little")
