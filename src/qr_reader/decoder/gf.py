"""GF(256) arithmetic for QR Code Reed-Solomon error correction.

Uses the primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)
with generator α = 0x02.  Precomputes log/exp tables for fast
multiplication and division.

Reference: zxing-cpp GenericGF.h / GenericGF.cpp
"""


def _build_tables():
    """Build the 256-entry log table and 512-entry exp table."""
    exp = [0] * 512
    log = [0] * 256

    exp[0] = 1
    for i in range(1, 256):
        val = exp[i - 1] << 1
        if val >= 256:
            val ^= 0x11D
        exp[i] = val

    # Duplicate for bounds-free multiplication lookups
    # log values are 0..254, so max sum is 508.  The duplicated
    # range at exp[255..509] gives exp[(log_a + log_b)] without a modulo.
    for i in range(255, 512):
        exp[i] = exp[i - 255]

    # Build log table: log[α^i] = i for i in [0, 254] (α^255 = α^0 = 1)
    for i in range(255):
        log[exp[i]] = i

    return exp, log


_EXP, _LOG = _build_tables()


class GF256:
    """GF(256) field with the QR-code primitive polynomial 0x11D."""

    @staticmethod
    def add(a: int, b: int) -> int:
        """Addition (and subtraction) in GF(256) — XOR."""
        return a ^ b

    @staticmethod
    def subtract(a: int, b: int) -> int:
        """Same as add."""
        return a ^ b

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply two field elements using log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return _EXP[_LOG[a] + _LOG[b]]

    @staticmethod
    def inverse(a: int) -> int:
        """Multiplicative inverse.  Raises ValueError on a == 0."""
        if a == 0:
            raise ValueError("Cannot invert zero in GF(256)")
        # inv(α^i) = α^(255 - i)
        return _EXP[255 - _LOG[a]]

    @staticmethod
    def pow(a: int, n: int) -> int:
        """Raise a field element to an integer power."""
        if n == 0:
            return 1
        if a == 0:
            return 0
        # a^n = α^(log[a] * n mod 255)
        log_a = _LOG[a]
        log_result = (log_a * n) % 255
        return _EXP[log_result]

    @staticmethod
    def exp(n: int) -> int:
        """Return α^n (for generator polynomial construction)."""
        return _EXP[n % 255]

    @staticmethod
    def log(a: int) -> int:
        """Return i where α^i = a.  Raises ValueError on a == 0."""
        if a == 0:
            raise ValueError("Cannot take log of zero in GF(256)")
        return _LOG[a]
