"""Reed-Solomon error correction over GF(256) for QR Code decoding.

Given a block of data + EC bytes (with possible errors), correct the data bytes.

Algorithm (from zxing-cpp ReedSolomonDecoder.cpp):
  1. Syndrome calculation: evaluate received polynomial at α^0..α^(2t-1).
     If all zero → no errors.
  2. Extended Euclidean algorithm: find error locator σ(x) and error
     evaluator ω(x) such that σ(x)·S(x) ≡ ω(x) mod x^(2t)
     with deg(ω) < deg(σ) ≤ t.
  3. Find error locations: brute-force search for roots of σ(x) over GF(256).
  4. Forney's formula for error magnitudes.
  5. XOR correct and re-validate syndromes.

Reference:
  - zxing-cpp GenericGFPoly.h/.cpp
  - zxing-cpp ReedSolomonDecoder.cpp
"""

from __future__ import annotations

from .gf import GF256

# ---------------------------------------------------------------------------
# GF(256) Polynomial — minimal subset required for RS decoding
# ---------------------------------------------------------------------------


class GF256Poly:
    """Polynomial over GF(256), coefficients stored MSB-first (highest degree first).

    Immutable-style: arithmetic methods return new instances.
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: list[int] | None = None):
        if coeffs is None:
            self._coeffs = [0]
        else:
            self._coeffs = list(coeffs)
        self._normalize()

    # -- factories --

    @staticmethod
    def monomial(coefficient: int, degree: int) -> GF256Poly:
        """Create ``coefficient * x^degree``."""
        coeffs = [0] * (degree + 1)
        coeffs[0] = coefficient
        return GF256Poly(coeffs)

    @staticmethod
    def zero() -> GF256Poly:
        return GF256Poly([0])

    # -- queries --

    @property
    def coeffs(self) -> list[int]:
        return self._coeffs

    def degree(self) -> int:
        return len(self._coeffs) - 1

    def is_zero(self) -> bool:
        return self._coeffs == [0]

    def constant(self) -> int:
        return self._coeffs[-1]

    def evaluate_at(self, a: int) -> int:
        """Evaluate this polynomial at x = a (Horner's method)."""
        if a == 0:
            return self.constant()
        result = 0
        for c in self._coeffs:
            result = GF256.multiply(result, a) ^ c
        return result

    # -- arithmetic --

    def add(self, other: GF256Poly) -> GF256Poly:
        """Add (XOR) two polynomials, returning a new one."""
        a = self._coeffs
        b = other._coeffs
        if len(a) < len(b):
            a, b = b, a
        result = a[:]
        offset = len(a) - len(b)
        for i in range(len(b)):
            result[offset + i] ^= b[i]
        return GF256Poly(result)

    def multiply(self, other: GF256Poly) -> GF256Poly:
        """Multiply two polynomials."""
        if self.is_zero() or other.is_zero():
            return GF256Poly.zero()
        a = self._coeffs
        b = other._coeffs
        result = [0] * (len(a) + len(b) - 1)
        for i, ci in enumerate(a):
            if ci == 0:
                continue
            for j, cj in enumerate(b):
                result[i + j] ^= GF256.multiply(ci, cj)
        return GF256Poly(result)

    def multiply_by_scalar(self, scalar: int) -> GF256Poly:
        """Multiply each coefficient by a scalar."""
        if scalar == 0:
            return GF256Poly.zero()
        return GF256Poly([GF256.multiply(c, scalar) for c in self._coeffs])

    # -- internal --

    def _normalize(self) -> None:
        """Strip leading zero coefficients in-place."""
        if not self._coeffs:
            self._coeffs = [0]
            return
        first = 0
        while first < len(self._coeffs) and self._coeffs[first] == 0:
            first += 1
        if first == len(self._coeffs):
            self._coeffs = [0]
        elif first > 0:
            self._coeffs = self._coeffs[first:]

    def __repr__(self) -> str:
        return f"GF256Poly({self._coeffs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GF256Poly):
            return NotImplemented
        return self._coeffs == other._coeffs


# ---------------------------------------------------------------------------
# Helpers: generator polynomial, syndrome, polynomial division
# ---------------------------------------------------------------------------

# QR codes use generator base 0:
#   g(x) = (x - α^0)(x - α^1)...(x - α^(t-1))
GENERATOR_BASE = 0


def _build_generator(degree: int) -> GF256Poly:
    """Build the RS generator polynomial of given degree for QR codes."""
    g = GF256Poly([1])
    for i in range(degree):
        term = GF256Poly([1, GF256.exp(GENERATOR_BASE + i)])
        g = g.multiply(term)
    return g


def _compute_syndromes(received: list[int], num_ec: int) -> list[int] | None:
    """Compute syndrome values.  Returns None if all zero (no errors)."""
    poly = GF256Poly(received)
    syndromes = [0] * num_ec
    all_zero = True
    for i in range(num_ec):
        s = poly.evaluate_at(GF256.exp(GENERATOR_BASE + i))
        syndromes[i] = s
        if s != 0:
            all_zero = False
    return None if all_zero else syndromes


def _poly_divide(
    dividend: GF256Poly, divisor: GF256Poly
) -> tuple[GF256Poly, GF256Poly]:
    """Divide dividend by divisor, returning (quotient, remainder)."""
    if divisor.is_zero():
        raise ValueError("Division by zero polynomial")

    if dividend.degree() < divisor.degree():
        return GF256Poly.zero(), dividend

    d = list(dividend._coeffs)
    dvr = divisor._coeffs
    dvr_deg = divisor.degree()
    q_len = len(d) - dvr_deg
    quotient = [0] * q_len

    normalizer = GF256.inverse(dvr[0])

    for i in range(q_len):
        ci = d[i]
        if ci == 0:
            continue
        ci = GF256.multiply(ci, normalizer)
        quotient[i] = ci
        for j in range(1, len(dvr)):
            d[i + j] ^= GF256.multiply(dvr[j], ci)

    # Extract remainder
    rem = d[q_len:]
    first = 0
    while first < len(rem) and rem[first] == 0:
        first += 1
    rem = [0] if first == len(rem) else rem[first:]

    return GF256Poly(quotient), GF256Poly(rem)


# ---------------------------------------------------------------------------
# Extended Euclidean algorithm for σ(x) and ω(x)
# ---------------------------------------------------------------------------


def _run_euclidean(
    syndromes: list[int], num_ec: int
) -> tuple[GF256Poly, GF256Poly] | None:
    """Run the extended Euclidean algorithm.

    Finds σ(x) (error locator) and ω(x) (error evaluator) such that
    σ(x) · S(x) ≡ ω(x)  (mod x^(2t))  with  deg(ω) < deg(σ) ≤ t.

    Port of zxing-cpp ``RunEuclideanAlgorithm()``.

    Returns (sigma, omega) or None if uncorrectable.
    """
    # Syndrome polynomial S(x) MSB-first:
    # syndromes[0] = S_0, so reversed gives S_{R-1} x^{R-1} + ... + S_1 x + S_0
    r = GF256Poly(list(reversed(syndromes)))

    # rLast = x^R,  tLast = 0,  t = 1
    r_last = GF256Poly.monomial(1, num_ec)
    t_last = GF256Poly.zero()
    t = GF256Poly([1])

    # Ensure r has lower degree than rLast
    if r.degree() >= r_last.degree():
        r, r_last = r_last, r
        t, t_last = t_last, t

    half_ec = num_ec // 2

    while r.degree() >= half_ec:
        # Swap old → one step back
        t_last, t = t, t_last
        r_last, r = r, r_last

        if r_last.is_zero():
            return None

        # Divide r (dividend) by r_last (divisor)
        q, remainder = _poly_divide(r, r_last)
        r = remainder

        # Update t:  t_new = q · t_last + t
        t = q.multiply(t_last).add(t)

        if r.degree() >= r_last.degree():
            return None  # Division failed to reduce degree

    # Normalize so that σ(0) = 1
    sigma_tilde_at_zero = t.constant()
    if sigma_tilde_at_zero == 0:
        return None

    inv = GF256.inverse(sigma_tilde_at_zero)
    sigma = t.multiply_by_scalar(inv)
    omega = r.multiply_by_scalar(inv)

    return sigma, omega


# ---------------------------------------------------------------------------
# Forney's formula: error locations & magnitudes
# ---------------------------------------------------------------------------


def _find_error_locations(error_locator: GF256Poly) -> list[int]:
    """Brute-force search for roots of σ(x) over GF(256).

    For each root α^i, the error location is α^(-i) = inverse(α^i).
    """
    num_errors = error_locator.degree()
    if num_errors == 0:
        return []
    locations: list[int] = []
    for i in range(1, 256):  # skip α^0 = 1 (would mean error at log(1) = 0)
        if error_locator.evaluate_at(GF256.exp(i)) == 0:
            locations.append(GF256.inverse(GF256.exp(i)))
            if len(locations) == num_errors:
                break
    return locations


def _find_error_magnitudes(
    error_evaluator: GF256Poly, error_locations: list[int]
) -> list[int]:
    """Apply Forney's formula to find error magnitudes.

    For QR codes (generator_base = 0), we do NOT multiply by xiInverse.
    """
    s = len(error_locations)
    magnitudes = [0] * s
    for i in range(s):
        xi_inv = GF256.inverse(error_locations[i])
        denom = 1
        for j in range(s):
            if i != j:
                term = 1 ^ GF256.multiply(error_locations[j], xi_inv)
                denom = GF256.multiply(denom, term)
        magnitude = GF256.multiply(
            error_evaluator.evaluate_at(xi_inv), GF256.inverse(denom)
        )
        if GENERATOR_BASE != 0:
            magnitude = GF256.multiply(magnitude, xi_inv)
        magnitudes[i] = magnitude
    return magnitudes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rs_decode(received: list[int], num_ec: int) -> list[int] | None:
    """Decode a Reed-Solomon encoded block.

    Args:
        received: List of byte values (0–255). Data bytes followed by EC bytes.
        num_ec: Number of error-correction codewords (= 2t, where t errors
                can be corrected).

    Returns:
        Corrected byte list, or ``None`` if the block is uncorrectable.
    """
    if num_ec == 0:
        return list(received)

    # 1. Syndromes
    syndromes = _compute_syndromes(received, num_ec)
    if syndromes is None:
        return list(received)  # all zero → no errors

    # 2. Euclidean → σ(x), ω(x)
    euclid = _run_euclidean(syndromes, num_ec)
    if euclid is None:
        return None
    sigma, omega = euclid

    # 3. Error locations
    error_locations = _find_error_locations(sigma)
    if len(error_locations) != sigma.degree():
        return None  # degree mismatch → too many errors

    # 4. Error magnitudes (Forney)
    error_magnitudes = _find_error_magnitudes(omega, error_locations)

    # 5. Correct
    corrected = list(received)
    msg_len = len(corrected)
    for loc, mag in zip(error_locations, error_magnitudes):
        position = msg_len - 1 - GF256.log(loc)
        if position < 0 or position >= msg_len:
            return None
        corrected[position] ^= mag

    # 6. Re-validate
    verify_poly = GF256Poly(corrected)
    for i in range(num_ec):
        if verify_poly.evaluate_at(GF256.exp(GENERATOR_BASE + i)) != 0:
            return None

    return corrected
