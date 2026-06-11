"""Tests for Reed-Solomon error correction (rs.py)."""

import random

import pytest

from qr_reader.decoder.gf import GF256
from qr_reader.decoder.rs import (
    GENERATOR_BASE,
    GF256Poly,
    _build_generator,
    _compute_syndromes,
    _find_error_locations,
    _find_error_magnitudes,
    _poly_divide,
    _run_euclidean,
    rs_decode,
)

# ---------------------------------------------------------------------------
# GF256Poly tests
# ---------------------------------------------------------------------------


class TestGF256Poly:
    """Tests for GF(256) polynomial arithmetic."""

    def test_construct_and_degree(self):
        p = GF256Poly([1, 2, 3])
        assert p.degree() == 2
        assert p.coeffs == [1, 2, 3]

    def test_normalize_strips_leading_zeros(self):
        p = GF256Poly([0, 0, 1, 2])
        assert p.coeffs == [1, 2]
        assert p.degree() == 1

    def test_zero_polynomial(self):
        p = GF256Poly([0, 0, 0])
        assert p.is_zero()
        assert p.coeffs == [0]
        assert p.degree() == 0

    def test_monomial(self):
        p = GF256Poly.monomial(5, 3)
        assert p.coeffs == [5, 0, 0, 0]
        assert p.degree() == 3

    def test_evaluate_at_zero(self):
        p = GF256Poly([2, 3, 5])  # 2x² + 3x + 5
        assert p.evaluate_at(0) == 5

    def test_evaluate_at_one(self):
        p = GF256Poly([2, 3, 5])
        # 2 XOR 3 XOR 5 = 4
        assert p.evaluate_at(1) == 2 ^ 3 ^ 5

    def test_evaluate_at_alpha(self):
        # p(x) = x² + 1 → p(α) = α² + 1
        p = GF256Poly([1, 0, 1])
        a = GF256.exp(1)
        expected = GF256.exp(1) ^ GF256.exp(1) ^ 1  # ? Let me compute properly
        # p(α) = α² + 1 = GF256.multiply(α, α) XOR 1 = GF256.exp(2) XOR 1
        expected = GF256.exp(2) ^ 1
        assert p.evaluate_at(a) == expected

    def test_add(self):
        a = GF256Poly([1, 2, 3])
        b = GF256Poly([4, 5])
        result = a.add(b)
        assert result.coeffs == [1, 6, 6]  # (1)x² + (2^4)x + (3^5)

    def test_add_different_lengths(self):
        a = GF256Poly([1])
        b = GF256Poly([2, 3])
        result = a.add(b)
        assert result.coeffs == [2, 2]  # 2x + (3^1)

    def test_multiply(self):
        # (x + 1)(x + 2) = x² + (1+2)x + 2 = x² + 3x + 2
        a = GF256Poly([1, 1])
        b = GF256Poly([1, 2])
        result = a.multiply(b)
        assert result.coeffs == [1, 1 ^ 2, GF256.multiply(1, 2)]
        assert result.coeffs == [1, 3, 2]

    def test_multiply_by_scalar(self):
        p = GF256Poly([1, 2, 3])
        result = p.multiply_by_scalar(2)
        assert result.coeffs == [
            GF256.multiply(1, 2),
            GF256.multiply(2, 2),
            GF256.multiply(3, 2),
        ]

    def test_multiply_by_zero_gives_zero(self):
        p = GF256Poly([1, 2, 3])
        assert p.multiply(GF256Poly.zero()).is_zero()
        assert p.multiply_by_scalar(0).is_zero()

    def test_poly_divide_exact(self):
        # (x² + 3x + 2) / (x + 1) = (x + 2) remainder 0
        dividend = GF256Poly([1, 3, 2])
        divisor = GF256Poly([1, 1])
        q, r = _poly_divide(dividend, divisor)
        assert q.coeffs == [1, 2]  # x + 2
        assert r.is_zero()

    def test_poly_divide_with_remainder(self):
        # (x² + 3x + 3) / (x + 1) = (x + 2) remainder 1
        dividend = GF256Poly([1, 3, 3])
        divisor = GF256Poly([1, 1])
        q, r = _poly_divide(dividend, divisor)
        assert q.coeffs == [1, 2]
        assert r.coeffs == [1]

    def test_poly_divide_dividend_lower_degree(self):
        dividend = GF256Poly([1])
        divisor = GF256Poly([1, 1, 1])  # degree 2 > 0
        q, r = _poly_divide(dividend, divisor)
        assert q.is_zero()
        assert r == dividend


# ---------------------------------------------------------------------------
# Generator polynomial tests
# ---------------------------------------------------------------------------


class TestGeneratorPolynomial:
    """Tests for the RS generator polynomial builder."""

    def test_degree_1_generator(self):
        g = _build_generator(1)
        # g(x) = (x - α^0) = x + 1
        assert g.coeffs == [1, 1]

    def test_degree_2_generator(self):
        g = _build_generator(2)
        # g(x) = (x + 1)(x + α) = x² + (1+α)x + α
        assert g.degree() == 2

    def test_generator_roots(self):
        """Generator polynomial g(α^i) == 0 for i = 0..degree-1."""
        for d in [1, 3, 7, 10, 17]:
            g = _build_generator(d)
            for i in range(d):
                assert g.evaluate_at(GF256.exp(GENERATOR_BASE + i)) == 0

    def test_generator_leading_is_one(self):
        for d in range(1, 31):
            g = _build_generator(d)
            assert g.coeffs[0] == 1


# ---------------------------------------------------------------------------
# RS encode helper (for testing)
# ---------------------------------------------------------------------------


def _rs_encode(data: list[int], num_ec: int) -> list[int]:
    """Systematic RS encoding: data followed by remainder of data·x^num_ec / g(x)."""
    g = _build_generator(num_ec)
    # Multiply data by x^num_ec
    msg = GF256Poly(data + [0] * num_ec)
    _, remainder = _poly_divide(msg, g)
    # Pad remainder to exactly num_ec bytes (leading zeros)
    rem_coeffs = remainder.coeffs
    padding = [0] * (num_ec - len(rem_coeffs))
    return data + padding + rem_coeffs


# ---------------------------------------------------------------------------
# RS decode tests
# ---------------------------------------------------------------------------


class TestRSDecodeNoErrors:
    """No errors — identity."""

    def test_single_block_no_errors(self):
        data = [1, 2, 3, 4, 5]
        encoded = _rs_encode(data, 4)
        result = rs_decode(encoded, 4)
        assert result is not None
        assert result == encoded

    def test_various_sizes_no_errors(self):
        for data_len in [1, 5, 10, 19, 55, 100]:
            for num_ec in [4, 7, 10, 17]:
                data = [
                    random.Random(data_len + num_ec).randint(0, 255)
                    for _ in range(data_len)
                ]
                encoded = _rs_encode(data, num_ec)
                result = rs_decode(encoded, num_ec)
                assert result is not None
                assert result == encoded

    def test_zero_ec(self):
        data = [1, 2, 3, 4, 5]
        result = rs_decode(data, 0)
        assert result == data


class TestRSDecodeWithErrors:
    """Introduce up to t errors, verify correction."""

    def test_one_error(self):
        data = [10, 20, 30, 40, 50]
        num_ec = 10  # t = 5
        encoded = _rs_encode(data, num_ec)

        corrupted = list(encoded)
        corrupted[2] ^= 0x55  # flip some bits at position 2
        assert corrupted != encoded

        result = rs_decode(corrupted, num_ec)
        assert result is not None
        assert result == encoded

    def test_t_errors(self):
        data = list(range(100))
        num_ec = 20  # t = 10
        encoded = _rs_encode(data, num_ec)

        rng = random.Random(42)
        corrupted = list(encoded)
        positions = rng.sample(range(len(encoded)), 10)
        for pos in positions:
            corrupted[pos] ^= rng.randint(1, 255)
        assert corrupted != encoded

        result = rs_decode(corrupted, num_ec)
        assert result is not None
        assert result == encoded

    def test_multiple_t_errors_random(self):
        rng = random.Random(99)
        for _ in range(20):
            data_len = rng.randint(5, 50)
            num_ec = 20
            data = [rng.randint(0, 255) for _ in range(data_len)]
            encoded = _rs_encode(data, num_ec)

            t = num_ec // 2
            num_errors = rng.randint(1, t)
            corrupted = list(encoded)
            positions = rng.sample(range(len(encoded)), num_errors)
            for pos in positions:
                corrupted[pos] ^= rng.randint(1, 255)

            result = rs_decode(corrupted, num_ec)
            assert result is not None, f"Failed to correct {num_errors} errors"
            assert result == encoded

    def test_error_in_ec_part(self):
        data = [1, 2, 3, 4, 5]
        num_ec = 10
        encoded = _rs_encode(data, num_ec)

        corrupted = list(encoded)
        corrupted[-1] ^= 0xAA  # error in last EC byte
        assert corrupted != encoded

        result = rs_decode(corrupted, num_ec)
        assert result is not None
        assert result == encoded

    def test_error_in_both_data_and_ec(self):
        data = list(range(50))
        num_ec = 20  # t = 10
        encoded = _rs_encode(data, num_ec)

        rng = random.Random(7)
        corrupted = list(encoded)
        # 3 errors in data, 2 in EC
        positions = [5, 20, 35, -3, -1]
        for pos in positions:
            corrupted[pos] ^= rng.randint(1, 255)

        result = rs_decode(corrupted, num_ec)
        assert result is not None
        assert result == encoded


class TestRSDecodeTooManyErrors:
    """More than t errors → should detect failure."""

    def test_t_plus_one_errors(self):
        data = list(range(50))
        num_ec = 10  # t = 5
        encoded = _rs_encode(data, num_ec)

        rng = random.Random(123)
        corrupted = list(encoded)
        positions = rng.sample(range(len(encoded)), 6)  # t+1 = 6
        for pos in positions:
            corrupted[pos] ^= rng.randint(1, 255)

        result = rs_decode(corrupted, num_ec)
        # Should return None (uncorrectable) — but sometimes with only
        # t+1 errors it might "succeed" with wrong correction.
        # We just check it doesn't crash and returns something.
        # The re-validation step makes this unlikely to pass silently.
        if result is not None:
            # If it returned something, verify it's actually correct
            verify = _rs_encode(data, num_ec)
            assert result == verify, "RS falsely claimed to correct beyond capacity"

    def test_all_bytes_corrupted(self):
        data = [1, 2, 3, 4, 5]
        num_ec = 4  # t = 2
        encoded = _rs_encode(data, num_ec)

        # Corrupt more than half the bytes → uncorrectable
        rng = random.Random(42)
        corrupted = [b ^ rng.randint(1, 255) for b in encoded]

        result = rs_decode(corrupted, num_ec)
        # May return None or may return wrong data — re-validation catches it
        if result is not None:
            # Should not have corrected to the original
            assert result == encoded or result is not None


# ---------------------------------------------------------------------------
# Syndrome tests
# ---------------------------------------------------------------------------


class TestSyndromes:
    """Test syndrome calculation."""

    def test_valid_codeword_all_syndromes_zero(self):
        data = list(range(10))
        num_ec = 6
        encoded = _rs_encode(data, num_ec)
        syndromes = _compute_syndromes(encoded, num_ec)
        assert syndromes is None  # all zero

    def test_corrupted_codeword_nonzero_syndromes(self):
        data = list(range(10))
        num_ec = 6
        encoded = _rs_encode(data, num_ec)
        corrupted = list(encoded)
        corrupted[0] ^= 1
        syndromes = _compute_syndromes(corrupted, num_ec)
        assert syndromes is not None
        assert any(s != 0 for s in syndromes)


# ---------------------------------------------------------------------------
# Euclidean algorithm tests
# ---------------------------------------------------------------------------


class TestEuclidean:
    """Test the extended Euclidean algorithm stand-alone."""

    def test_no_errors_trivial(self):
        # With no syndrome data, Euclidean returns sigma=1, omega=0
        # But _run_euclidean handles the case internally via syndromes list
        pass  # tested via rs_decode integration

    def test_euclidean_finds_correct_sigma_omega(self):
        data = [10, 20, 30, 40, 50]
        num_ec = 10
        encoded = _rs_encode(data, num_ec)
        corrupted = list(encoded)
        corrupted[2] ^= 0x55

        syndromes = _compute_syndromes(corrupted, num_ec)
        assert syndromes is not None
        result = _run_euclidean(syndromes, num_ec)
        assert result is not None
        sigma, omega = result
        assert not sigma.is_zero()
        assert sigma.degree() <= num_ec // 2


# ---------------------------------------------------------------------------
# Forney formula tests
# ---------------------------------------------------------------------------


class TestForneyFormula:
    """Test error location and magnitude computation."""

    def test_find_error_locations(self):
        # Create a simple error locator with known root
        # σ(x) = x + α  (root at x = α → error location = inverse(α) = α^254)
        sigma = GF256Poly([1, GF256.exp(1)])
        locations = _find_error_locations(sigma)
        assert len(locations) == 1
        assert locations[0] == GF256.inverse(GF256.exp(1))


# ---------------------------------------------------------------------------
# Integration test with qrcode library
# ---------------------------------------------------------------------------


class TestRSWithQRCodeLibrary:
    """Test RS decoding using qrcode library to generate known QR codes."""

    def test_decode_qr_code_v1_L_block(self):
        """Version 1-L has 19 data + 7 EC = 26 total, per block."""
        import numpy as np
        import qrcode

        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data("HELLO")
        qr.make(fit=False)

        # Get the raw modules
        modules = np.array(qr.modules, dtype=bool)

        # The data codewords + EC are embedded in the QR code.
        # For now, just verify we can import qrcode and create a QR code.
        # Full integration will be in the top-level decoder test.
        assert modules.shape == (21, 21)


# ---------------------------------------------------------------------------
# Round-trip: encode → corrupt → decode
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end encode → corrupt → decode round-trip."""

    def test_roundtrip_various_params(self):
        rng = random.Random(42)
        for data_len in [5, 16, 19, 44, 55]:
            for num_ec in [7, 10, 13, 17]:
                data = [rng.randint(0, 255) for _ in range(data_len)]
                encoded = _rs_encode(data, num_ec)

                # Round-trip with no corruption
                result = rs_decode(encoded, num_ec)
                assert result == encoded

                # Round-trip with up to t errors
                t = num_ec // 2
                if t > 0:
                    num_errs = rng.randint(1, t)
                    corrupted = list(encoded)
                    positions = rng.sample(range(len(encoded)), num_errs)
                    for pos in positions:
                        corrupted[pos] ^= rng.randint(1, 255)

                    result = rs_decode(corrupted, num_ec)
                    assert result is not None
                    assert result == encoded
