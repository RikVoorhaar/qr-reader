"""Tests for GF(256) arithmetic module."""

import pytest

from qr_reader.decoder.gf import GF256


class TestGF256Tables:
    """Verify the log/exp tables are consistent."""

    def test_log_exp_consistency(self):
        """log(exp[i]) == i for all i in [0, 255)."""
        for i in range(256):
            e = GF256.exp(i)
            assert GF256.log(e) == (i % 255 if i > 0 else 0)

    def test_one(self):
        """exp(0) = 1."""
        assert GF256.exp(0) == 1

    def test_alpha_to_255_is_one(self):
        """exp(255) = α^255 = 1."""
        assert GF256.exp(255) == 1

    def test_log_of_one(self):
        """log(1) = 0."""
        assert GF256.log(1) == 0


class TestGF256Multiply:
    """Test multiplication."""

    def test_multiply_by_zero(self):
        for i in range(256):
            assert GF256.multiply(i, 0) == 0
            assert GF256.multiply(0, i) == 0

    def test_multiply_by_one(self):
        for i in range(256):
            assert GF256.multiply(i, 1) == i
            assert GF256.multiply(1, i) == i

    def test_exp_rule(self):
        """multiply(α^i, α^j) == α^(i+j) for i,j >= 0."""
        for i in range(0, 256, 13):
            for j in range(0, 256, 13):
                a = GF256.exp(i)
                b = GF256.exp(j)
                expected = (
                    GF256.exp((i + j) % 255)
                    if (i % 255) + (j % 255) > 0
                    else GF256.exp(0)
                )
                assert GF256.multiply(a, b) == expected, f"i={i}, j={j}"

    def test_commutative(self):
        import random

        rng = random.Random(42)
        for _ in range(500):
            a = rng.randint(0, 255)
            b = rng.randint(0, 255)
            assert GF256.multiply(a, b) == GF256.multiply(b, a)

    def test_associative(self):
        import random

        rng = random.Random(42)
        for _ in range(500):
            a = rng.randint(0, 255)
            b = rng.randint(0, 255)
            c = rng.randint(0, 255)
            assert GF256.multiply(GF256.multiply(a, b), c) == GF256.multiply(
                a, GF256.multiply(b, c)
            )


class TestGF256Add:
    """Test addition (XOR)."""

    def test_add_is_xor(self):
        assert GF256.add(0x5A, 0x3C) == (0x5A ^ 0x3C)
        assert GF256.add(0xFF, 0x01) == 0xFE
        assert GF256.add(0x00, 0x00) == 0x00

    def test_subtract_equals_add(self):
        for i in range(256):
            assert GF256.subtract(i, 42) == GF256.add(i, 42)


class TestGF256Inverse:
    """Test multiplicative inverse."""

    def test_inverse_multiply_is_one(self):
        for a in range(1, 256):
            inv = GF256.inverse(a)
            assert GF256.multiply(a, inv) == 1, f"a={a}, inv={inv}"

    def test_inverse_of_one(self):
        assert GF256.inverse(1) == 1

    def test_inverse_of_zero_raises(self):
        with pytest.raises(ValueError):
            GF256.inverse(0)


class TestGF256Pow:
    """Test exponentiation."""

    def test_pow_zero(self):
        for a in range(256):
            assert GF256.pow(a, 0) == 1

    def test_pow_one(self):
        for a in range(256):
            assert GF256.pow(a, 1) == a

    def test_zero_pow(self):
        assert GF256.pow(0, 5) == 0
        assert GF256.pow(0, 0) == 1

    def test_pow_matches_multiply(self):
        """a^3 should equal a * a * a."""
        import random

        rng = random.Random(42)
        for _ in range(100):
            a = rng.randint(1, 255)
            p3 = GF256.pow(a, 3)
            m3 = GF256.multiply(GF256.multiply(a, a), a)
            assert p3 == m3, f"a={a}"


class TestGF256KnownProducts:
    """Test known products from spec examples."""

    def test_known_products(self):
        # From the QR code spec: α^8 = α^4 + α^3 + α^2 + 1 = 0x1D = 29
        assert GF256.exp(8) == 29
        # Verify: α^8 * α^8 = α^16
        assert GF256.multiply(GF256.exp(8), GF256.exp(8)) == GF256.exp(16)


class TestGF256LogZero:
    """log(0) should raise."""

    def test_log_zero_raises(self):
        with pytest.raises(ValueError):
            GF256.log(0)
