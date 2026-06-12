"""Tests for version information BCH decoding (Phase 3).

Uses the `qrcode` library to generate QR codes at various versions
and verifies that version information bits are correctly read and decoded.
"""

import numpy as np
import pytest

from qr_reader.decoder.tables import VERSION_INFO_PATTERNS
from qr_reader.decoder.version_info import (
    decode_version,
    decode_version_info,
    read_version_bits,
)


def _make_qr_matrix(version: int, ecc_name: str = "L", mask: int = 0) -> np.ndarray:
    """Generate a QR code bit matrix with full control over ECL and mask.

    Uses qrcode library; forces a specific mask pattern by calling
    makeImpl() directly.
    """
    import qrcode

    ecc_consts = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.QRCode(
        version=version,
        error_correction=ecc_consts[ecc_name.upper()],
        box_size=1,
        border=0,
    )
    qr.add_data("Hello QR")
    # Force the specific mask pattern by calling makeImpl directly
    qr.makeImpl(False, mask)
    return np.array(qr.modules, dtype=bool)


class TestReadVersionBits:
    """Test reading version bits from the two locations in the bit matrix."""

    @pytest.mark.parametrize("version", [7, 10, 15, 20, 25, 30, 35, 40])
    @pytest.mark.parametrize("mask", [0, 3, 7])
    def test_exact_bits_match_table(self, version, mask):
        """Both copies should match the known 18-bit pattern from the tables."""
        matrix = _make_qr_matrix(version, "L", mask)
        bits_a, bits_b = read_version_bits(matrix)

        expected = VERSION_INFO_PATTERNS[version - 7]
        assert bits_a == expected, (
            f"v{version} mask{mask}: copy A 0x{bits_a:05X} != expected 0x{expected:05X}"
        )
        assert bits_b == expected, (
            f"v{version} mask{mask}: copy B 0x{bits_b:05X} != expected 0x{expected:05X}"
        )

    @pytest.mark.parametrize("version", [7, 20, 40])
    def test_both_copies_identical(self, version):
        """Both copies should read the same bits for a clean QR code."""
        matrix = _make_qr_matrix(version, "L", 0)
        bits_a, bits_b = read_version_bits(matrix)
        assert bits_a == bits_b

    def test_v7_dimension(self):
        """v7 should produce a 45×45 matrix."""
        matrix = _make_qr_matrix(7, "L", 0)
        assert matrix.shape == (45, 45)


class TestDecodeVersion:
    """Test BCH decoding logic."""

    @pytest.mark.parametrize(
        "version",
        [7, 8, 9, 10, 15, 20, 25, 30, 35, 40],
    )
    def test_decode_exact_match(self, version):
        """Decode an exact (no-error) version info pattern."""
        pattern = VERSION_INFO_PATTERNS[version - 7]
        result = decode_version(pattern, pattern)
        assert result == version

    @pytest.mark.parametrize("version", [7, 20, 40])
    @pytest.mark.parametrize("errors", [1, 2, 3])
    def test_decode_with_errors(self, version, errors):
        """Should tolerate up to 3 bit errors (BCH distance ≥ 8)."""
        import random

        rng = random.Random(42 + version * 100 + errors)
        pattern = VERSION_INFO_PATTERNS[version - 7]

        # Pick `errors` distinct bit positions to flip
        positions = rng.sample(range(18), errors)
        corrupted = pattern
        for pos in positions:
            corrupted ^= 1 << pos

        # Corrupt both copies the same way (worst case) — should still work
        result = decode_version(corrupted, corrupted)
        assert result == version, (
            f"v{version} with {errors} errors: got {result}, expected {version}"
        )

    @pytest.mark.parametrize("version", [7, 20, 40])
    def test_decode_rejects_4_errors(self, version):
        """Should NOT decode with 4 bit errors (exceeds correction capability)."""
        import random

        rng = random.Random(123 + version)
        pattern = VERSION_INFO_PATTERNS[version - 7]
        positions = rng.sample(range(18), 4)
        corrupted = pattern
        for pos in positions:
            corrupted ^= 1 << pos

        result = decode_version(corrupted, corrupted)
        # With 4 errors, distance is 4, should fail (threshold is ≤ 3)
        assert result is None, f"v{version}: should reject 4 errors"

    def test_decode_one_copy_corrupted(self):
        """If only one copy is corrupted, the clean copy should save us."""
        pattern = VERSION_INFO_PATTERNS[20 - 7]  # v20

        # Corrupt copy A with 5 errors
        corrupted = pattern
        for pos in [0, 3, 7, 12, 16]:
            corrupted ^= 1 << pos

        result = decode_version(corrupted, pattern)  # clean copy B
        assert result == 20

    def test_decode_rejects_garbage(self):
        """Verify that all-zeroes and high-distance values are rejected."""
        result = decode_version(0, 0)
        assert result is None

        # Flip 4 bits of a known pattern → distance 4 (rejected)
        v7_pattern = VERSION_INFO_PATTERNS[0]
        corrupted = v7_pattern ^ 0x0F  # flip 4 low bits
        result = decode_version(corrupted, corrupted)
        assert result is None, f"4-bit corruption 0x{corrupted:05X} decoded to {result}"


class TestDecodeVersionInfo:
    """Test the top-level decode_version_info convenience function."""

    @pytest.mark.parametrize("version", [7, 15, 25, 40])
    def test_decode_from_matrix(self, version):
        """Full pipeline: read bits from matrix → decode → version."""
        matrix = _make_qr_matrix(version, "M", 0)
        result = decode_version_info(matrix)
        assert result == version

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
    def test_no_false_positives_on_v1_v6(self, version):
        """v1–v6 have no version info region; should return None."""
        matrix = _make_qr_matrix(version, "L", 0)
        result = decode_version_info(matrix)
        assert result is None, (
            f"v{version} should not decode version info (no version info region)"
        )


class TestVersionPatternConsistency:
    """Verify the consistency of version info patterns stored in tables."""

    def test_version_pattern_ecc_correctness(self):
        """Verify each pattern is a valid BCH(18,6) codeword with generator 0x1F25."""
        # The BCH(18,6) generator polynomial is 0x1F25 (binary: 1 1111 0010 0101)
        # For each pattern, the low 12 bits should be the BCH remainder of the
        # high 6 bits (version number).
        for v in range(7, 41):
            pattern = VERSION_INFO_PATTERNS[v - 7]
            version_field = pattern >> 12  # high 6 bits
            ecc_field = pattern & 0xFFF  # low 12 bits

            # Compute BCH remainder
            rem = v
            for _ in range(12):
                rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)

            assert version_field == v, f"v{v}: version field mismatch"
            assert (rem & 0xFFF) == ecc_field, (
                f"v{v}: ECC field mismatch: got 0x{ecc_field:03X}, "
                f"computed 0x{(rem & 0xFFF):03X}"
            )
