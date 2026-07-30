"""Tests for format information BCH decoding (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.decoder.format_info import (
    FORMAT_INFO_MASK,
    MAX_HAMMING_DISTANCE,
    FormatInfoDecodeError,
    _hamming_distance,
    _read_format_info_bits,
    decode_format_info,
)
from qr_reader.decoder.tables import (
    ECL_NAMES,
    VALID_FORMAT_PATTERNS,
)
from qr_reader.tests.decoder.helpers import make_qr_bitmatrix


class TestHammingDistance:
    """Test the _hamming_distance helper."""

    def test_same(self):
        for p in VALID_FORMAT_PATTERNS[:4]:
            assert _hamming_distance(p, p) == 0

    def test_one_bit_difference(self):
        # 0x77C4 ^ 0x77C5 = 1
        assert _hamming_distance(0x77C4, 0x77C5) == 1

    def test_all_bits_differ(self):
        # 0x0000 ^ 0x7FFF = all 15 bits
        assert _hamming_distance(0x0000, 0x7FFF) == 15


class TestBitReading:
    """Test the _read_format_info_bits function."""

    def test_manual_known_v1_l_mask0(self):
        """Manually set the format bits for V1-L mask0 (0x77C4) and verify reading."""
        size = 21  # V1
        matrix = np.zeros((size, size), dtype=bool)

        # Expected pattern: 0x77C4 = 0b0111_0111_1100_0100
        pattern = 0x77C4

        # Place location 1 bits manually
        loc1_order = [
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (7, 8),  # skip (6,8) timing
            (8, 8),
            (8, 7),
            (8, 5),  # skip (8,6) timing
            (8, 4),
            (8, 3),
            (8, 2),
            (8, 1),
            (8, 0),
        ]
        for bit_pos, (x, y) in enumerate(loc1_order):
            if pattern & (1 << bit_pos):
                matrix[y, x] = True

        # Place location 2 bits (LSB-first, same order as reader)
        loc2_order = [(size - 1 - i, 8) for i in range(8)] + [(8, size - 1 - i) for i in range(7)]
        for bit_pos, (x, y) in enumerate(loc2_order):
            if pattern & (1 << bit_pos):
                matrix[y, x] = True

        assert _read_format_info_bits(matrix, size, 1) == pattern
        assert _read_format_info_bits(matrix, size, 2) == pattern

    def test_manual_known_v1_h_mask7(self):
        """V1-H mask7 = 0x083B."""
        size = 21
        matrix = np.zeros((size, size), dtype=bool)
        pattern = 0x083B

        loc1_order = [
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (7, 8),
            (8, 8),
            (8, 7),
            (8, 5),
            (8, 4),
            (8, 3),
            (8, 2),
            (8, 1),
            (8, 0),
        ]
        for bit_pos, (x, y) in enumerate(loc1_order):
            if pattern & (1 << bit_pos):
                matrix[y, x] = True

        loc2_order = [(size - 1 - i, 8) for i in range(8)] + [(8, size - 1 - i) for i in range(7)]
        for bit_pos, (x, y) in enumerate(loc2_order):
            if pattern & (1 << bit_pos):
                matrix[y, x] = True

        assert _read_format_info_bits(matrix, size, 1) == pattern
        assert _read_format_info_bits(matrix, size, 2) == pattern

    def test_all_patterns_roundtrip_v1(self):
        """Every valid pattern can be written and read back on a V1 matrix."""
        size = 21
        for pattern in VALID_FORMAT_PATTERNS:
            matrix = np.zeros((size, size), dtype=bool)

            loc1_order = [
                (0, 8),
                (1, 8),
                (2, 8),
                (3, 8),
                (4, 8),
                (5, 8),
                (7, 8),
                (8, 8),
                (8, 7),
                (8, 5),
                (8, 4),
                (8, 3),
                (8, 2),
                (8, 1),
                (8, 0),
            ]
            for bit_pos, (x, y) in enumerate(loc1_order):
                if pattern & (1 << bit_pos):
                    matrix[y, x] = True

            loc2_order = [(size - 1 - i, 8) for i in range(8)] + [(8, size - 1 - i) for i in range(7)]
            for bit_pos, (x, y) in enumerate(loc2_order):
                if pattern & (1 << bit_pos):
                    matrix[y, x] = True

            assert _read_format_info_bits(matrix, size, 1) == pattern, (
                f"loc1 roundtrip failed for {pattern:#05x}"
            )
            assert _read_format_info_bits(matrix, size, 2) == pattern, (
                f"loc2 roundtrip failed for {pattern:#05x}"
            )

    def test_location_2_different_versions(self):
        """Verify location 2 bit reading adapts to larger symbol sizes."""
        for version in [2, 5, 10]:
            size = 17 + 4 * version
            pattern = 0x5412  # M, mask 0
            matrix = np.zeros((size, size), dtype=bool)

            loc2_order = [(size - 1 - i, 8) for i in range(8)] + [(8, size - 1 - i) for i in range(7)]
            for bit_pos, (x, y) in enumerate(loc2_order):
                if pattern & (1 << bit_pos):
                    matrix[y, x] = True

            assert _read_format_info_bits(matrix, size, 2) == pattern, (
                f"V{version} loc2 roundtrip failed"
            )


class TestDecodeFormatInfo:
    """Test the decode_format_info public API."""

    def test_v1_l_basic(self):
        """V1-L QR: ECL L, some mask."""
        matrix = make_qr_bitmatrix("HELLO", version=1, ecl="L")
        ecl_idx, mask_idx = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "L"
        assert 0 <= mask_idx <= 7

    def test_v1_m_basic(self):
        matrix = make_qr_bitmatrix("HELLO", version=1, ecl="M")
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "M"

    def test_v1_q_basic(self):
        matrix = make_qr_bitmatrix("HELLO", version=1, ecl="Q")
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "Q"

    def test_v1_h_basic(self):
        matrix = make_qr_bitmatrix("HELLO", version=1, ecl="H")
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "H"

    @pytest.mark.parametrize(
        "version, ecl_name, data",
        [
            (1, "L", "QR TEST DATA"),
            (1, "M", "QR TEST DATA"),
            (1, "Q", "QR TEST DATA"),
            (1, "H", "HI"),  # V1-H has only 9 bytes capacity
            (3, "L", "QR TEST DATA"),
            (3, "M", "QR TEST DATA"),
            (3, "Q", "QR TEST DATA"),
            (3, "H", "QR TEST DATA"),
            (7, "L", "VERSION 7 TEST"),
            (7, "M", "VERSION 7 TEST"),
            (7, "Q", "VERSION 7 TEST"),
            (7, "H", "VERSION 7 TEST"),
            (10, "L", "QR TEST DATA V10"),
            (10, "M", "QR TEST DATA V10"),
            (10, "Q", "QR TEST DATA V10"),
            (10, "H", "QR TEST DATA V10"),
            (20, "L", "QR TEST DATA V20"),
            (20, "M", "QR TEST DATA V20"),
            (20, "Q", "QR TEST DATA V20"),
            (20, "H", "QR TEST DATA V20"),
        ],
    )
    def test_all_ecl_all_versions(self, version, ecl_name, data):
        """Exhaustive: ECL correctly decoded for various versions."""
        matrix = make_qr_bitmatrix(data, version=version, ecl=ecl_name)
        ecl_idx, mask_idx = decode_format_info(matrix, version)
        assert ECL_NAMES[ecl_idx] == ecl_name
        assert 0 <= mask_idx <= 7

    def test_mask_matches_qrcode_library(self):
        """When we force a specific mask, it decodes correctly."""
        for mask in range(8):
            matrix = make_qr_bitmatrix("TEST", version=2, ecl="L", mask=mask)
            ecl_idx, mask_idx = decode_format_info(matrix, 2)
            assert mask_idx == mask, f"Forced mask {mask} decoded as {mask_idx}"
            assert ECL_NAMES[ecl_idx] == "L"

    def test_single_bit_error_corrected(self):
        """Flip one format bit; should still decode correctly (Hamming distance 1 ≤ 3)."""
        matrix = make_qr_bitmatrix("TEST", version=1, ecl="M")
        matrix[8, 0] ^= True  # flip one format-info bit in location 1
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "M"

    def test_two_bit_errors_corrected(self):
        """Flip two format bits in same copy; should still decode."""
        matrix = make_qr_bitmatrix("TEST", version=1, ecl="M")
        matrix[8, 0] ^= True
        matrix[8, 1] ^= True
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "M"

    def test_three_bit_errors_corrected(self):
        """Flip three format bits in same copy; should still decode."""
        matrix = make_qr_bitmatrix("TEST", version=1, ecl="M")
        matrix[8, 0] ^= True
        matrix[8, 1] ^= True
        matrix[8, 2] ^= True
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "M"

    def test_four_bit_errors_rejected(self):
        """Enough bit flips in both copies to exceed 3-bit error budget."""
        from qr_reader.decoder.tables import VALID_FORMAT_PATTERNS

        for seed in range(100):
            matrix = make_qr_bitmatrix("TEST", version=1, ecl="M")
            rng = np.random.default_rng(seed)
            for x in [0, 1, 2, 3, 4, 5, 7, 8]:
                matrix[8, x] = rng.integers(0, 2, dtype=bool)
            for c in [0, 1, 2, 3, 4, 5, 7, 8]:
                matrix[c, 8] = rng.integers(0, 2, dtype=bool)
            size = 21
            for i in range(7):
                matrix[size - 1 - i, 8] = rng.integers(0, 2, dtype=bool)
            for i in range(7):
                matrix[8, size - 1 - i] = rng.integers(0, 2, dtype=bool)
            try:
                decode_format_info(matrix, 1)
            except FormatInfoDecodeError:
                return  # test passed
        pytest.fail("No seed produced an uncorrectable format info pattern")

    def test_invalid_format_info_rejected(self):
        """Scramble format bits so the pattern is far from any valid one."""
        for seed in range(100):
            matrix = make_qr_bitmatrix("TEST", version=1, ecl="M")
            rng = np.random.default_rng(seed)
            for x in [0, 1, 2, 3, 4, 5, 7, 8]:
                matrix[8, x] = rng.integers(0, 2, dtype=bool)
            for c in [0, 1, 2, 3, 4, 5, 7, 8]:
                matrix[c, 8] = rng.integers(0, 2, dtype=bool)
            size = 21
            for i in range(7):
                matrix[size - 1 - i, 8] = rng.integers(0, 2, dtype=bool)
            for i in range(7):
                matrix[8, size - 1 - i] = rng.integers(0, 2, dtype=bool)
            try:
                decode_format_info(matrix, 1)
            except FormatInfoDecodeError:
                return  # test passed
        pytest.fail("No seed produced an uncorrectable format info pattern")

    def test_two_copies_correct_each_other(self):
        """Copy 1 has 2 errors, copy 2 is clean → still decodes via copy 2."""
        matrix = make_qr_bitmatrix("TEST", version=1, ecl="L")
        # Corrupt copy 1 only (bits 0,1 of the format info)
        matrix[8, 0] ^= True
        matrix[8, 1] ^= True
        # Copy 2 is clean → should still decode correctly
        ecl_idx, _ = decode_format_info(matrix, 1)
        assert ECL_NAMES[ecl_idx] == "L"

    def test_v40_still_works(self):
        """Smoke test: V40 format info can be read (though big)."""
        # V40-H with a short message still fits
        matrix = make_qr_bitmatrix("HI", version=40, ecl="H")
        ecl_idx, mask_idx = decode_format_info(matrix, 40)
        assert ECL_NAMES[ecl_idx] == "H"
        assert 0 <= mask_idx <= 7

    def test_v7_still_works(self):
        """V7 is the first version with version info, but format info still works."""
        matrix = make_qr_bitmatrix("VERSION 7 TEST", version=7, ecl="Q")
        ecl_idx, _ = decode_format_info(matrix, 7)
        assert ECL_NAMES[ecl_idx] == "Q"


class TestDecodeFormatInfoConstants:
    """Verify exported constants."""

    def test_format_info_mask_is_correct(self):
        assert FORMAT_INFO_MASK == 0x5412

    def test_max_hamming_distance(self):
        assert MAX_HAMMING_DISTANCE == 3
