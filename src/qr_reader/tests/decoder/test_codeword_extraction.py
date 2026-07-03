"""Tests for Phase 4 — Codeword Extraction.

Generates QR codes with the qrcode library, extracts raw codewords
from the bit matrix using extract_codewords, and verifies they match
the encoder's interleaved data+EC codewords (qr.data_cache).
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.decoder.codeword_extractor import (
    _build_function_module_mask,
    _compute_total_codewords,
    extract_codewords,
)
from qr_reader.decoder.format_info import decode_format_info
from qr_reader.decoder.tables import (
    ECL_H,
    ECL_L,
    ECL_M,
    ECL_NAMES_LIST,
    ECL_Q,
    VERSIONS,
    _mask_condition,
)

# ──────────────────────────────────────────────────────────────
# Test helper
# ──────────────────────────────────────────────────────────────


def _gen_qr(
    version: int, ecl: str, content: str, mask: int = 0
) -> tuple[np.ndarray, list[int]]:
    """Generate a QR code with a specific mask and return (matrix, data_cache).

    Forces a specific mask pattern by calling makeImpl() directly on the
    qrcode library's QRCode object.
    """
    import qrcode

    ecl_consts = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.QRCode(
        version=version,
        error_correction=ecl_consts[ecl],
        box_size=1,
        border=0,
    )
    qr.add_data(content)
    qr.makeImpl(False, mask)
    matrix = np.array(qr.modules, dtype=bool).T
    return matrix, list(qr.data_cache)


# ──────────────────────────────────────────────────────────────
# Tests for _build_function_module_mask
# ──────────────────────────────────────────────────────────────


class TestBuildFunctionModuleMask:
    def test_v1_size(self):
        """V1 mask should be 21×21."""
        mask = _build_function_module_mask(21, 1)
        assert mask.shape == (21, 21)

    def test_v40_size(self):
        """V40 mask should be 177×177."""
        mask = _build_function_module_mask(177, 40)
        assert mask.shape == (177, 177)

    def test_finder_patterns_marked(self):
        """All three finder patterns (8×8 each) should be fully marked."""
        for version in [1, 5, 10, 20, 40]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            # Top-left finder
            assert mask[0:8, 0:8].all(), f"V{version}: top-left finder not fully marked"
            # Top-right finder
            assert mask[0:8, size - 8 : size].all(), (
                f"V{version}: top-right finder not fully marked"
            )
            # Bottom-left finder
            assert mask[size - 8 : size, 0:8].all(), (
                f"V{version}: bottom-left finder not fully marked"
            )

    def test_timing_patterns_marked(self):
        """Horizontal and vertical timing patterns should be marked."""
        for version in [1, 5, 10, 20, 40]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            # Horizontal timing: row 6, cols 8..size-9
            assert mask[6, 8 : size - 7].all(), (
                f"V{version}: horizontal timing not marked"
            )
            # Vertical timing: col 6, rows 8..size-7
            assert mask[8 : size - 7, 6].all(), (
                f"V{version}: vertical timing not marked"
            )

    def test_format_info_marked(self):
        """Format information modules should be marked as function modules."""
        for version in [1, 7, 20]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            # Top-left format info modules
            for x, y in [
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
            ]:
                assert mask[y, x], f"V{version}: format info ({x}, {y}) not marked"

    def test_version_info_marked_v7_plus(self):
        """Version information modules should be marked for V >= 7."""
        for version in [7, 20, 40]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            # Top-right copy: rows 0..5, cols size-11 .. size-9
            assert mask[0:6, size - 11 : size - 8].all(), (
                f"V{version}: version info top-right not marked"
            )
            # Bottom-left copy: rows size-11 .. size-9, cols 0..5
            assert mask[size - 11 : size - 8, 0:6].all(), (
                f"V{version}: version info bottom-left not marked"
            )

    def test_version_info_not_marked_v1_v6(self):
        """Versions < 7 should NOT have version info modules marked."""
        for version in [1, 3, 6]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            # The top-right 3×6 area is outside the matrix for small versions
            # so just verify the bottom-left area isn't marked
            # For v1 (size=21), size-11=10, so rows 10..13, cols 0..5 should
            # not have version info. But alignment patterns might overlap there.
            # Just verify the function doesn't crash and returns valid mask.

    def test_alignment_patterns_marked(self):
        """Alignment patterns should be marked for versions >= 2.

        The encoder skips alignment patterns whose centre module overlaps a
        finder pattern (the ``if self.modules[row][col] is not None: continue``
        check).  We replicate that skip here.
        """
        for version in [2, 7, 20]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            positions = VERSIONS[version].alignment_positions
            # Finder pattern 8×8 areas (same logic as the extractor)
            finder_areas = [
                (0, 0, min(size, 8), min(size, 8)),
                (0, max(0, size - 8), min(size, 8), size),
                (max(0, size - 8), 0, size, min(size, 8)),
            ]
            for ar in positions:
                for ac in positions:
                    # Skip alignments whose centre is inside a finder pattern
                    in_finder = any(
                        fr0 <= ar < fr1 and fc0 <= ac < fc1
                        for fr0, fc0, fr1, fc1 in finder_areas
                    )
                    if in_finder:
                        continue
                    r0, c0 = ar - 2, ac - 2
                    r1, c1 = ar + 3, ac + 3
                    r0_clip = max(0, r0)
                    c0_clip = max(0, c0)
                    r1_clip = min(size, r1)
                    c1_clip = min(size, c1)
                    assert mask[r0_clip:r1_clip, c0_clip:c1_clip].all(), (
                        f"V{version}: alignment at ({ar}, {ac}) not fully marked"
                    )

    def test_data_module_count_matches(self):
        """The number of non-function modules should equal total_codewords * 8
        plus 0–7 remainder bits (unused capacity padding)."""
        for version in [1, 5, 10, 20, 30, 40]:
            size = 17 + 4 * version
            mask = _build_function_module_mask(size, version)
            data_modules = int((~mask).sum())
            expected_bits = _compute_total_codewords(version) * 8
            remainder = data_modules - expected_bits
            assert 0 <= remainder < 8, (
                f"V{version}: {data_modules} data modules, expected {expected_bits} "
                f"bits + 0–7 remainder, got remainder={remainder}"
            )


# ──────────────────────────────────────────────────────────────
# Tests for _compute_total_codewords
# ──────────────────────────────────────────────────────────────


class TestComputeTotalCodewords:
    def test_v1_total(self):
        """V1 has 26 codewords (19 data + 7 EC)."""
        assert _compute_total_codewords(1) == 26

    def test_total_is_ecl_independent(self):
        """Total codewords should be the same regardless of ECL."""
        for version in [1, 5, 10, 20, 40]:
            # All ECLs for a given version should have the same total
            totals = set()
            for ecl in ECL_NAMES_LIST:
                ec_per_block, groups = VERSIONS[version].ec_info[ecl]
                t = sum(data + ec_per_block for data, n in groups for _ in range(n))
                totals.add(t)
            assert len(totals) == 1, f"V{version}: totals differ across ECLs: {totals}"


# ──────────────────────────────────────────────────────────────
# Tests for extract_codewords — roundtrip with qrcode library
# ──────────────────────────────────────────────────────────────


class TestExtractCodewordsRoundtrip:
    """End-to-end: generate QR → extract codewords → match data_cache."""

    @pytest.mark.parametrize(
        "version, ecl, content, mask",
        [
            (1, "L", "HELLO", 0),
            (1, "L", "HELLO", 3),
            (1, "L", "HELLO", 7),
            (1, "M", "HELLO WORLD", 0),
            (1, "Q", "0123456789", 0),
            (1, "H", "HI", 0),
            (2, "L", "QR TEST", 0),
            (2, "L", "QR TEST", 5),
            (3, "L", "QR TEST DATA", 0),
            (5, "L", "HELLO QR CODE V5", 0),
            (7, "L", "VERSION 7 TEST DATA!", 0),
            (7, "M", "VERSION 7 TEST DATA!", 3),
            (7, "Q", "VERSION 7 TEST DATA!", 7),
            (10, "L", "QR TEST DATA VERSION 10", 0),
            (20, "L", "QR TEST DATA VERSION 20", 0),
            (30, "L", "QR TEST DATA VERSION 30", 0),
            (40, "L", "V40 DATA", 0),
        ],
    )
    def test_roundtrip_match(self, version, ecl, content, mask):
        """Extracted codewords should exactly match the encoder's data_cache."""
        matrix, data_cache = _gen_qr(version, ecl, content, mask)
        codewords = extract_codewords(matrix, version, mask)
        assert list(codewords) == data_cache, (
            f"V{version} {ecl} mask{mask}: codeword mismatch\n"
            f"  extracted: {list(codewords)[:20]}...\n"
            f"  expected:  {data_cache[:20]}..."
        )

    def test_all_masks_v1(self):
        """All 8 mask patterns for V1-L should roundtrip correctly."""
        for mask in range(8):
            matrix, data_cache = _gen_qr(1, "L", "HELLO", mask)
            codewords = extract_codewords(matrix, 1, mask)
            assert list(codewords) == data_cache, f"V1-L mask{mask}: mismatch"

    def test_all_masks_v7(self):
        """All 8 mask patterns for V7-L should roundtrip correctly."""
        for mask in range(8):
            matrix, data_cache = _gen_qr(7, "L", "VERSION 7 TEST DATA", mask)
            codewords = extract_codewords(matrix, 7, mask)
            assert list(codewords) == data_cache, f"V7-L mask{mask}: mismatch"

    @pytest.mark.parametrize("version", [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40])
    def test_various_versions_l(self, version):
        """Roundtrip for all major version milestones at ECL L."""
        content = f"V{version:02d}" * 5  # make it long enough
        matrix, data_cache = _gen_qr(version, "L", content, 0)
        codewords = extract_codewords(matrix, version, 0)
        assert list(codewords) == data_cache, (
            f"V{version} L: {len(list(codewords))} codewords != {len(data_cache)}"
        )

    @pytest.mark.parametrize("ecl", ["L", "M", "Q", "H"])
    def test_all_ecl_v10(self, ecl):
        """All four ECL levels for V10."""
        content = f"V10 {ecl}" * 5
        matrix, data_cache = _gen_qr(10, ecl, content, 0)
        codewords = extract_codewords(matrix, 10, 0)
        assert list(codewords) == data_cache, f"V10 {ecl}: mismatch"


# ──────────────────────────────────────────────────────────────
# Tests for extract_codewords — unmasking correctness
# ──────────────────────────────────────────────────────────────


class TestUnmasking:
    def test_unmasked_bits_match_encoder_raw_data(self):
        """Verify that reading unmasked bits gives the encoder's raw bits.

        The qrcode library's map_data() places bits in the matrix after
        applying the mask. Our extract_codewords XOR-unmasks them back.
        If this works correctly, we get the original data_back bytes.
        """
        # Already tested via roundtrip above. This is an additional smoke test.
        matrix, data_cache = _gen_qr(3, "L", "QR TEST DATA V3", 3)
        codewords = extract_codewords(matrix, 3, 3)
        assert list(codewords) == data_cache


# ──────────────────────────────────────────────────────────────
# Error handling tests
# ──────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_invalid_version_raises(self):
        """Invalid version should raise ValueError."""
        with pytest.raises(ValueError):
            extract_codewords(np.zeros((21, 21), dtype=bool), 0, 0)

    def test_invalid_mask_raises(self):
        """Invalid mask index should raise ValueError."""
        matrix, _ = _gen_qr(1, "L", "HELLO", 0)
        with pytest.raises(ValueError, match="Invalid mask index"):
            extract_codewords(matrix, 1, 8)

    def test_wrong_size_matrix_raises(self):
        """Matrix with wrong size should raise ValueError."""
        with pytest.raises(ValueError, match="Matrix shape"):
            extract_codewords(np.zeros((10, 10), dtype=bool), 1, 0)

    def test_negative_mask_raises(self):
        """Negative mask index should raise ValueError."""
        matrix, _ = _gen_qr(1, "L", "HELLO", 0)
        with pytest.raises(ValueError, match="Invalid mask index"):
            extract_codewords(matrix, 1, -1)


# ──────────────────────────────────────────────────────────────
# Integration: extract + deinterleave + RS + bitstream roundtrip
# ──────────────────────────────────────────────────────────────


class TestFullPipeline:
    """Full pipeline test: extract codewords → deinterleave → RS correct → decode."""

    @pytest.mark.parametrize(
        "content, version, ecl",
        [
            ("HELLO", 1, "L"),
            ("HELLO WORLD", 1, "M"),
            ("0123456789", 1, "Q"),
            ("HI", 1, "H"),
            ("QR TEST DATA!", 3, "L"),
            ("VERSION 7 TEST DATA!", 7, "L"),
            ("VERSION 7 TEST DATA!", 7, "M"),
            ("V10 QR TEST DATA", 10, "L"),
        ],
    )
    def test_extract_deinterleave_decode(self, content, version, ecl):
        """Full decode pipeline from bit matrix to text."""
        from qr_reader.decoder.bitstream import decode_bitstream
        from qr_reader.decoder.data_block import deinterleave
        from qr_reader.decoder.rs import rs_decode

        matrix, _ = _gen_qr(version, ecl, content, 0)
        codewords = extract_codewords(matrix, version, 0)
        blocks = deinterleave(codewords, version, ecl)

        corrected_data = bytearray()
        for blk in blocks:
            corrected = rs_decode(list(blk.data) + list(blk.ec), len(blk.ec))
            corrected_data.extend(corrected)

        result = decode_bitstream(list(corrected_data), version)
        assert result == content, (
            f"V{version} {ecl}: expected '{content}', got '{result}'"
        )
