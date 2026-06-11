"""Tests for Phase 8 — Top-level Decoder.

Exercises the full decode() pipeline: format info → codeword extraction →
de-interleaving → RS correction → bit-stream decoding → text.

Uses the qrcode library to generate known-good QR codes and verifies
round-trip correctness.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from qr_reader.decoder.decoder import DecodeError, decode
from qr_reader.decoder.tables import ECL_NAMES, VERSIONS

# ──────────────────────────────────────────────────────────────
# Helper — generate QR bit matrix using qrcode library
# ──────────────────────────────────────────────────────────────


def _make_qr(
    content: str,
    version: int | None = None,
    ecl: str = "L",
    mask: int | None = None,
) -> np.ndarray:
    """Return a bool 2D array of QR modules (True = dark)."""
    import qrcode

    ecl_consts = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    extra = {}
    if version is not None:
        extra["version"] = version
    if mask is not None:
        extra["mask_pattern"] = mask

    qr = qrcode.QRCode(
        error_correction=ecl_consts[ecl],
        border=0,
        **extra,
    )
    qr.add_data(content)
    qr.make(fit=False)
    return np.array(qr.modules, dtype=bool)


# ──────────────────────────────────────────────────────────────
# Round-trip tests — basic
# ──────────────────────────────────────────────────────────────


class TestDecodeRoundTrip:
    """Full decode pipeline round-trip tests."""

    def test_v1_L_hello(self):
        matrix = _make_qr("HELLO", version=1, ecl="L")
        assert decode(matrix) == "HELLO"

    def test_v1_M_hello_world(self):
        matrix = _make_qr("HELLO WORLD", version=1, ecl="M")
        assert decode(matrix) == "HELLO WORLD"

    def test_v1_Q_numeric(self):
        matrix = _make_qr("0123456789", version=1, ecl="Q")
        assert decode(matrix) == "0123456789"

    def test_v1_H_short(self):
        matrix = _make_qr("HI", version=1, ecl="H")
        assert decode(matrix) == "HI"

    def test_v3_L(self):
        content = "QR TEST DATA!"
        matrix = _make_qr(content, version=3, ecl="L")
        assert decode(matrix) == content

    def test_v7_L(self):
        content = "VERSION 7 TEST DATA!"
        matrix = _make_qr(content, version=7, ecl="L")
        assert decode(matrix) == content

    def test_v7_M(self):
        content = "VERSION 7 TEST DATA!"
        matrix = _make_qr(content, version=7, ecl="M")
        assert decode(matrix) == content

    def test_v10_L(self):
        content = "V10 QR TEST DATA"
        matrix = _make_qr(content, version=10, ecl="L")
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# Numeric mode tests
# ──────────────────────────────────────────────────────────────


class TestDecodeNumeric:
    """Tests with purely numeric content."""

    def test_small_number(self):
        content = "42"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content

    def test_long_number(self):
        content = "9" * 10
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content

    def test_zero_leading(self):
        content = "00123456"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# Alphanumeric mode tests
# ──────────────────────────────────────────────────────────────


class TestDecodeAlphanumeric:
    """Tests with alphanumeric content."""

    def test_short_string(self):
        matrix = _make_qr("HELLO", version=1, ecl="L")
        assert decode(matrix) == "HELLO"

    def test_with_space(self):
        matrix = _make_qr("HELLO WORLD", version=1, ecl="L")
        assert decode(matrix) == "HELLO WORLD"

    def test_url_like(self):
        content = "HTTPS://EXAMPLE.COM/PATH"
        matrix = _make_qr(content, version=2, ecl="L")
        assert decode(matrix) == content

    def test_special_chars(self):
        content = "ABC: $%*+-./"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# Byte mode tests
# ──────────────────────────────────────────────────────────────


class TestDecodeByte:
    """Tests with byte-encoded content (Latin-1)."""

    def test_ascii(self):
        content = "Hello World"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content

    def test_lowercase(self):
        content = "hello, world!"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content

    def test_punctuation(self):
        content = "Test@123!"
        matrix = _make_qr(content, version=1, ecl="L")
        assert decode(matrix) == content

    def test_longer_text(self):
        content = "The quick brown fox jumps over the lazy dog."
        matrix = _make_qr(content, version=3, ecl="L")
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# All ECL levels
# ──────────────────────────────────────────────────────────────


class TestAllECLLevels:
    """Round-trip through all four ECL levels."""

    @pytest.mark.parametrize("ecl", ["L", "M", "Q", "H"])
    def test_v3_all_ecl(self, ecl):
        content = "QR " + ecl
        matrix = _make_qr(content, version=3, ecl=ecl)
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# All 8 mask patterns
# ──────────────────────────────────────────────────────────────


class TestAllMasks:
    """Round-trip through all mask patterns (0–7)."""

    @pytest.mark.parametrize("mask", list(range(8)))
    def test_v1_all_masks(self, mask):
        content = "MASK"
        matrix = _make_qr(content, version=1, ecl="H", mask=mask)
        assert decode(matrix) == content

    @pytest.mark.parametrize("mask", list(range(8)))
    def test_v7_all_masks(self, mask):
        content = "MASK"
        matrix = _make_qr(content, version=7, ecl="H", mask=mask)
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# Version info tests (v1–v6 without, v7+ with)
# ──────────────────────────────────────────────────────────────


class TestVersionInfoIntegration:
    """Tests that cover the version-info presence boundary (v6 → v7)."""

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
    def test_v1_v6_no_version_info(self, version):
        content = f"V{version}"
        matrix = _make_qr(content, version=version, ecl="L")
        assert decode(matrix) == content

    @pytest.mark.parametrize("version", [7, 8, 10, 14, 20])
    def test_v7_plus_with_version_info(self, version):
        content = f"V{version}"
        matrix = _make_qr(content, version=version, ecl="L")
        assert decode(matrix) == content


# ──────────────────────────────────────────────────────────────
# Sampled version/ECL combinations
# ──────────────────────────────────────────────────────────────


class TestSampledVersionECL:
    """Representative version × ECL combinations.

    Not exhaustive (that would be 160), but covers each version class
    and ECL pairing at least once.
    """

    # (version, ecl) pairs sampled across the range
    SAMPLES = [
        (1, "L"),
        (1, "H"),
        (2, "M"),
        (3, "Q"),
        (4, "L"),
        (5, "M"),
        (6, "H"),
        (7, "L"),
        (7, "Q"),
        (8, "M"),
        (9, "H"),
        (10, "L"),
        (11, "Q"),
        (12, "M"),
        (13, "L"),
        (14, "H"),
        (15, "Q"),
        (16, "L"),
        (17, "M"),
        (18, "H"),
        (19, "Q"),
        (20, "L"),
        (25, "M"),
        (30, "Q"),
        (35, "H"),
        (40, "L"),
    ]

    @pytest.mark.parametrize("version, ecl", SAMPLES)
    def test_roundtrip(self, version, ecl):
        content = f"V{version} {ecl}"
        matrix = _make_qr(content, version=version, ecl=ecl)
        result = decode(matrix)
        assert result == content


# ──────────────────────────────────────────────────────────────
# Error correction tests — introduce bit flips
# ──────────────────────────────────────────────────────────────


class TestErrorCorrection:
    """Tests that verify RS error correction in the full pipeline."""

    def test_single_bit_flip_low_ecl(self):
        """A single bit flip in a QR code should be correctable."""
        content = "HELLO"
        matrix = _make_qr(content, version=1, ecl="L")

        # Flip a data-region bit (not in a function pattern area)
        corrupted = matrix.copy()
        # v1 is 21×21. Position (12, 12) is safely in the data region.
        corrupted[12, 12] = not corrupted[12, 12]

        result = decode(corrupted)
        assert result == content

    def test_single_bit_flip_high_ecl(self):
        """High ECL should handle a flipped bit easily."""
        content = "EC TEST"
        matrix = _make_qr(content, version=3, ecl="H")

        corrupted = matrix.copy()
        corrupted[20, 20] = not corrupted[20, 20]

        result = decode(corrupted)
        assert result == content

    def test_multiple_bit_flips(self):
        """Multiple bit flips within ECL capacity should be corrected."""
        content = "ERROR CORRECTION TEST"
        matrix = _make_qr(content, version=5, ecl="H")

        size = matrix.shape[0]

        # Use the decoder's own function-module mask to identify data modules.
        from qr_reader.decoder.codeword_extractor import _build_function_module_mask

        fn_mask = _build_function_module_mask(size, 5)
        data_positions = [
            (r, c) for r in range(size) for c in range(size) if not fn_mask[r, c]
        ]
        assert len(data_positions) > 10, "Need at least some data modules"

        corrupted = matrix.copy()
        # Flip 3 well-separated data modules
        rng = random.Random(42)
        flip_positions = rng.sample(data_positions, 3)
        for r, c in flip_positions:
            corrupted[r, c] = not corrupted[r, c]

        result = decode(corrupted)
        assert result == content


# ──────────────────────────────────────────────────────────────
# Error / edge-case tests
# ──────────────────────────────────────────────────────────────


class TestDecodeErrors:
    """Tests for error handling in the top-level decoder."""

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            decode(np.zeros((21, 22), dtype=bool))

    def test_1d_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            decode(np.zeros(21, dtype=bool))

    def test_invalid_size_raises(self):
        # size that doesn't match any valid version
        with pytest.raises(ValueError, match="Invalid QR symbol size"):
            decode(np.zeros((22, 22), dtype=bool))

    def test_version_out_of_range_raises(self):
        # size = 17 + 4*41 = 181 → version 41 (invalid)
        with pytest.raises(ValueError, match="Invalid QR symbol size"):
            decode(np.zeros((185, 185), dtype=bool))

    def test_garbage_matrix_raises_decode_error(self):
        """A matrix of all zeros should fail to decode format info."""
        with pytest.raises(DecodeError, match="Format info"):
            decode(np.zeros((21, 21), dtype=bool))

    def test_all_ones_matrix_raises_decode_error(self):
        """A matrix of all ones should also fail format info decode."""
        with pytest.raises(DecodeError, match="Format info"):
            decode(np.ones((21, 21), dtype=bool))
