"""Tests for Phase 7 — Bit Stream Decoding.

Decode the data codewords (after de-interleaving and RS correction) back into
the original text string.  Uses v1 QR codes (single EC block) to keep things simple.
"""

import pytest

from qr_reader.decoder.bitstream import (
    BitBuffer,
    data_codewords,
    decode_bitstream,
)

# ──────────────────────────────────────────────────────────────
# Helper — generate data codewords using the qrcode library
# ──────────────────────────────────────────────────────────────


def _make_data_codewords(
    content: str,
    version: int = 1,
    ecl: str = "L",
) -> list[int]:
    """Return the data codewords (no EC) for a QR code with *content*.

    Only works correctly for single-block codes (v1–L, v1–M, v1–Q, v1–H, etc.).
    For multi-block codes the interleaving would scramble the bitstream.
    """
    import qrcode

    ecl_map = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(
        version=version,
        error_correction=ecl_map[ecl],
    )
    qr.add_data(content)
    qr.make()

    dc = data_codewords(version, ecl)
    return list(qr.data_cache[:dc])


# ──────────────────────────────────────────────────────────────
# BitBuffer tests
# ──────────────────────────────────────────────────────────────


class TestBitBuffer:
    def test_available(self):
        bb = BitBuffer([0xAA, 0x55])
        assert bb.available() == 16

    def test_read_all_bits(self):
        bb = BitBuffer([0x80])  # 10000000
        assert bb.read_bits(1) == 1
        assert bb.read_bits(7) == 0
        assert bb.available() == 0

    def test_read_multibyte(self):
        bb = BitBuffer([0x11, 0x22])  # 00010001 00100010
        val = bb.read_bits(12)
        assert val == 0x112

    def test_read_zero_bits(self):
        bb = BitBuffer([0xFF])
        assert bb.read_bits(0) == 0
        assert bb.available() == 8

    def test_read_past_end(self):
        bb = BitBuffer([0xFF])
        val = bb.read_bits(20)
        assert val == 0xFF
        assert bb.available() == 0

    def test_peek_does_not_consume(self):
        bb = BitBuffer([0xAB, 0xCD])
        peeked = bb.peek_bits(8)
        pos_before = bb.available()
        read = bb.read_bits(8)
        assert peeked == read
        assert pos_before == 16

    def test_read_across_byte_boundary(self):
        # 0xFF = 11111111, 0x00 = 00000000
        # Read 9 bits: 11111111 0 = 0x1FE
        bb = BitBuffer([0xFF, 0x00])
        assert bb.read_bits(9) == 0x1FE

    def test_empty_buffer(self):
        bb = BitBuffer([])
        assert bb.available() == 0
        assert bb.read_bits(8) == 0


# ──────────────────────────────────────────────────────────────
# Decode tests — numeric
# ──────────────────────────────────────────────────────────────


class TestDecodeNumeric:
    def test_10_digits(self):
        content = "0123456789"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_11_digits(self):
        """11 digits — 3 groups of 3 + 2-digit remainder."""
        content = "01234567890"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_12_digits(self):
        """12 digits — 4 groups of 3."""
        content = "012345678901"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_single_digit(self):
        content = "7"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_two_digits(self):
        content = "42"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_three_digits(self):
        content = "123"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content


# ──────────────────────────────────────────────────────────────
# Decode tests — alphanumeric
# ──────────────────────────────────────────────────────────────


class TestDecodeAlphanumeric:
    def test_short_string(self):
        content = "HELLO"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_with_space(self):
        content = "HELLO WORLD"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_alphanumeric_charset(self):
        """Representative chars from the alphanumeric table."""
        # v1-L fits up to 25 alphanumeric chars. A 20-char subset covers all char
        # classes: digits, uppercase, space, symbols.
        content = "0123456789ABCDEF: $%*+-"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_single_char(self):
        content = "A"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_two_chars(self):
        content = "AB"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_odd_count(self):
        """3 chars — 1 group of 2 + 1 single."""
        content = "ABC"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_url_like(self):
        """URL-like alphanumeric content."""
        content = "HTTPS://EXAMPLE.COM/PATH"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content


# ──────────────────────────────────────────────────────────────
# Decode tests — byte mode
# ──────────────────────────────────────────────────────────────


class TestDecodeByte:
    def test_ascii(self):
        content = "Hello World"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_lowercase(self):
        content = "hello, world!"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_punctuation(self):
        content = "Test@123!"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_single_char(self):
        content = "X"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content


# ──────────────────────────────────────────────────────────────
# Mixed-mode test (the qrcode library may auto-select modes)
# ──────────────────────────────────────────────────────────────


class TestDecodeMixed:
    def test_numeric_alphanumeric_boundary(self):
        """Content that the library may encode as numeric."""
        content = "42"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content

    def test_alphanumeric_boundary(self):
        content = "ABC 123"
        data = _make_data_codewords(content)
        assert decode_bitstream(data, 1) == content


# ──────────────────────────────────────────────────────────────
# Error / edge-case tests
# ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_terminator_stops_parsing(self):
        """Feed bytes where a terminator follows the data. Should stop cleanly."""
        content = "0123456789"
        data = _make_data_codewords(content)
        result = decode_bitstream(data, 1)
        assert result == content

    def test_unrecognized_mode(self):
        """Bytes starting with an unknown mode indicator should raise."""
        # Mode indicator 0x3 is not defined in the spec
        with pytest.raises(ValueError, match="Unrecognised mode indicator"):
            decode_bitstream([0x30], 1)

    def test_short_data(self):
        """Data that ends before a full segment."""
        # Single byte with mode = byte (0x4) in top 4 bits, no room for count+data
        # byte mode for v1: 4 bits mode + 8 bits count = 12 bits needed, but we have 8
        # The decoder should handle this gracefully by breaking out
        data = [0x40]  # 01000000 — mode=4 (byte), then only 4 bits left
        result = decode_bitstream(data, 1)
        # Should not crash; returns what it can
        assert isinstance(result, str)

    def test_ec_codewords_count(self):
        """Verify data_codewords matches known v1 values."""
        assert data_codewords(1, "L") == 19
        assert data_codewords(1, "M") == 16
        assert data_codewords(1, "Q") == 13
        assert data_codewords(1, "H") == 9


# ──────────────────────────────────────────────────────────────
# Version range character count test
# ──────────────────────────────────────────────────────────────


class TestCharCountBits:
    def test_v1_numeric_count_bits(self):
        """v1 uses 10-bit character count for numeric mode."""
        content = "9" * 10
        data = _make_data_codewords(content, version=1)
        assert decode_bitstream(data, 1) == content
