"""QR Code bit-stream decoding — parse the corrected data codewords into a text string.

Wraps a byte array in a bit reader, then decodes segments by reading
mode indicators, character counts, and encoded data per the QR spec.

Reference: zxing-cpp QRDecoder.cpp DecodeBitStream()
"""

from __future__ import annotations

from qr_reader.decoder.tables import (
    ALPHANUMERIC_VALUE_TO_CHAR,
    CHAR_COUNT_BITS,
    MODE_BY_INDICATOR,
    MODE_INDICATORS,
    char_count_version_range,
)


class BitBuffer:
    """Read-only bit reader over a byte array, MSB-first within each byte."""

    def __init__(self, data: list[int]) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0
        self._total_bits = len(data) * 8

    def available(self) -> int:
        """Return the number of bits remaining."""
        return self._total_bits - (self._byte_pos * 8 + self._bit_pos)

    def read_bits(self, n: int) -> int:
        """Read the next *n* bits as a big-endian integer."""
        if n == 0:
            return 0
        if n > self.available():
            n = self.available()

        result = 0
        for _ in range(n):
            result <<= 1
            if self._data[self._byte_pos] & (0x80 >> self._bit_pos):
                result |= 1
            self._bit_pos += 1
            if self._bit_pos == 8:
                self._bit_pos = 0
                self._byte_pos += 1
        return result

    def peek_bits(self, n: int) -> int:
        """Return the next *n* bits without advancing the position."""
        saved_byte = self._byte_pos
        saved_bit = self._bit_pos
        result = self.read_bits(n)
        self._byte_pos = saved_byte
        self._bit_pos = saved_bit
        return result


def _decode_numeric(bits: BitBuffer, count: int) -> str:
    """Decode *count* numeric digits from the bit stream."""
    result = []
    remaining = count

    # Groups of 3 digits (10 bits)
    while remaining >= 3:
        val = bits.read_bits(10)
        result.append(f"{val:03d}")
        remaining -= 3

    # Group of 2 digits (7 bits)
    if remaining == 2:
        val = bits.read_bits(7)
        result.append(f"{val:02d}")
    elif remaining == 1:
        # Single digit (4 bits)
        val = bits.read_bits(4)
        result.append(str(val))

    return "".join(result)


def _decode_alphanumeric(bits: BitBuffer, count: int) -> str:
    """Decode *count* alphanumeric characters from the bit stream."""
    result = []
    remaining = count

    # Groups of 2 chars (11 bits)
    while remaining >= 2:
        val = bits.read_bits(11)
        c1 = val // 45
        c2 = val % 45
        result.append(ALPHANUMERIC_VALUE_TO_CHAR[c1])
        result.append(ALPHANUMERIC_VALUE_TO_CHAR[c2])
        remaining -= 2

    # Single remaining char (6 bits)
    if remaining == 1:
        val = bits.read_bits(6)
        result.append(ALPHANUMERIC_VALUE_TO_CHAR[val])

    return "".join(result)


def _decode_byte(bits: BitBuffer, count: int) -> str:
    """Decode *count* bytes (ISO 8859-1 / Latin-1) from the bit stream."""
    result = []
    for _ in range(count):
        result.append(chr(bits.read_bits(8)))
    return "".join(result)


def decode_bitstream(data_bytes: list[int], version: int) -> str:
    """Decode a corrected data codeword sequence into a text string.

    Args:
        data_bytes: The data codewords as a list of 8-bit integers
                    (after de-interleaving and Reed-Solomon correction).
        version: QR code version (1–40), used to determine character count
                 indicator bit lengths.

    Returns:
        The decoded text string.

    Raises:
        ValueError: if an unrecognised mode indicator is encountered.
    """
    bits = BitBuffer(data_bytes)
    vr = char_count_version_range(version)
    result: list[str] = []

    while bits.available() >= 4:
        mode_indicator = bits.read_bits(4)

        if mode_indicator == MODE_INDICATORS["terminator"]:
            break

        if mode_indicator not in MODE_BY_INDICATOR:
            raise ValueError(f"Unrecognised mode indicator: 0x{mode_indicator:X}")

        mode_name = MODE_BY_INDICATOR[mode_indicator]

        # Read character count
        count_bits = CHAR_COUNT_BITS[mode_name][vr]
        if count_bits > bits.available():
            break
        char_count = bits.read_bits(count_bits)

        # Decode the segment
        if mode_name == "numeric":
            result.append(_decode_numeric(bits, char_count))
        elif mode_name == "alphanumeric":
            result.append(_decode_alphanumeric(bits, char_count))
        elif mode_name == "byte":
            result.append(_decode_byte(bits, char_count))
        else:
            # Future modes (kanji, etc.) — skip
            # For kanji: 13 bits per char, count is chars
            break

    return "".join(result)


# ──────────────────────────────────────────────────────────────
# Convenience: extract data codewords count for a version+ECL
# ──────────────────────────────────────────────────────────────


def data_codewords(version: int, ecl: str) -> int:
    """Return the number of data codewords (excluding EC) for a given version and ECL."""
    from qr_reader.decoder.tables import VERSIONS

    _ec_per_block, groups = VERSIONS[version].ec_info[ecl]
    total = 0
    for data_bytes, num_blocks in groups:
        total += num_blocks * data_bytes
    return total
