"""Format Information BCH decoding for QR Code Model 2.

Reads the 15-bit format information from the two copies in the bit matrix,
unmasks them with XOR 0x5412, and decodes via Hamming-distance matching
against the 32 valid BCH(15,5) patterns.

Reference: zxing-cpp QRFormatInformation.cpp
"""

from __future__ import annotations

from qr_reader.decoder.tables import FORMAT_PATTERN_TO_INFO, VALID_FORMAT_PATTERNS

FORMAT_INFO_MASK: int = 0x5412
MAX_HAMMING_DISTANCE: int = 3


class FormatInfoDecodeError(Exception):
    """Could not decode format information (all Hamming distances > 3)."""


def _hamming_distance(a: int, b: int) -> int:
    """Return the number of differing bits between two 15-bit integers."""
    return (a ^ b).bit_count()


def _read_format_info_bits(matrix, size: int, location: int) -> int:
    """Read the 15 format-info bits from one of the two locations in the QR code.

    Args:
        matrix: 2D numpy bool array (True = dark).
        size: symbol size (modules per side).
        location: 1 for the top-left copy, 2 for the bottom-left / top-right copy.

    Returns:
        15-bit integer with bits ordered MSB-first as in the spec.
    """
    bits = 0

    if location == 1:
        # Top-left copy (around the finder pattern at row=0..8, col=0..8).
        # Per ISO 18004:2015 Figure 25, bits are read MSB at (8,0) to LSB at (0,8).
        # Bit order (MSB → LSB):
        #   bit 14 (MSB): (8,0)
        #   bit 13:       (8,1)
        #   ...
        #   bit 9:        (8,5)
        #   bit 8:        (8,7)  ← skip timing at (8,6)
        #   bit 7:        (7,8)  ← skip timing at (6,8)
        #   bit 6:        (8,8)
        #   bit 5:        (5,8)
        #   ...
        #   bit 0 (LSB):  (0,8)
        order = [
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (7, 8),  # skip timing at (6, 8)
            (8, 8),
            (8, 7),
            (8, 5),  # skip timing at (8, 6)
            (8, 4),
            (8, 3),
            (8, 2),
            (8, 1),
            (8, 0),
        ]
    else:
        # Location 2: along the bottom-left column and top-right row.
        # Bit order (MSB → LSB):
        #   bit 14 (MSB): (8, size-1)
        #   bit 13:       (8, size-2)
        #   ...
        #   bit 8:        (8, size-7)
        #   bit 7:        (size-8, 8)
        #   bit 6:        (size-7, 8)
        #   ...
        #   bit 0 (LSB):  (size-1, 8)
        # The dark module at (8, size-8) is NOT a format bit; we skip it.
        # Verified against nayuki's _draw_format_bits() and zxing-cpp's
        # ReadFormatInformation() / DecodeQR().
        order = [
            (8, size - 1 - i) for i in range(7)
        ]  # bits 14..8, skipping dark module
        order += [(size - 8 + i, 8) for i in range(8)]  # bits 7..0

    for shift, (x, y) in enumerate(order):
        if matrix[y, x]:
            bits |= 1 << (14 - shift)

    return bits


def decode_format_info(matrix, version: int) -> tuple[int, int]:
    """Decode the error correction level and mask index from the QR bit matrix.

    Reads both copies of the 15-bit format information, unmasks each with
    XOR 0x5412, and finds the valid BCH(15,5) pattern with the smallest
    Hamming distance.

    Args:
        matrix: 2D numpy bool array (True = dark), shape (size, size).
        version: QR code version number (1–40).

    Returns:
        (ecl_index, mask_index) where ecl_index is ECL_L/ECL_M/ECL_Q/ECL_H
        and mask_index is 0–7.

    Raises:
        FormatInfoDecodeError: if the best match has Hamming distance > 3.
        ValueError: if version is invalid.
    """
    # Symbol size
    size = 17 + 4 * version

    # Read raw masked bits from both locations
    raw1 = _read_format_info_bits(matrix, size, location=1)
    raw2 = _read_format_info_bits(matrix, size, location=2)

    # Note: VALID_FORMAT_PATTERNS already include the XOR 0x5412 mask.
    # We compare raw readings directly against them (no unmasking needed).

    # Find best match across both copies and all valid patterns
    best_distance = 99
    best_pattern = None

    for pattern in VALID_FORMAT_PATTERNS:
        d = _hamming_distance(raw1, pattern)
        if d < best_distance:
            best_distance = d
            best_pattern = pattern
        d = _hamming_distance(raw2, pattern)
        if d < best_distance:
            best_distance = d
            best_pattern = pattern

    if best_distance > MAX_HAMMING_DISTANCE or best_pattern is None:
        raise FormatInfoDecodeError(
            f"Cannot decode format info: best Hamming distance = {best_distance} "
            f"(raw1=0x{raw1:04x}, raw2=0x{raw2:04x})"
        )

    return FORMAT_PATTERN_TO_INFO[best_pattern]
