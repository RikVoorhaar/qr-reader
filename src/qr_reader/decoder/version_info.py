"""Version Information BCH decoding for QR Code versions 7–40.

Reads the 18-bit version information from two locations in the bit matrix
and decodes it by hamming-distance matching against the 34 valid BCH(18,6)
patterns.

Reference:
- zxing-cpp QRVersion.cpp: DecodeVersionInformation()
- nayuki qrcodegen.py: _draw_version()
"""

from __future__ import annotations

from qr_reader.decoder.tables import VERSION_INFO_PATTERNS


def read_version_bits(bit_matrix) -> tuple[int, int]:
    """Read the two 18-bit version info copies from the bit matrix.

    The bit matrix is a 2D array with bit_matrix[row, col] access (True = dark).

    Copy A (top-right): columns (dim-11) to (dim-9), rows 0 to 5.
    Copy B (bottom-left): columns 0 to 5, rows (dim-11) to (dim-9).

    Bit ordering (matching nayuki _draw_version):
        i=0  → LSB, row=0, col=dim-11
        i=17 → MSB, row=5, col=dim-9

    Args:
        bit_matrix: 2D bool array, bit_matrix[row, col].

    Returns:
        (bits_a, bits_b) — two 18-bit integers.
    """
    dim: int = bit_matrix.shape[0]
    bits_a = 0
    bits_b = 0

    for i in range(18):
        row: int = i // 3
        col: int = dim - 11 + i % 3

        if bit_matrix[row, col]:
            bits_a |= 1 << i

        # Transposed for copy B
        if bit_matrix[col, row]:
            bits_b |= 1 << i

    return bits_a, bits_b


def decode_version(bits_a: int, bits_b: int) -> int | None:
    """BCH-decode version information from two 18-bit copies.

    Performs hamming-distance matching against the 34 valid BCH(18,6) patterns.
    Checks both copies and picks the one with the smallest distance.
    If the best distance ≤ 3, returns the version number (7–40).
    Otherwise returns None.

    Reference: zxing-cpp QRVersion.cpp DecodeVersionInformation()

    Args:
        bits_a: 18-bit version info from top-right copy.
        bits_b: 18-bit version info from bottom-left copy.

    Returns:
        Version number (7–40) on success, None on failure.
    """
    best_distance: int = 999
    best_version: int | None = None

    for v, target in enumerate(VERSION_INFO_PATTERNS, start=7):
        for bits in (bits_a, bits_b):
            # Hamming distance = popcount(bits XOR target)
            diff: int = (bits ^ target).bit_count()
            if diff < best_distance:
                best_distance = diff
                best_version = v
                if best_distance == 0:
                    break

    # Two valid version info codewords differ in at least 8 bits,
    # so up to 3 bit errors can be reliably corrected.
    if best_distance <= 3:
        return best_version
    return None


def decode_version_info(bit_matrix) -> int | None:
    """Read and decode version information from a bit matrix.

    Convenience function that combines read_version_bits + decode_version.
    Only applicable for versions ≥ 7; returns None for smaller matrices.

    Args:
        bit_matrix: 2D bool array, bit_matrix[row, col].

    Returns:
        Version number (7–40) on success, None on failure or if version < 7.
    """
    dim: int = bit_matrix.shape[0]
    # Version 6 → symbol size 17+4*6 = 41. Version 7 → 17+4*7 = 45.
    # Version info region starts at dim-11, so we need dim ≥ 45 for v7.
    if dim < 45:
        return None
    bits_a, bits_b = read_version_bits(bit_matrix)
    return decode_version(bits_a, bits_b)
