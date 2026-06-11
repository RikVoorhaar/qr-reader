"""Codeword Extraction from QR Code Bit Matrix (Phase 4).

Walks the QR code in the zigzag reading order, skips function modules,
unmasks data modules (XOR with mask pattern), and packs bits into
codeword bytes (MSB-first within each byte).

Reference:
- ISO 18004:2015 § 7.7.3
- zxing-cpp QRBitMatrixParser.cpp
"""

from __future__ import annotations

import numpy as np

from qr_reader.decoder.tables import VERSIONS, _mask_condition


def _build_function_module_mask(size: int, version: int) -> np.ndarray:
    """Return a bool matrix where True = function module (NOT a data module).

    Function modules include:
    - Finder patterns (3 copies, 8×8 each, including separators)
    - Timing patterns (row 6 horizontal, col 6 vertical)
    - Alignment patterns (5×5 each)
    - Format information modules
    - Version information modules (versions ≥ 7)
    - Dark module at (8, size-8) in format info area

    Args:
        size: Symbol size (modules per side).
        version: QR code version (1–40).

    Returns:
        2D numpy bool array, shape (size, size), True where the module is
        occupied by a function pattern.
    """
    mask = np.zeros((size, size), dtype=bool)

    # ── Finder patterns (3 copies at corners) ──
    # Each finder pattern is 7×7 actual modules plus a 1-module separator
    # border, making a 9×9 occupied area.  The encoder sets all modules
    # in this 9×9 region (the "separators" are explicitly set to False).
    # Top-left, top-right, bottom-left.
    for fr, fc in [(0, 0), (0, size - 7), (size - 7, 0)]:
        r0 = max(0, fr - 1)
        c0 = max(0, fc - 1)
        r1 = min(size, fr + 8)
        c1 = min(size, fc + 8)
        mask[r0:r1, c0:c1] = True

    # ── Timing patterns ──
    # Horizontal: row 6, cols 8 .. size-9 inclusive (range(8, size-8))
    mask[6, 8 : size - 8] = True
    # Vertical: col 6, rows 8 .. size-9 inclusive
    mask[8 : size - 8, 6] = True

    # ── Alignment patterns ──
    # The encoder places alignment patterns at all positions in the list,
    # but it skips any whose centre module is already occupied by a finder
    # pattern (the ``if self.modules[row][col] is not None: continue``
    # check).  We replicate that: mark only alignments whose centre is NOT
    # inside a finder pattern area.
    positions = VERSIONS[version].alignment_positions
    # Finder pattern areas: 9×9 regions at the three corners.
    finder_areas = [
        (0, 0, min(size, 8), min(size, 8)),  # top-left
        (0, max(0, size - 8), min(size, 8), size),  # top-right
        (max(0, size - 8), 0, size, min(size, 8)),  # bottom-left
    ]
    for ar in positions:
        for ac in positions:
            in_finder = False
            for fr0, fc0, fr1, fc1 in finder_areas:
                if fr0 <= ar < fr1 and fc0 <= ac < fc1:
                    in_finder = True
                    break
            if in_finder:
                continue
            r0, c0 = ar - 2, ac - 2
            r1, c1 = ar + 3, ac + 3
            r0_clip = max(0, r0)
            c0_clip = max(0, c0)
            r1_clip = min(size, r1)
            c1_clip = min(size, c1)
            mask[r0_clip:r1_clip, c0_clip:c1_clip] = True

    # ── Format information modules ──
    # Top-left copy (around the finder pattern)
    # The format bits at (0,8) through (8,8) and (8,0) through (8,8)
    # Bit positions per ISO 18004:2015 Figure 25:
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
        mask[y, x] = True

    # Bottom-left / top-right copy
    for i in range(8):
        mask[8, size - 1 - i] = True  # top-right row
        mask[size - 8 + i, 8] = True  # bottom-left column

    # Dark module at (8, size-8) in format info (already set above)

    # ── Version information modules (versions ≥ 7) ──
    if version >= 7:
        # Top-right copy: rows 0..5, cols size-11 .. size-9 (6×3)
        for i in range(18):
            r = i // 3
            c = size - 11 + i % 3
            mask[r, c] = True
        # Bottom-left copy: rows size-11 .. size-9, cols 0..5 (3×6)
        for i in range(18):
            r = size - 11 + i % 3
            c = i // 3
            mask[r, c] = True

    return mask


def extract_codewords(matrix: np.ndarray, version: int, mask_idx: int) -> bytes:
    """Extract raw codeword bytes from the bit matrix.

    Args:
        matrix: 2D numpy bool array (True = dark, False = light),
                shape (size, size).
        version: QR code version (1–40).
        mask_idx: Mask pattern index (0–7).

    Returns:
        Raw codeword bytes (data + EC, interleaved).

    Raises:
        ValueError: if version or mask_idx is invalid, or if the extracted
                    codeword count doesn't match the expected total.
    """
    size = 17 + 4 * version

    if matrix.shape != (size, size):
        raise ValueError(
            f"Matrix shape {matrix.shape} != expected ({size}, {size}) "
            f"for version {version}"
        )

    if not (0 <= mask_idx <= 7):
        raise ValueError(f"Invalid mask index: {mask_idx}")

    expected_total = _compute_total_codewords(version)

    fn_mask = _build_function_module_mask(size, version)

    expected_bits = expected_total * 8

    # Walk the zigzag pattern
    bits: list[int] = []  # collect bits MSB-first

    # Start at the bottom-right corner
    row = size - 1
    row_inc = -1  # going UP initially

    # Iterate over column pairs from right to left
    col = size - 1
    while col > 0:
        # Skip the vertical timing pattern column (column 6)
        if col == 6:
            col -= 1

        # Each iteration processes a pair of columns: col, col-1
        c0, c1 = col, col - 1

        while True:
            for c in (c0, c1):
                if 0 <= row < size and 0 <= c < size:
                    if not fn_mask[row, c]:
                        # Data module: read bit
                        dark = bool(matrix[row, c])
                        # Unmask: XOR with mask condition
                        if _mask_condition(mask_idx, row, c):
                            dark = not dark
                        bits.append(1 if dark else 0)
                        if len(bits) == expected_bits:
                            # Stop once we have all expected bits (ignore
                            # remainder bits that pad the symbol capacity).
                            col = 0
                            break

            if col == 0:
                break

            row += row_inc

            if row < 0 or row >= size:
                # Reverse direction
                row -= row_inc
                row_inc = -row_inc
                break

        col -= 2

    # Pack bits into bytes (MSB-first within each byte)
    codewords = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits) and bits[i + j]:
                byte_val |= 1 << (7 - j)
            # unfilled bits in the last byte are 0
        codewords.append(byte_val)

    if len(codewords) != expected_total:
        raise ValueError(
            f"Extracted {len(codewords)} codewords, expected {expected_total} "
            f"for version {version}"
        )

    return bytes(codewords)


def _compute_total_codewords(version: int) -> int:
    """Return the total number of codewords (data + EC) in the QR symbol.

    This is the same for all ECL levels of the same version, since the
    data region size is fixed per version. We sum (data + EC) for "L".
    """
    ec_per_block, groups = VERSIONS[version].ec_info["L"]
    total = 0
    for data_bytes, num_blocks in groups:
        total += num_blocks * (data_bytes + ec_per_block)
    return total
